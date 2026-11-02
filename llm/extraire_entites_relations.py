import os
import re
import json
import time
import hashlib
import logging
import subprocess
import unicodedata
import shutil

from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY absent du fichier .env")

MODELS = [
    x.strip()
    for x in os.getenv(
        "GROQ_MODELS",
        "llama-3.3-70b-versatile,llama-3.1-8b-instant"
    ).split(",")
    if x.strip()
]

if not MODELS:
    raise RuntimeError("Aucun modèle Groq configuré.")

HDFS_CONTAINER = os.getenv("HDFS_CONTAINER", "namenode")

# Taille des chunks réduite : le palier gratuit Groq limite les
# tokens par minute (TPM), et max_tokens est compté dans la limite.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# Tokens de sortie demandés (réservés par Groq dans le calcul du TPM).
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))

# Tokens supplémentaires pour les modèles à raisonnement (gpt-oss).
REASONING_EXTRA_TOKENS = int(os.getenv("REASONING_EXTRA_TOKENS", "1500"))

# Plafond estimé par requête. Les limites réelles dépendent du modèle
# et du palier : voir console.groq.com/settings/limits.
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "7900"))

# Pause fixe entre deux chunks (secondes). Le limiteur de débit de
# call_groq gère déjà l'attente nécessaire pour respecter le TPM.
CHUNK_PAUSE = float(os.getenv("CHUNK_PAUSE", "2"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_DOCUMENTS_TEST = int(os.getenv("MAX_DOCUMENTS_TEST", "0"))
HDFS_TIMEOUT = int(os.getenv("HDFS_TIMEOUT", "180"))

OLD_MANIFEST = (
    BASE_DIR / os.getenv(
        "OLD_MANIFEST",
        "../crawler/manifest_avant.json"
    )
).resolve()

CURRENT_MANIFEST = (
    BASE_DIR / os.getenv(
        "CURRENT_MANIFEST",
        "../crawler/manifest_actuel.json"
    )
).resolve()

OUTPUT_DIR = BASE_DIR / "extraction"

BACKUP_DIR = OUTPUT_DIR / "backups"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_FILE = OUTPUT_DIR / "entites_maroc.json"
RELATIONS_FILE = OUTPUT_DIR / "relations_maroc.json"
PROGRESS_FILE = OUTPUT_DIR / "progression.json"
FAILURES_FILE = OUTPUT_DIR / "echecs.json"
DOCUMENTS_FILE = OUTPUT_DIR / "documents_traites.json"

# max_retries=0 : les relances sont gérées par call_groq.
client = Groq(api_key=GROQ_API_KEY, max_retries=0, timeout=90.0)


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            OUTPUT_DIR / "extraction.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("GraphRAG-Maroc")


# ============================================================
# 3. OUTILS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value):
    if value is None:
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_name(value):
    value = normalize_text(value).lower()

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = re.sub(r"[^\w\s-]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def make_id(prefix, value):
    digest = hashlib.sha256(
        normalize_name(value).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}_{digest}"


# ============================================================
# 4. JSON ET SAUVEGARDES
# ============================================================

def load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    for encoding in ("utf-8-sig", "utf-16", "utf-8"):

        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)

        except (UnicodeError, json.JSONDecodeError):
            continue

    raise RuntimeError(f"JSON illisible : {path}")


def save_json_atomic(path, data):
    path = Path(path)

    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(temp, path)


def backup_file(path):
    path = Path(path)

    if not path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    destination = BACKUP_DIR / (
        f"{path.stem}_{timestamp}{path.suffix}"
    )

    shutil.copy2(path, destination)

    logger.info("Backup : %s", destination.name)


# ============================================================
# 5. MANIFESTES
# ============================================================

def read_manifest(path):
    data = load_json(path, [])

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("documents", "entries", "files"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Structure de manifeste invalide : {path}")


def manifest_key(entry):
    return (
        entry.get("normalized_url")
        or entry.get("source_url")
        or entry.get("hdfs_path")
    )


def get_new_documents(old_entries, current_entries):
    old_map = {}

    for entry in old_entries:
        key = manifest_key(entry)

        if key:
            old_map[key] = entry.get("sha256")

    result = []

    for entry in current_entries:
        key = manifest_key(entry)

        if not key:
            continue

        old_sha = old_map.get(key)
        current_sha = entry.get("sha256")

        if old_sha != current_sha:
            result.append(entry)

    return result


# ============================================================
# 6. LECTURE HDFS
# ============================================================

def read_hdfs_file(hdfs_path):
    if not hdfs_path:
        raise ValueError("hdfs_path manquant")

    command = [
        "docker",
        "exec",
        HDFS_CONTAINER,
        "hdfs",
        "dfs",
        "-cat",
        hdfs_path
    ]

    logger.info("Lecture HDFS : %s", hdfs_path)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=HDFS_TIMEOUT
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timeout HDFS : {hdfs_path}"
        )

    if result.returncode != 0:
        error = result.stderr.decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(error)

    logger.info(
        "Taille lue : %.2f Mo",
        len(result.stdout) / 1048576
    )

    return result.stdout


# ============================================================
# 7. EXTRACTION HTML
# ============================================================

def extract_html(data):
    html = data.decode(
        "utf-8",
        errors="replace"
    )

    try:
        soup = BeautifulSoup(html, "lxml")

    except Exception:
        logger.warning(
            "Fallback vers html.parser"
        )

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "footer",
        "nav",
        "header"
    ]):
        element.decompose()

    title = ""

    if soup.title:
        title = normalize_text(
            soup.title.get_text(" ", strip=True)
        )

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
    )

    if main:
        text = main.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    return title, normalize_text(text)


# ============================================================
# 8. EXTRACTION PDF
# ============================================================

def extract_pdf(data):
    document = pymupdf.open(
        stream=data,
        filetype="pdf"
    )

    metadata = document.metadata or {}

    title = normalize_text(
        metadata.get("title", "")
    )

    pages = []

    for page_number, page in enumerate(document):

        text = normalize_text(
            page.get_text("text")
        )

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return title, pages


# ============================================================
# 9. EXTRACTION JSON
# ============================================================

def extract_json_text(data):
    try:
        obj = json.loads(
            data.decode("utf-8-sig")
        )

    except Exception:
        return "", []

    strings = []

    def walk(value):

        if isinstance(value, dict):
            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

        elif isinstance(value, str):
            text = normalize_text(value)

            if len(text) >= 20:
                strings.append(text)

    walk(obj)

    title = ""

    if isinstance(obj, dict):
        title = normalize_text(
            obj.get("title")
            or obj.get("titre")
            or obj.get("name")
            or obj.get("nom")
            or ""
        )

    return title, [{
        "page": None,
        "text": "\n".join(strings)
    }]


# ============================================================
# 10. DÉTECTION DU FORMAT
# ============================================================

def extract_document(entry, raw_data):
    file_type = (
        entry.get("file_type") or ""
    ).lower()

    content_type = (
        entry.get("content_type") or ""
    ).lower()

    path = (
        entry.get("hdfs_path") or ""
    ).lower()

    if (
        file_type == "pdf"
        or path.endswith(".pdf")
        or content_type == "application/pdf"
    ):
        return extract_pdf(raw_data)

    if (
        file_type == "json"
        or path.endswith(".json")
        or "json" in content_type
    ):
        return extract_json_text(raw_data)

    title, text = extract_html(raw_data)

    return title, [{
        "page": None,
        "text": text
    }]


# ============================================================
# 11. CHUNKING
# ============================================================

def chunk_text(text):
    text = normalize_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text)
        )

        if end < len(text):

            boundaries = [
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("; ", start, end)
            ]

            boundary = max(boundaries)

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk = text[start:end].strip()

        if len(chunk) >= 80:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1
        )

    return chunks


# ============================================================
# 12. PROMPT SÉMANTIQUE
# ============================================================

SYSTEM_PROMPT = """
Tu es un expert en extraction de connaissances
et en construction de graphes de connaissances.

Tu travailles sur un corpus documentaire consacré au Maroc.

Tu dois extraire les entités et les relations avec
une grande précision sémantique.

RÈGLES :

- Ne jamais inventer une information.
- Ne jamais compléter une relation par supposition.
- Respecter les noms exacts.
- Distinguer les homonymes.
- Respecter les dates.
- Distinguer les fonctions actuelles et anciennes.
- Conserver les relations historiques.
- Une relation doit posséder une preuve exacte.
- Ne pas transformer une accusation en fait établi.
- Ne pas confondre affiliation et simple participation.
- Ne pas confondre président d'une organisation et membre.
- Ne pas considérer une mention comme une relation.

TYPES D'ENTITÉS :

PERSONNE
PARTI_POLITIQUE
INSTITUTION
ENTREPRISE
ORGANISATION
ADMINISTRATION
FONCTION
LIEU
EVENEMENT
LOI
DOCUMENT
PROJET
UNIVERSITE
MEDIA
AUTRE

TYPES DE RELATIONS :

MEMBRE_DE
ADHERENT_DE
FONDATEUR_DE
PRESIDENT_DE
PRESIDE
DIRIGE
DIRECTEUR_DE
OCCUPE_FONCTION
NOMME_A
ELU_A
CANDIDAT_A
APPARTIENT_A
AFFILIE_A
ACTIONNAIRE_DE
ADMINISTRATEUR_DE
EMPLOYE_PAR
FILIALE_DE
PARTICIPE_A
ORGANISE
FINANCE
SIGNE
COLLABORE_AVEC
OPPOSE_A
SUCCEDE_A
REMPLACE
FAMILLE_DE
PARENT_DE
MARIE_A
ETUDIE_A
DIPLOME_DE
SIEGE_A
LOCALISE_A
CREE_PAR
DEPEND_DE
AUTRE

Pour chaque entité :
- name
- type
- description
- aliases
- attributes

Pour chaque relation :
- source
- target
- type
- description
- start_date
- end_date
- status
- certainty
- evidence

Valeurs de status :
reported, asserted, alleged, denied, planned,
historical, uncertain.

Valeurs de certainty :
high, medium, low.

La preuve doit être une citation exacte du texte.

Réponds uniquement avec un JSON valide.
"""


def build_prompt(chunk, metadata):
    return f"""
DOCUMENT :
Titre : {metadata.get("title", "")}
URL : {metadata.get("source_url", "")}
Type : {metadata.get("file_type", "")}

TEXTE :

{chunk}

Extrais toutes les entités et les relations pertinentes.

Cherche notamment :
- responsables politiques ;
- dirigeants ;
- membres de partis ;
- ministres ;
- anciens ministres ;
- députés ;
- élus ;
- dirigeants d'entreprises ;
- administrateurs ;
- institutions ;
- organisations ;
- affiliations ;
- nominations ;
- mandats ;
- participations ;
- liens familiaux explicitement établis ;
- relations entre entreprises et institutions.

N'oublie pas les relations historiques.

Structure obligatoire :

{{
  "entities": [
    {{
      "name": "",
      "type": "",
      "description": "",
      "aliases": [],
      "attributes": {{}}
    }}
  ],
  "relations": [
    {{
      "source": "",
      "target": "",
      "type": "",
      "description": "",
      "start_date": null,
      "end_date": null,
      "status": "reported",
      "certainty": "medium",
      "evidence": ""
    }}
  ]
}}
"""


# ============================================================
# 13. APPEL GROQ
# ============================================================

def parse_json_response(content):
    content = (content or "").strip()

    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start >= 0 and end > start:
            return json.loads(
                content[start:end + 1]
            )

        raise ValueError(
            "La réponse du modèle n'est pas un JSON valide"
        )


class TruncatedResponse(Exception):
    """Sortie coupée par max_tokens : réessayer à l'identique est inutile."""


# Consommation réelle de tokens par modèle sur la dernière minute.
# Les limites TPM de Groq sont comptées par modèle.
_usage = {}


def estimate_tokens(*texts):
    # Estimation prudente : environ 3 caractères par token.
    return sum(len(t) for t in texts) // 3


def get_status_code(exc):
    code = getattr(exc, "status_code", None)

    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)

    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _usage_window(model):
    now = time.time()

    log = [
        (t, n)
        for t, n in _usage.get(model, [])
        if now - t < 60
    ]

    _usage[model] = log

    return log


def seconds_until_budget(model, needed):
    """Secondes à attendre avant que `needed` tokens tiennent dans la minute."""

    log = _usage_window(model)
    used = sum(n for _, n in log)

    if used + needed <= TPM_LIMIT:
        return 0.0

    to_free = used + needed - TPM_LIMIT
    freed = 0
    now = time.time()

    for t, n in log:
        freed += n

        if freed >= to_free:
            return max(0.0, 60 - (now - t)) + 1.0

    return 61.0


def record_usage(model, tokens):
    _usage.setdefault(model, []).append((time.time(), tokens))


def retry_after_seconds(exc, default):
    """Lit « try again in 24.915s » dans le message d'erreur Groq."""

    match = re.search(
        r"try again in (?:(\d+)m)?\s*([0-9.]+)s",
        str(exc)
    )

    if match:
        return (
            int(match.group(1) or 0) * 60
            + float(match.group(2))
            + 1
        )

    return default


def model_settings(model):
    extra = {}
    out_tokens = MAX_OUTPUT_TOKENS

    # Les modèles gpt-oss raisonnent avant de répondre :
    # effort bas et marge de sortie supplémentaire.
    if "gpt-oss" in model:
        extra["extra_body"] = {"reasoning_effort": "low"}
        out_tokens = MAX_OUTPUT_TOKENS + REASONING_EXTRA_TOKENS

    return extra, out_tokens


def call_groq(system_prompt, user_prompt):
    last_error = None

    input_tokens = estimate_tokens(system_prompt, user_prompt)

    candidates = []

    for model in MODELS:

        extra, out_tokens = model_settings(model)
        needed = input_tokens + out_tokens

        if needed > TPM_LIMIT:
            logger.warning(
                "Modèle %s ignoré : requête estimée à %s tokens "
                "(limite configurée %s).",
                model,
                needed,
                TPM_LIMIT
            )
            last_error = RuntimeError(
                f"Requête estimée à {needed} tokens "
                f"pour une limite de {TPM_LIMIT}"
            )
            continue

        candidates.append((model, extra, out_tokens, needed))

    # Le modèle qui a de la marge le plus tôt passe en premier.
    candidates.sort(
        key=lambda c: seconds_until_budget(c[0], c[3])
    )

    for model, extra, out_tokens, needed in candidates:

        for attempt in range(MAX_RETRIES):

            wait = seconds_until_budget(model, needed)

            if wait > 0:
                logger.info(
                    "Limite TPM : attente de %.0f s avant %s.",
                    wait,
                    model
                )
                time.sleep(wait)

            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    max_tokens=out_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    **extra
                )

                usage = getattr(response, "usage", None)

                record_usage(
                    model,
                    getattr(usage, "total_tokens", None) or needed
                )

                if not response.choices:
                    raise ValueError("Groq n'a retourné aucun choix.")

                choice = response.choices[0]

                if getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedResponse(
                        "Réponse tronquée (finish_reason=length)."
                    )

                result = parse_json_response(
                    choice.message.content
                )

                if not isinstance(result, dict):
                    raise ValueError("JSON retourné invalide")

                result["_model"] = model

                return result

            except Exception as e:
                last_error = e
                code = get_status_code(e)

                logger.warning(
                    "Erreur modèle %s, essai %s, HTTP %s : %s",
                    model,
                    attempt + 1,
                    code,
                    str(e)[:300]
                )

                # Même requête = même troncature : modèle suivant.
                if isinstance(e, TruncatedResponse):
                    break

                # 413 : requête trop grosse, réessayer est inutile.
                if code == 413:
                    break

                # 401/403 : clé invalide ou accès refusé.
                if code in (401, 403):
                    raise

                # 404 : modèle introuvable, passer au suivant.
                if code == 404:
                    break

                # 429 : quota par minute atteint. On attend le délai
                # indiqué par Groq, puis on réessaie.
                if code == 429:
                    time.sleep(
                        retry_after_seconds(e, 20 * (attempt + 1))
                    )
                    continue

                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Tous les modèles ont échoué : {last_error}"
    )


def extract_chunk(chunk, metadata):
    return call_groq(
        SYSTEM_PROMPT,
        build_prompt(chunk, metadata)
    )


# ============================================================
# 14. VALIDATION DES ENTITÉS
# ============================================================

VALID_ENTITY_TYPES = {
    "PERSONNE",
    "PARTI_POLITIQUE",
    "INSTITUTION",
    "ENTREPRISE",
    "ORGANISATION",
    "ADMINISTRATION",
    "FONCTION",
    "LIEU",
    "EVENEMENT",
    "LOI",
    "DOCUMENT",
    "PROJET",
    "UNIVERSITE",
    "MEDIA",
    "AUTRE"
}


def normalize_entity_type(value):
    value = normalize_text(value).upper()

    aliases = {
        "PERSON": "PERSONNE",
        "HUMAN": "PERSONNE",
        "PARTI": "PARTI_POLITIQUE",
        "POLITICAL_PARTY": "PARTI_POLITIQUE",
        "COMPANY": "ENTREPRISE",
        "ORGANIZATION": "ORGANISATION",
        "FUNCTION": "FONCTION",
        "LOCATION": "LIEU",
        "EVENT": "EVENEMENT"
    }

    value = aliases.get(value, value)

    return (
        value
        if value in VALID_ENTITY_TYPES
        else "AUTRE"
    )


def validate_entity(raw):
    if not isinstance(raw, dict):
        return None

    name = normalize_text(raw.get("name"))

    if len(name) < 2:
        return None

    entity_type = normalize_entity_type(
        raw.get("type")
    )

    aliases = raw.get("aliases", [])

    if not isinstance(aliases, list):
        aliases = []

    aliases = [
        normalize_text(a)
        for a in aliases
        if isinstance(a, str)
        and len(normalize_text(a)) > 1
    ]

    attributes = raw.get("attributes", {})

    if not isinstance(attributes, dict):
        attributes = {}

    return {
        "id": make_id(entity_type, name),
        "name": name,
        "normalized_name": normalize_name(name),
        "type": entity_type,
        "description": normalize_text(
            raw.get("description", "")
        ),
        "aliases": aliases,
        "attributes": attributes,
        "sources": []
    }


# ============================================================
# 15. FUSION DES ENTITÉS
# ============================================================

def find_entity(entities, name):
    normalized = normalize_name(name)

    for entity in entities.values():

        if entity["normalized_name"] == normalized:
            return entity

        for alias in entity.get("aliases", []):

            if normalize_name(alias) == normalized:
                return entity

    return None


def add_entity(entities, raw, source):
    entity = validate_entity(raw)

    if not entity:
        return None

    existing = find_entity(
        entities,
        entity["name"]
    )

    if existing is None:

        entity["sources"].append(source)

        entities[entity["id"]] = entity

        return entity

    # Ne pas fusionner automatiquement des types différents.
    if existing["type"] != entity["type"]:
        entity["sources"].append(source)
        entities[entity["id"]] = entity
        return entity

    aliases = set(existing.get("aliases", []))

    aliases.update(entity.get("aliases", []))

    if existing["name"] != entity["name"]:
        aliases.add(entity["name"])

    existing["aliases"] = sorted(aliases)

    if (
        not existing.get("description")
        and entity.get("description")
    ):
        existing["description"] = entity["description"]

    for key, value in entity["attributes"].items():

        if value not in (None, "", [], {}):
            existing["attributes"].setdefault(key, value)

    source_key = (
        source.get("document_sha256"),
        source.get("chunk_id")
    )

    known_sources = {
        (
            s.get("document_sha256"),
            s.get("chunk_id")
        )
        for s in existing["sources"]
    }

    if source_key not in known_sources:
        existing["sources"].append(source)

    return existing


# ============================================================
# 16. VALIDATION DES RELATIONS
# ============================================================

ALLOWED_STATUS = {
    "reported",
    "asserted",
    "alleged",
    "denied",
    "planned",
    "historical",
    "uncertain"
}


def validate_relation(raw, chunk, metadata):
    if not isinstance(raw, dict):
        return None

    source = normalize_text(raw.get("source"))
    target = normalize_text(raw.get("target"))
    rel_type = normalize_text(raw.get("type")).upper()
    evidence = normalize_text(raw.get("evidence"))

    if not source or not target or not rel_type:
        return None

    if normalize_name(source) == normalize_name(target):
        return None

    # Vérifier la preuve dans le texte original normalisé.
    if not evidence or evidence not in chunk:
        return None

    status = normalize_text(
        raw.get("status", "reported")
    ).lower()

    if status not in ALLOWED_STATUS:
        status = "reported"

    certainty = normalize_text(
        raw.get("certainty", "medium")
    ).lower()

    if certainty not in {"high", "medium", "low"}:
        certainty = "medium"

    return {
        "source_name": source,
        "target_name": target,
        "type": rel_type,
        "description": normalize_text(
            raw.get("description", "")
        ),
        "start_date": raw.get("start_date"),
        "end_date": raw.get("end_date"),
        "status": status,
        "certainty": certainty,
        "evidence": evidence,
        "source_url": metadata.get("source_url"),
        "document_sha256": metadata.get("sha256"),
        "hdfs_path": metadata.get("hdfs_path"),
        "chunk_id": metadata.get("_chunk_id"),
        "page": metadata.get("_page"),
        "extracted_at": utc_now()
    }


# ============================================================
# 17. FUSION DES RELATIONS
# ============================================================

def relation_identity(rel):
    return "|".join([
        normalize_name(rel["source_name"]),
        rel["type"],
        normalize_name(rel["target_name"]),
        normalize_text(rel.get("start_date") or "")
    ])


def add_relation(relations, rel, entities):
    source_entity = find_entity(
        entities,
        rel["source_name"]
    )

    target_entity = find_entity(
        entities,
        rel["target_name"]
    )

    source_id = (
        source_entity["id"] if source_entity else None
    )

    target_id = (
        target_entity["id"] if target_entity else None
    )

    key = relation_identity(rel)

    relation_id = make_id("REL", key)

    if relation_id not in relations:

        relations[relation_id] = {
            "id": relation_id,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "source_name": rel["source_name"],
            "target_name": rel["target_name"],
            "type": rel["type"],
            "description": rel["description"],
            "start_date": rel["start_date"],
            "end_date": rel["end_date"],
            "status": rel["status"],
            "certainty": rel["certainty"],
            "resolution_status": (
                "resolved"
                if source_id and target_id
                else "unresolved"
            ),
            "evidences": []
        }

    current = relations[relation_id]

    # Actualiser les références si une entité vient d'être trouvée.
    if source_id:
        current["source_entity_id"] = source_id

    if target_id:
        current["target_entity_id"] = target_id

    current["resolution_status"] = (
        "resolved"
        if current["source_entity_id"]
        and current["target_entity_id"]
        else "unresolved"
    )

    evidence = {
        "text": rel["evidence"],
        "source_url": rel["source_url"],
        "document_sha256": rel["document_sha256"],
        "hdfs_path": rel["hdfs_path"],
        "chunk_id": rel["chunk_id"],
        "page": rel["page"],
        "extracted_at": rel["extracted_at"]
    }

    evidence_key = (
        evidence["document_sha256"],
        evidence["chunk_id"],
        evidence["text"]
    )

    known = {
        (
            x.get("document_sha256"),
            x.get("chunk_id"),
            x.get("text")
        )
        for x in current["evidences"]
    }

    if evidence_key not in known:
        current["evidences"].append(evidence)


# ============================================================
# 18. TRAITEMENT D'UN DOCUMENT
# ============================================================

def save_checkpoint(entities, relations):
    save_json_atomic(
        ENTITIES_FILE,
        list(entities.values())
    )

    save_json_atomic(
        RELATIONS_FILE,
        list(relations.values())
    )


def process_document(
    entry,
    entities,
    relations,
    documents,
    progress
):
    document_sha = entry.get("sha256")
    hdfs_path = entry.get("hdfs_path")

    if not document_sha or not hdfs_path:
        raise ValueError("SHA256 ou hdfs_path absent")

    if progress.get(document_sha, {}).get("status") == "completed":
        logger.info("Document déjà traité : %s", hdfs_path)
        return

    raw_data = read_hdfs_file(hdfs_path)

    actual_sha = sha256_bytes(raw_data)

    if actual_sha != document_sha:
        raise ValueError(
            "SHA256 différent entre le manifeste et HDFS. "
            "Vérifier la méthode de calcul du SHA dans le crawler."
        )

    title, pages = extract_document(
        entry,
        raw_data
    )

    metadata = {
        "title": title,
        "source_url": entry.get("source_url"),
        "normalized_url": entry.get("normalized_url"),
        "hdfs_path": hdfs_path,
        "sha256": document_sha,
        "file_type": entry.get("file_type"),
        "downloaded_at": entry.get("downloaded_at"),
        "corpus_version": entry.get("corpus_version")
    }

    chunks = []

    for page in pages:

        for index, text in enumerate(
            chunk_text(page.get("text", ""))
        ):

            chunks.append({
                "text": text,
                "page": page.get("page"),
                "index": index
            })

    if not chunks:
        raise ValueError("Aucun texte exploitable")

    extracted_entities_count = 0
    extracted_relations_count = 0

    for index, item in enumerate(chunks):

        chunk = item["text"]

        chunk_id = (
            f"{document_sha[:12]}_{index:05d}"
        )

        chunk_metadata = {
            **metadata,
            "_chunk_id": chunk_id,
            "_page": item["page"]
        }

        logger.info(
            "Chunk %s/%s : %s",
            index + 1,
            len(chunks),
            chunk_id
        )

        result = extract_chunk(
            chunk,
            chunk_metadata
        )

        source_reference = {
            "source_url": metadata["source_url"],
            "document_sha256": document_sha,
            "hdfs_path": hdfs_path,
            "chunk_id": chunk_id,
            "page": item["page"],
            "evidence_context": chunk[:500]
        }

        extracted_entities = result.get("entities", [])

        if not isinstance(extracted_entities, list):
            extracted_entities = []

        for raw_entity in extracted_entities:

            if add_entity(
                entities,
                raw_entity,
                source_reference
            ):
                extracted_entities_count += 1

        extracted_relations = result.get("relations", [])

        if not isinstance(extracted_relations, list):
            extracted_relations = []

        for raw_relation in extracted_relations:

            rel = validate_relation(
                raw_relation,
                chunk,
                chunk_metadata
            )

            if not rel:
                continue

            add_relation(
                relations,
                rel,
                entities
            )

            extracted_relations_count += 1

        # Sauvegarder les checkpoints tous les 5 chunks.
        if (index + 1) % 5 == 0:
            save_checkpoint(entities, relations)

        # Pause pour rester sous la limite de tokens par minute.
        if index + 1 < len(chunks) and CHUNK_PAUSE > 0:
            time.sleep(CHUNK_PAUSE)

    documents[document_sha] = {
        **metadata,
        "chunks_count": len(chunks),
        "entities_extracted": extracted_entities_count,
        "relations_extracted": extracted_relations_count,
        "processed_at": utc_now(),
        "status": "completed"
    }

    progress[document_sha] = {
        "status": "completed",
        "hdfs_path": hdfs_path,
        "processed_at": utc_now()
    }

    logger.info(
        "Terminé : %s | entités=%s | relations=%s",
        hdfs_path,
        extracted_entities_count,
        extracted_relations_count
    )


# ============================================================
# 19. PROGRAMME PRINCIPAL
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("EXTRACTION GRAPHRAG MAROC")
    logger.info("=" * 65)

    logger.info(
        "Modèles : %s | chunk=%s | sortie=%s | TPM estimé max=%s | pause=%ss",
        MODELS,
        CHUNK_SIZE,
        MAX_OUTPUT_TOKENS,
        TPM_LIMIT,
        CHUNK_PAUSE
    )

    for path in [
        ENTITIES_FILE,
        RELATIONS_FILE,
        PROGRESS_FILE,
        FAILURES_FILE,
        DOCUMENTS_FILE
    ]:
        backup_file(path)

    entities_list = load_json(
        ENTITIES_FILE,
        []
    )

    relations_list = load_json(
        RELATIONS_FILE,
        []
    )

    progress = load_json(
        PROGRESS_FILE,
        {}
    )

    failures = load_json(
        FAILURES_FILE,
        []
    )

    documents = load_json(
        DOCUMENTS_FILE,
        {}
    )

    entities = {
        e["id"]: e
        for e in entities_list
        if isinstance(e, dict) and e.get("id")
    }

    relations = {
        r["id"]: r
        for r in relations_list
        if isinstance(r, dict) and r.get("id")
    }

    old_entries = read_manifest(OLD_MANIFEST)

    current_entries = read_manifest(CURRENT_MANIFEST)

    logger.info(
        "Anciennes entrées : %s",
        len(old_entries)
    )

    logger.info(
        "Entrées actuelles : %s",
        len(current_entries)
    )

    new_documents = get_new_documents(
        old_entries,
        current_entries
    )

    pending = [
        entry
        for entry in new_documents
        if progress.get(entry.get("sha256"), {}).get("status")
        != "completed"
    ]

    if MAX_DOCUMENTS_TEST > 0:
        pending = pending[:MAX_DOCUMENTS_TEST]

    logger.info(
        "Nouveaux ou modifiés : %s",
        len(new_documents)
    )

    logger.info(
        "Documents à traiter maintenant : %s",
        len(pending)
    )

    for index, entry in enumerate(pending, start=1):

        logger.info(
            "DOCUMENT %s/%s",
            index,
            len(pending)
        )

        try:

            process_document(
                entry,
                entities,
                relations,
                documents,
                progress
            )

        except KeyboardInterrupt:
            logger.warning(
                "Interruption utilisateur. Sauvegarde en cours."
            )
            raise

        except Exception as e:

            logger.exception(
                "Échec document : %s",
                entry.get("hdfs_path")
            )

            failures.append({
                "hdfs_path": entry.get("hdfs_path"),
                "source_url": entry.get("source_url"),
                "sha256": entry.get("sha256"),
                "error": str(e),
                "failed_at": utc_now()
            })

        finally:

            # Sauvegarder après chaque document, même en cas d'échec.
            save_checkpoint(entities, relations)

            save_json_atomic(
                PROGRESS_FILE,
                progress
            )

            save_json_atomic(
                FAILURES_FILE,
                failures
            )

            save_json_atomic(
                DOCUMENTS_FILE,
                documents
            )

    logger.info("=" * 65)
    logger.info("EXTRACTION TERMINÉE")
    logger.info("Entités : %s", len(entities))
    logger.info("Relations : %s", len(relations))
    logger.info("Documents traités : %s", len(documents))
    logger.info("Échecs cumulés : %s", len(failures))
    logger.info("=" * 65)


if __name__ == "__main__":
    main()