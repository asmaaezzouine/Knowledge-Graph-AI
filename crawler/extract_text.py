"""
Extraction du texte depuis HDFS pour le projet « liens entre personnalités
publiques ».

- Pages HTML (Wikipédia, institutions, presse) -> page_XXXXX.json
  (texte propre + url, qid, langue issus du manifeste du crawler)
- Fiches Wikidata (JSON) -> entity_Qxxx.json
  (libellés, relations orientées pour le graphe, comptes officiels)

Notes de minimisation des données :
- les enfants (P40) ne sont volontairement PAS extraits (mineurs possibles) ;
- seule l'année de naissance est conservée.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

HDFS_CONTAINER = os.getenv("HDFS_CONTAINER", "namenode")
HDFS_BASE = os.getenv("HDFS_BASE", "/DataLake/raw_personnes")
HDFS_OUTPUT = f"{HDFS_BASE}/processed"
HDFS_MANIFEST = f"{HDFS_BASE}/manifest.json"

MIN_TEXT_LENGTH = 100

# Propriété Wikidata -> (relation du graphe, inverse ?)
# inverse=True : la valeur Wikidata est le SUJET de la relation du graphe
# (ex. « société P127 détenue par X » devient X -ACTIONNAIRE_DE-> société).
RELATIONS = {
    "P102": ("MEMBRE_DE", False),          # parti politique
    "P463": ("MEMBRE_DE", False),          # membre de
    "P39": ("OCCUPE_FONCTION", False),     # fonction
    "P108": ("EMPLOYE_PAR", False),        # employeur
    "P127": ("ACTIONNAIRE_DE", True),      # détenu par
    "P1830": ("ACTIONNAIRE_DE", False),    # propriétaire de
    "P112": ("FONDATEUR_DE", True),        # fondé par
    "P169": ("DIRIGE", True),              # directeur général
    "P1037": ("DIRIGE", True),             # directeur
    "P488": ("PRESIDE", True),             # président
    "P3320": ("ADMINISTRATEUR_DE", True),  # membre du conseil
    "P749": ("FILIALE_DE", False),         # organisation mère
    "P355": ("FILIALE_DE", True),          # filiale
    "P26": ("CONJOINT_DE", False),         # conjoint
    "P22": ("PARENT_DE", True),            # père
    "P25": ("PARENT_DE", True),            # mère
    "P3373": ("FRATRIE_DE", False),        # frère / sœur
}

COMPTES = {
    "P2002": "x_twitter",
    "P2013": "facebook",
    "P2003": "instagram",
    "P2397": "youtube",
    "P856": "site_officiel",
}

SECTIONS_FINALES = {
    "notes_et_références", "références", "voir_aussi", "liens_externes",
    "bibliographie", "notes", "annexes", "articles_connexes",
}


# ============================================================
# COMMANDES HDFS
# ============================================================

def hdfs_command(args, input_data=None):
    command = [
        "docker", "exec", "-i", HDFS_CONTAINER,
        "hdfs", "dfs", "-Ddfs.replication=1", *args,
    ]

    kwargs = {"input": input_data} if input_data is not None else {
        "stdin": subprocess.DEVNULL
    }

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        **kwargs,
    )

    if result.returncode != 0:
        error = result.stderr.decode(errors="replace")
        raise RuntimeError(f"Erreur HDFS : {' '.join(args)}\n{error}")

    return result.stdout


def list_hdfs(directory, extensions):
    try:
        result = hdfs_command(["-ls", directory])
    except RuntimeError:
        return []

    files = []

    for line in result.decode(errors="replace").splitlines():
        parts = line.split()

        if len(parts) >= 8 and parts[-1].lower().endswith(extensions):
            files.append(parts[-1])

    return files


def upload_bytes(content, hdfs_path):
    hdfs_command(["-mkdir", "-p", hdfs_path.rsplit("/", 1)[0]])
    hdfs_command(["-put", "-f", "-", hdfs_path], input_data=content)


def upload_json(data, hdfs_path):
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(content + b"\n", hdfs_path)


def load_manifest():
    """Retourne {hdfs_path: entrée du manifeste}."""
    try:
        data = json.loads(hdfs_command(["-cat", HDFS_MANIFEST]).decode("utf-8"))
    except Exception as exc:
        print(f"[AVERTISSEMENT] Manifeste illisible : {exc}")
        return {}

    if isinstance(data, dict):
        data = data.get("files", [])

    index = {}
    for entry in data:
        if isinstance(entry, dict) and entry.get("hdfs_path"):
            index.setdefault(entry["hdfs_path"], entry)

    return index


# ============================================================
# EXTRACTION HTML
# ============================================================

def is_wikipedia(url):
    return ".wikipedia.org" in (url or "")


def infobox_lines(content):
    """Transforme l'infobox Wikipédia en lignes « libellé : valeur »."""
    box = content.select_one(
        "table.infobox, table.infobox_v2, table.infobox_v3"
    )

    if not box:
        return ""

    lines = []

    for row in box.find_all("tr"):
        header = row.find("th")
        cell = row.find("td")

        if header and cell:
            label = header.get_text(" ", strip=True)
            value = cell.get_text(" ", strip=True)

            if label and value:
                lines.append(f"{label} : {value}")

    box.decompose()

    return "\n".join(lines)


def cut_final_sections(content):
    """Supprime « Notes et références », « Liens externes », etc."""
    for heading in content.find_all("h2"):
        heading_id = (heading.get("id") or "").lower()

        if heading_id in SECTIONS_FINALES:
            parent = heading.parent
            anchor = (
                parent
                if parent is not None
                and "mw-heading" in (parent.get("class") or [])
                else heading
            )

            for sibling in list(anchor.find_next_siblings()):
                sibling.decompose()

            anchor.decompose()
            break


def extract_text(html_bytes, source_url=""):
    soup = BeautifulSoup(html_bytes, "html.parser")

    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    title = ""
    infobox = ""

    if is_wikipedia(source_url):
        heading = soup.find(id="firstHeading")
        title = (
            heading.get_text(" ", strip=True) if heading
            else (soup.title.get_text(" ", strip=True) if soup.title else "")
        )

        content = soup.find(id="mw-content-text") or soup.body or soup

        for element in content.select(
            ".reference, .reflist, .mw-editsection, .navbox, "
            ".printfooter, .noprint, .hatnote, table.ambox"
        ):
            element.decompose()

        cut_final_sections(content)
        infobox = infobox_lines(content)
        body = content.get_text(" ", strip=True)

    else:
        for element in soup.find_all(["nav", "footer", "header", "aside"]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        content = soup.find("main") or soup.find("article") or soup.body or soup
        body = content.get_text(" ", strip=True)

    body = re.sub(r"\s+", " ", body.replace("\xa0", " ")).strip()
    text = f"{infobox}\n\n{body}".strip() if infobox else body

    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError(f"Texte extrait trop court ({len(text)} caractères).")

    return title, text


# ============================================================
# EXTRACTION WIKIDATA
# ============================================================

def format_time(raw):
    if not raw or raw.startswith("-"):
        return None

    date = raw.lstrip("+").split("T")[0]
    parts = date.split("-")

    while parts and parts[-1] == "00":
        parts.pop()

    return "-".join(parts) or None


def entity_id(statement):
    snak = statement.get("mainsnak", {})

    if snak.get("snaktype") != "value":
        return None

    datavalue = snak.get("datavalue", {})

    if datavalue.get("type") == "wikibase-entityid":
        return datavalue.get("value", {}).get("id")

    return None


def string_value(statement):
    snak = statement.get("mainsnak", {})

    if snak.get("snaktype") != "value":
        return None

    value = snak.get("datavalue", {}).get("value")

    return value if isinstance(value, str) else None


def qualifier_date(statement, prop):
    for snak in statement.get("qualifiers", {}).get(prop, []):
        datavalue = snak.get("datavalue") or {}

        if datavalue.get("type") == "time":
            return format_time(datavalue.get("value", {}).get("time", ""))

    return None


def extract_entity(raw_bytes, source_url, hdfs_path):
    data = json.loads(raw_bytes.decode("utf-8"))
    entities = data.get("entities", {})

    if not entities:
        raise ValueError("Fiche Wikidata vide.")

    qid, entity = next(iter(entities.items()))
    claims = entity.get("claims", {})
    langs = ("fr", "ar", "en")

    labels = {
        lang: entity["labels"][lang]["value"]
        for lang in langs if lang in entity.get("labels", {})
    }
    descriptions = {
        lang: entity["descriptions"][lang]["value"]
        for lang in langs if lang in entity.get("descriptions", {})
    }

    relations = []

    for prop, (relation, inverse) in RELATIONS.items():
        for statement in claims.get(prop, []):
            if statement.get("rank") == "deprecated":
                continue

            other = entity_id(statement)

            if not other:
                continue

            subject, obj = (other, qid) if inverse else (qid, other)

            relations.append({
                "subject": subject,
                "relation": relation,
                "object": obj,
                "wikidata_property": prop,
                "debut": qualifier_date(statement, "P580"),
                "fin": qualifier_date(statement, "P582"),
            })

    comptes = {}

    for prop, name in COMPTES.items():
        values = [
            v for v in (string_value(s) for s in claims.get(prop, [])) if v
        ]

        if values:
            comptes[name] = values[0]

    birth_year = None

    for statement in claims.get("P569", []):
        datavalue = statement.get("mainsnak", {}).get("datavalue", {})
        date = format_time(datavalue.get("value", {}).get("time", ""))

        if date:
            birth_year = int(date.split("-")[0])
            break

    def ids(prop):
        return [
            i for i in (entity_id(s) for s in claims.get(prop, [])) if i
        ]

    return {
        "document_id": f"entity_{qid}",
        "qid": qid,
        "labels": labels,
        "descriptions": descriptions,
        "types": ids("P31"),
        "professions": ids("P106"),
        "nationalites": ids("P27"),
        "annee_naissance": birth_year,
        "comptes_officiels": comptes,
        "relations": relations,
        "source": "wikidata",
        "source_url": source_url,
        "source_hdfs_path": hdfs_path,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# TRAITEMENT
# ============================================================

def process_html(hdfs_path, manifest):
    entry = manifest.get(hdfs_path, {})
    source_url = entry.get("source_url", "")
    stem = Path(hdfs_path).stem

    title, text = extract_text(hdfs_command(["-cat", hdfs_path]), source_url)

    document = {
        "document_id": stem,
        "source_hdfs_path": hdfs_path,
        "url": source_url,
        "qid": entry.get("qid", ""),
        "language": entry.get("language", ""),
        "title": title,
        "text": text,
        "text_length": len(text),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output = f"{HDFS_OUTPUT}/{stem}.json"
    upload_json(document, output)

    return {"document_id": stem, "output_hdfs_path": output,
            "title": title, "text_length": len(text)}


def process_entity(hdfs_path, manifest):
    entry = manifest.get(hdfs_path, {})
    qid = Path(hdfs_path).stem

    document = extract_entity(
        hdfs_command(["-cat", hdfs_path]),
        entry.get("source_url", ""),
        hdfs_path,
    )

    output = f"{HDFS_OUTPUT}/entity_{qid}.json"
    upload_json(document, output)

    return {"document_id": f"entity_{qid}", "output_hdfs_path": output,
            "relations": len(document["relations"])}


def main():
    print("[INFO] Lecture du manifeste et des fichiers HDFS...")

    manifest = load_manifest()
    html_files = list_hdfs(f"{HDFS_BASE}/html", (".html", ".htm"))
    json_files = list_hdfs(f"{HDFS_BASE}/json", (".json",))

    print(f"[INFO] Pages HTML : {len(html_files)}")
    print(f"[INFO] Fiches Wikidata : {len(json_files)}")

    if not html_files and not json_files:
        print("[AVERTISSEMENT] Aucun fichier à traiter.")
        return

    processed, errors = [], []

    jobs = (
        [(p, process_html) for p in html_files]
        + [(p, process_entity) for p in json_files]
    )

    for index, (hdfs_path, handler) in enumerate(jobs, start=1):
        print(f"[TRAITEMENT] ({index}/{len(jobs)}) {hdfs_path}")

        try:
            result = handler(hdfs_path, manifest)
            processed.append(result)
            print(f"[OK] {result['output_hdfs_path']}")

        except Exception as exc:
            errors.append({"hdfs_path": hdfs_path, "error": str(exc)})
            print(f"[ERREUR] {hdfs_path} : {exc}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_html": len(html_files),
        "input_wikidata": len(json_files),
        "processed_count": len(processed),
        "error_count": len(errors),
        "documents": processed,
        "errors": errors,
    }

    try:
        upload_json(report, f"{HDFS_OUTPUT}/extraction_report.json")
        print(f"[HDFS] Rapport : {HDFS_OUTPUT}/extraction_report.json")
    except Exception as exc:
        print(f"[AVERTISSEMENT] Rapport non envoyé : {exc}")

    print("\n========== BILAN EXTRACTION ==========")
    print(f"Fichiers traités : {len(processed)}/{len(jobs)}")
    print(f"Erreurs : {len(errors)}")
    print("======================================")


if __name__ == "__main__":
    main()