
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crawler multi-sources GraphRAG Maroc.

- Collecte incrémentale.
- Conservation des anciennes données.
- Déduplication URL et SHA-256.
- Réutilisation des anciens HTML.
- Collecte HTML, PDF et JSON Wikidata.
- Sauvegarde automatique des anciens manifestes.
- Publication atomique des fichiers JSON dans HDFS.
- Aucune suppression de documents du corpus.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid

from collections import deque
from datetime import datetime, timezone
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    unquote,
    quote,
)
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

HDFS_BASE = os.getenv(
    "CRAWLER_HDFS_BASE",
    "/DataLake/raw_personnes"
)

HDFS_HTML = f"{HDFS_BASE}/html"
HDFS_PDF = f"{HDFS_BASE}/pdf"
HDFS_JSON = f"{HDFS_BASE}/json"
HDFS_PROCESSED = f"{HDFS_BASE}/processed"
HDFS_MANIFEST = f"{HDFS_BASE}/manifest.json"
HDFS_BACKUPS = f"{HDFS_BASE}/manifest_backups"

DOCKER_CONTAINER = os.getenv(
    "HDFS_CONTAINER",
    "namenode"
)

CONTACT = os.getenv(
    "CRAWLER_CONTACT",
    "contact@exemple.ma"
)

USER_AGENT = (
    f"PersonalitiesGraphResearchBot/3.0 "
    f"(projet academique; {CONTACT})"
)

CORPUS_VERSION = "morocco_v3"

MAX_RETRIES = 4
REQUEST_TIMEOUT = (10, 60)


# ============================================================
# DOMAINES AUTORISÉS
# ============================================================

ALLOWED_DOMAINS = (
    "fr.wikipedia.org",
    "ar.wikipedia.org",
    "en.wikipedia.org",
    "www.wikidata.org",

    "gouvernement.ma",
    "chambredesrepresentants.ma",
    "sgg.gov.ma",

    "hespress.com",
    "telquel.ma",
    "lebrief.ma",
    "ecoactu.ma",
    "leseco.ma",
    "medias24.com",

    "politicsinmorocco.com",
    "welovebuzz.com",
    "le360.ma",
)


# ============================================================
# SOURCES DE DÉPART
# ============================================================

SEED_URLS = [
    "https://fr.wikipedia.org/wiki/Gouvernement_Akhannouch",

    "https://fr.wikipedia.org/wiki/Chambre_des_repr%C3%A9sentants_(Maroc)",

    "https://fr.wikipedia.org/wiki/Chambre_des_conseillers",

    "https://fr.wikipedia.org/wiki/Liste_des_partis_politiques_au_Maroc",

    "https://fr.wikipedia.org/wiki/%C3%89lections_l%C3%A9gislatives_marocaines_de_2021",

    "https://fr.wikipedia.org/wiki/Bourse_de_Casablanca",

    "https://www.gouvernement.ma/fr",

    "https://politicsinmorocco.com/personnalites-politiques/",

    "https://www.welovebuzz.com/quel-est-le-niveau-detudes-de-nos-ministres-marocains/",

    "https://fr.le360.ma/economie/legislatives-ces-patrons-dentreprises-qui-se-presentent-aux-elections_5GKXX73SUNEXZN2HOP6VOQXZNY/",
]


# ============================================================
# MOTS-CLÉS
# ============================================================

KEYWORDS = [
    # Politique
    "ministre",
    "ministère",
    "gouvernement",
    "parlement",
    "député",
    "deputes",
    "conseiller",
    "chambre des représentants",
    "élu",
    "elus",
    "parti",
    "politique",
    "élections",
    "elections",
    "maire",
    "wali",
    "nomination",
    "conseil des ministres",
    "personnalité politique",
    "personnalites politiques",
    "président",
    "president",
    "secrétaire général",
    "secretaire general",

    # Économie
    "entreprise",
    "société",
    "societe",
    "holding",
    "groupe",
    "actionnaire",
    "administrateur",
    "conseil d'administration",
    "pdg",
    "directeur général",
    "homme d'affaires",
    "femme d'affaires",
    "fondateur",
    "investisseur",
    "patrimoine",
    "marché public",
    "marches publics",
    "dirigeant",

    # Réseaux et formation
    "famille",
    "épouse",
    "epouse",
    "fils de",
    "frère",
    "frere",
    "candidature",
    "candidat",
    "éducation",
    "diplôme",
    "diplome",
    "université",
    "universite",
    "formation",
    "parcours professionnel",

    # Arabe
    "وزير",
    "الحكومة",
    "البرلمان",
    "حزب",
    "شركة",
    "رجل أعمال",
    "رئيس الحكومة",
    "مجلس النواب",
    "مجلس المستشارين",
    "انتخابات",
]


PDF_CONTENT_TYPES = (
    "application/pdf",
    "application/x-pdf",
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("crawler")


# ============================================================
# UTILITAIRES
# ============================================================

def now_utc():
    return datetime.now(timezone.utc).isoformat()


def timestamp():
    return datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def normalize_url(url):
    """
    Normalise les URL pour réduire les doublons liés
    aux caractères Unicode et aux encodages.
    """

    if not url:
        return ""

    try:
        parsed = urlparse(url.strip())

        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()

        if not scheme or not hostname:
            return url.strip()

        netloc = hostname

        try:
            port = parsed.port
        except ValueError:
            port = None

        if port and not (
            (scheme == "https" and port == 443)
            or (scheme == "http" and port == 80)
        ):
            netloc = f"{hostname}:{port}"

        decoded_path = unquote(parsed.path or "/")

        path = quote(
            decoded_path,
            safe="/:@!$&'()*+,;=-._~"
        )

        if path != "/":
            path = path.rstrip("/")

        return urlunparse((
            scheme,
            netloc,
            path,
            "",
            parsed.query,
            ""
        ))

    except (ValueError, TypeError):
        return url.strip()


def is_allowed_domain(url):

    try:
        hostname = (
            urlparse(url).hostname or ""
        ).lower()

        return any(
            hostname == domain
            or hostname.endswith("." + domain)
            for domain in ALLOWED_DOMAINS
        )

    except (ValueError, TypeError):
        return False


def is_wikipedia(url):

    hostname = (
        urlparse(url).hostname or ""
    ).lower()

    return hostname.endswith(".wikipedia.org")


def wiki_article_ok(url):

    parsed = urlparse(url)

    if parsed.query or not parsed.path.startswith("/wiki/"):
        return False

    title = unquote(
        parsed.path[len("/wiki/"):]
    )

    if not title or ":" in title:
        return False

    lowered = title.lower()

    forbidden = (
        "accueil",
        "homonymie",
        "main_page",
    )

    return not any(
        item in lowered
        for item in forbidden
    )


def is_relevant(url, anchor_text=""):

    text = unquote(
        f"{url} {anchor_text}"
    ).lower()

    return any(
        keyword.lower() in text
        for keyword in KEYWORDS
    )


def is_pdf_url(url):

    path = unquote(
        urlparse(url).path
    ).lower()

    return path.endswith(".pdf")


def looks_like_pdf(url, content_type, content):

    mime = (content_type or "").lower()

    return (
        is_pdf_url(url)
        or any(
            item in mime
            for item in PDF_CONTENT_TYPES
        )
        or content.lstrip().startswith(b"%PDF")
    )


def safe_json_loads(content):

    return json.loads(
        content.decode("utf-8-sig")
    )


# ============================================================
# WIKIDATA
# ============================================================

QID_JSON_RE = re.compile(
    r'"wgWikibaseItemId"\s*:\s*"(Q\d+)"'
)

QID_LINK_RE = re.compile(
    r"Special:EntityPage/(Q\d+)"
)

QID_URL_RE = re.compile(
    r"(?:[?&]ids=|Special:EntityData/)(Q\d+)"
)


def extract_qid(html_bytes):

    text = html_bytes.decode(
        "utf-8",
        errors="replace"
    )

    match = (
        QID_JSON_RE.search(text)
        or QID_LINK_RE.search(text)
    )

    return match.group(1) if match else None


def wikidata_entity_url(qid):

    return (
        "https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={qid}&format=json"
    )


def is_wikidata_api(url):

    parsed = urlparse(url)

    return (
        (parsed.hostname or "").lower()
        == "www.wikidata.org"
        and parsed.path == "/w/api.php"
    )


# ============================================================
# CLIENT HDFS
# ============================================================

class HdfsClient:

    def __init__(self, container=DOCKER_CONTAINER):
        self.container = container

    def docker_exec(self, args, input_data=None):

        command = ["docker", "exec"]

        if input_data is not None:
            command.append("-i")

        command.extend(
            [self.container] + list(args)
        )

        result = subprocess.run(
            command,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        if result.returncode != 0:

            error = result.stderr.decode(
                "utf-8",
                errors="replace"
            ).strip()

            raise RuntimeError(
                f"Commande Docker échouée : {error}"
            )

        return result.stdout

    def run(self, args):

        output = self.docker_exec(
            ["hdfs", "dfs"] + list(args)
        )

        return output.decode(
            "utf-8",
            errors="replace"
        )

    def exists(self, path):

        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "hdfs",
                "dfs",
                "-test",
                "-e",
                path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        return result.returncode == 0

    def ensure_directories(self):

        for path in (
            HDFS_BASE,
            HDFS_HTML,
            HDFS_PDF,
            HDFS_JSON,
            HDFS_PROCESSED,
            HDFS_BACKUPS,
        ):

            self.run([
                "-mkdir",
                "-p",
                path
            ])

    def read_bytes(self, path):

        return self.docker_exec([
            "hdfs",
            "dfs",
            "-cat",
            path
        ])

    def read_json(self, path):

        if not self.exists(path):
            return None

        content = self.read_bytes(path)

        return safe_json_loads(content)

    def put_bytes(self, content, hdfs_path):
        """
        Dépose un fichier dans HDFS sans écraser une destination
        existante.

        Le fichier local temporaire est créé dans le conteneur
        Docker, puis envoyé dans HDFS.
        """

        temp_path = (
            f"/tmp/crawler_{uuid.uuid4().hex}"
        )

        self.docker_exec(
            ["sh", "-c", f"cat > '{temp_path}'"],
            input_data=content
        )

        try:

            self.run([
                "-put",
                temp_path,
                hdfs_path
            ])

        finally:

            subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container,
                    "rm",
                    "-f",
                    temp_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )

    def atomic_upload(self, content, hdfs_path):
        """
        Publication sécurisée :
        1. Écrit dans un fichier temporaire HDFS.
        2. Renomme l'ancien fichier vers une sauvegarde.
        3. Renomme le temporaire vers la destination.
        4. Restaure l'ancien fichier si la publication échoue.

        Aucun ancien fichier n'est supprimé.
        """

        directory = hdfs_path.rsplit("/", 1)[0]

        self.run([
            "-mkdir",
            "-p",
            directory
        ])

        temp_path = (
            f"{directory}/.__tmp_{uuid.uuid4().hex}.json"
        )

        backup_path = None

        # 1. Préparer le nouveau fichier
        self.put_bytes(
            content,
            temp_path
        )

        try:

            # Vérifier que le temporaire est présent
            if not self.exists(temp_path):
                raise RuntimeError(
                    "Le fichier temporaire HDFS est introuvable."
                )

            # 2. Sauvegarder l'ancien fichier s'il existe
            if self.exists(hdfs_path):

                if hdfs_path == HDFS_MANIFEST:
                    backup_dir = HDFS_BACKUPS
                else:
                    backup_dir = f"{HDFS_BACKUPS}/reports"

                self.run([
                    "-mkdir",
                    "-p",
                    backup_dir
                ])

                base_name = hdfs_path.rsplit("/", 1)[-1]

                backup_path = (
                    f"{backup_dir}/"
                    f"{base_name}_{timestamp()}_{uuid.uuid4().hex[:8]}"
                )

                self.run([
                    "-mv",
                    hdfs_path,
                    backup_path
                ])

                logger.info(
                    "[BACKUP] Ancien fichier conservé : %s",
                    backup_path
                )

            # 3. Publier le nouveau fichier
            self.run([
                "-mv",
                temp_path,
                hdfs_path
            ])

            logger.info(
                "[HDFS] Fichier publié : %s",
                hdfs_path
            )

        except Exception:

            # Restaurer l'ancien fichier si la publication échoue
            if (
                backup_path
                and self.exists(backup_path)
                and not self.exists(hdfs_path)
            ):

                try:
                    self.run([
                        "-mv",
                        backup_path,
                        hdfs_path
                    ])

                    logger.warning(
                        "[RESTAURATION] Ancien fichier restauré : %s",
                        hdfs_path
                    )

                except Exception as restore_error:

                    logger.critical(
                        "[ERREUR RESTAURATION] %s",
                        restore_error
                    )

            raise

        finally:

            # On ne supprime que notre propre fichier temporaire.
            if self.exists(temp_path):

                try:
                    self.run([
                        "-rm",
                        temp_path
                    ])
                except Exception:
                    logger.warning(
                        "[NETTOYAGE] Temporaire conservé : %s",
                        temp_path
                    )

    def upload_json(self, data, hdfs_path):

        content = json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        self.atomic_upload(
            content,
            hdfs_path
        )


# ============================================================
# CRAWLER
# ============================================================

class Crawler:

    def __init__(
        self,
        max_pages=100,
        delay=2,
        max_depth=2,
        max_links_per_page=30
    ):

        self.max_pages = max_pages
        self.delay = delay
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page

        self.hdfs = HdfsClient()

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/json;q=0.9,"
                "application/pdf;q=0.9,*/*;q=0.8"
            )
        })

        self.frontier = deque()
        self.enqueued = set()
        self.visited = set()

        self.failed = []
        self.skipped = []

        self.robots_cache = {}
        self.last_request_by_domain = {}

        self.previous_files = []
        self.manifest = []

        self.saved_urls = set()
        self.saved_hashes = {}
        self.previous_html_by_url = {}

        self.counters = {
            "html": 0,
            "pdf": 0,
            "json": 0
        }

        self.new_entries = 0
        self.downloaded_files = 0
        self.reused_files = 0

        self.load_previous_manifest()

    # ========================================================
    # REQUÊTES HTTP AVEC RETRY
    # ========================================================

    def download_with_retry(self, url):

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                response = self.session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True
                )

                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504
                ):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    response.close()

                    if retry_after and retry_after.isdigit():
                        wait = min(int(retry_after), 120)
                    else:
                        wait = min(2 ** attempt, 30)

                    logger.warning(
                        "[RETRY] HTTP temporaire %s : tentative %s/%s, attente %ss",
                        url,
                        attempt,
                        MAX_RETRIES,
                        wait
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                return response

            except requests.RequestException as exc:

                last_error = exc

                if attempt >= MAX_RETRIES:
                    break

                wait = min(2 ** attempt, 30)

                logger.warning(
                    "[RETRY] %s : tentative %s/%s, attente %ss : %s",
                    url,
                    attempt,
                    MAX_RETRIES,
                    wait,
                    exc
                )

                time.sleep(wait)

        raise requests.RequestException(
            f"Échec après {MAX_RETRIES} tentatives : {last_error}"
        )

    # ========================================================
    # MANIFESTE EXISTANT
    # ========================================================

    def load_previous_manifest(self):
        """
        Charge le manifeste existant.

        En cas de manifeste illisible, le crawler s'arrête.
        Il ne doit jamais remplacer un manifeste dont le contenu
        n'a pas pu être récupéré.
        """

        data = self.hdfs.read_json(
            HDFS_MANIFEST
        )

        if data is None:

            logger.info(
                "[INFO] Aucun ancien manifeste."
            )

            self.previous_files = []
            self.manifest = []

            return

        if isinstance(data, dict):
            files = data.get("files", [])

        elif isinstance(data, list):
            files = data

        else:
            raise ValueError(
                "Format du manifeste inconnu."
            )

        self.previous_files = [
            item
            for item in files
            if isinstance(item, dict)
        ]

        # CRITIQUE : conserver toutes les anciennes entrées.
        self.manifest = list(
            self.previous_files
        )

        for item in self.previous_files:

            source_url = item.get("source_url")
            hdfs_path = item.get("hdfs_path")
            file_type = item.get("file_type")
            digest = item.get("sha256")

            if source_url:

                normalized = normalize_url(
                    source_url
                )

                self.saved_urls.add(
                    normalized
                )

                if (
                    file_type == "html"
                    and hdfs_path
                ):

                    self.previous_html_by_url.setdefault(
                        normalized,
                        item
                    )

            if digest and hdfs_path:

                self.saved_hashes.setdefault(
                    digest,
                    hdfs_path
                )

            if file_type in self.counters:
                self.counters[file_type] += 1

        logger.info(
            "[INFO] Ancien manifeste chargé : %s entrée(s)",
            len(self.previous_files)
        )

    # ========================================================
    # ROBOTS.TXT
    # ========================================================

    def can_fetch(self, url):

        parsed = urlparse(url)

        domain_root = (
            f"{parsed.scheme}://{parsed.netloc}"
        )

        if domain_root not in self.robots_cache:

            robots_url = urljoin(
                domain_root,
                "/robots.txt"
            )

            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:

                response = self.download_with_retry(
                    robots_url
                )

                try:
                    if response.status_code == 200:
                        parser.parse(
                            response.text.splitlines()
                        )
                    else:
                        parser.parse([
                            "User-agent: *",
                            "Allow: /"
                        ])
                finally:
                    response.close()

            except requests.RequestException as exc:

                logger.warning(
                    "[ROBOTS] Impossible de vérifier %s : %s",
                    robots_url,
                    exc
                )

                # Politique prudente : ne pas crawler ce domaine
                # si robots.txt est inaccessible.
                parser.parse([
                    "User-agent: *",
                    "Disallow: /"
                ])

            self.robots_cache[domain_root] = parser

        return self.robots_cache[
            domain_root
        ].can_fetch(USER_AGENT, url)

    # ========================================================
    # DÉLAI ENTRE LES REQUÊTES
    # ========================================================

    def respect_delay(self, url):

        domain = (
            urlparse(url).netloc.lower()
        )

        current_time = time.time()

        previous_time = (
            self.last_request_by_domain.get(domain)
        )

        if previous_time is not None:

            elapsed = current_time - previous_time

            if elapsed < self.delay:

                time.sleep(
                    self.delay - elapsed
                )

        self.last_request_by_domain[domain] = time.time()

    # ========================================================
    # FILE D'URL
    # ========================================================

    def budget_left(self):

        return (
            len(self.visited) + len(self.frontier)
            < self.max_pages
        )

    def add_url(self, url, depth=0, priority=False):

        normalized = normalize_url(url)

        if not normalized:
            return False

        if normalized in self.visited:
            return False

        if normalized in self.enqueued:
            return False

        if not is_allowed_domain(normalized):
            return False

        # Une ancienne page HTML peut être réutilisée.
        # Les anciennes URL PDF/JSON sont ignorées.
        if (
            normalized in self.saved_urls
            and normalized not in self.previous_html_by_url
        ):
            return False

        if depth > self.max_depth:
            return False

        if not self.budget_left():
            return False

        item = (
            normalized,
            depth
        )

        if priority:
            self.frontier.appendleft(item)
        else:
            self.frontier.append(item)

        self.enqueued.add(normalized)

        return True

    def enqueue_links(self, links, depth):

        if depth > self.max_depth:
            return 0

        added = 0

        for link in links:

            if not self.budget_left():
                break

            if self.add_url(link, depth):
                added += 1

        return added

    def enqueue_entity(self, qid, depth):

        if not qid:
            return False

        return self.add_url(
            wikidata_entity_url(qid),
            depth,
            priority=True
        )

    # ========================================================
    # EXTRACTION DES LIENS
    # ========================================================

    def extract_links(self, html_content, page_url):

        soup = BeautifulSoup(
            html_content,
            "html.parser"
        )

        wiki_page = is_wikipedia(page_url)

        page_host = (
            urlparse(page_url).hostname or ""
        ).lower()

        if wiki_page:

            scope = (
                soup.find(id="mw-content-text")
                or soup
            )

            for element in scope.select(
                ".reference, .reflist, .mw-editsection, "
                "table.ambox, .hatnote"
            ):

                element.decompose()

        else:

            scope = soup

        discovered = []

        for anchor in scope.find_all(
            "a",
            href=True
        ):

            href = anchor.get(
                "href",
                ""
            ).strip()

            if not href:
                continue

            if href.startswith((
                "#",
                "mailto:",
                "tel:",
                "javascript:",
                "data:"
            )):
                continue

            absolute_url = normalize_url(
                urljoin(page_url, href)
            )

            if not is_allowed_domain(absolute_url):
                continue

            anchor_text = anchor.get_text(
                " ",
                strip=True
            )

            if is_wikipedia(absolute_url):

                host = (
                    urlparse(absolute_url).hostname or ""
                ).lower()

                if (
                    host == page_host
                    and wiki_article_ok(absolute_url)
                    and is_relevant(
                        absolute_url,
                        anchor_text
                    )
                ):

                    discovered.append(
                        absolute_url
                    )

                continue

            if (
                is_pdf_url(absolute_url)
                or is_relevant(
                    absolute_url,
                    anchor_text
                )
            ):

                discovered.append(
                    absolute_url
                )

        unique = list(
            dict.fromkeys(discovered)
        )

        return unique[
            :self.max_links_per_page
        ]

    # ========================================================
    # SAUVEGARDE DES DOCUMENTS
    # ========================================================

    def save_file(
        self,
        content,
        source_url,
        file_type,
        content_type="",
        depth=0,
        qid=None
    ):

        normalized = normalize_url(
            source_url
        )

        digest = sha256_bytes(content)

        # Si la même URL est déjà connue, ne pas créer
        # une nouvelle entrée identique.
        if normalized in self.saved_urls:

            logger.info(
                "[DOUBLON URL] %s",
                source_url
            )

            return None

        existing_path = self.saved_hashes.get(
            digest
        )

        if existing_path:

            hdfs_path = existing_path

            logger.info(
                "[DOUBLON CONTENU] %s",
                source_url
            )

        else:

            if file_type == "html":

                self.counters["html"] += 1

                filename = (
                    f"page_{digest[:20]}.html"
                )

                directory = HDFS_HTML

            elif file_type == "pdf":

                self.counters["pdf"] += 1

                filename = (
                    f"document_{digest[:20]}.pdf"
                )

                directory = HDFS_PDF

            elif file_type == "json":

                self.counters["json"] += 1

                if qid:
                    filename = (
                        f"{qid}_{digest[:20]}.json"
                    )
                else:
                    filename = (
                        f"entity_{digest[:20]}.json"
                    )

                directory = HDFS_JSON

            else:

                raise ValueError(
                    f"Type inconnu : {file_type}"
                )

            hdfs_path = (
                f"{directory}/{filename}"
            )

            # Protection contre une collision de nom.
            if self.hdfs.exists(hdfs_path):

                existing_content = self.hdfs.read_bytes(
                    hdfs_path
                )

                if sha256_bytes(existing_content) != digest:

                    raise RuntimeError(
                        "Collision de nom HDFS : "
                        f"{hdfs_path}"
                    )

            else:

                self.hdfs.put_bytes(
                    content,
                    hdfs_path
                )

                self.downloaded_files += 1

                logger.info(
                    "[HDFS] Nouveau fichier : %s",
                    hdfs_path
                )

            self.saved_hashes[digest] = hdfs_path

        entry = {

            "source_url": source_url,

            "normalized_url": normalized,

            "hdfs_path": hdfs_path,

            "file_type": file_type,

            "content_type": content_type,

            "sha256": digest,

            "size_bytes": len(content),

            "depth": depth,

            "downloaded_at": now_utc(),

            "corpus_version": CORPUS_VERSION,

            "collection_method": "crawler",

        }

        if qid:
            entry["qid"] = qid

        if is_wikipedia(source_url):

            entry["language"] = (
                urlparse(source_url).hostname or ""
            ).split(".")[0]

        self.manifest.append(entry)

        self.new_entries += 1
        self.saved_urls.add(normalized)

        if file_type == "html":

            self.previous_html_by_url[
                normalized
            ] = entry

        return entry

    # ========================================================
    # RÉUTILISATION DES ANCIENS HTML
    # ========================================================

    def process_cached_html(self, url, depth):

        normalized = normalize_url(url)

        entry = self.previous_html_by_url.get(
            normalized
        )

        if not entry:
            return

        hdfs_path = entry.get("hdfs_path")

        if not hdfs_path:
            return

        if not self.hdfs.exists(hdfs_path):

            logger.warning(
                "[CACHE] Fichier absent dans HDFS : %s",
                hdfs_path
            )

            self.failed.append({
                "url": url,
                "error": "Ancien fichier HTML absent dans HDFS",
                "timestamp": now_utc()
            })

            return

        try:

            html_content = self.hdfs.read_bytes(
                hdfs_path
            )

            source_url = (
                entry.get("source_url")
                or url
            )

            qid = (
                entry.get("qid")
                or (
                    extract_qid(html_content)
                    if is_wikipedia(source_url)
                    else None
                )
            )

            self.enqueue_entity(
                qid,
                depth
            )

            links = self.extract_links(
                html_content,
                source_url
            )

            added = self.enqueue_links(
                links,
                depth + 1
            )

            self.reused_files += 1

            logger.info(
                "[CACHE] Ancienne page réutilisée : %s",
                source_url
            )

            logger.info(
                "[CACHE] Nouveaux liens : %s",
                added
            )

        except Exception as exc:

            logger.warning(
                "[ERREUR CACHE] %s : %s",
                url,
                exc
            )

            self.failed.append({
                "url": url,
                "error": str(exc),
                "timestamp": now_utc()
            })

    # ========================================================
    # TRAITEMENT DES URL
    # ========================================================

    def process_url(self, url, depth):

        normalized = normalize_url(url)

        # Réutilisation du HTML existant.
        if normalized in self.saved_urls:

            self.process_cached_html(
                normalized,
                depth
            )

            return

        if not is_allowed_domain(normalized):

            self.skipped.append({
                "url": normalized,
                "reason": "Domaine non autorisé"
            })

            return

        is_api = is_wikidata_api(normalized)

        if (
            not is_api
            and not self.can_fetch(normalized)
        ):

            logger.info(
                "[ROBOTS.TXT] Accès interdit : %s",
                normalized
            )

            self.skipped.append({
                "url": normalized,
                "reason": "Interdit par robots.txt"
            })

            return

        self.respect_delay(normalized)

        response = None

        try:

            response = self.download_with_retry(
                normalized
            )

            final_url = normalize_url(
                response.url
            )

            if not is_allowed_domain(final_url):

                self.skipped.append({
                    "url": normalized,
                    "reason": "Redirection hors domaine autorisé"
                })

                return

            content = response.content

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                ).lower()
            )

            # -----------------------------------------------
            # PDF
            # -----------------------------------------------

            if looks_like_pdf(
                final_url,
                content_type,
                content
            ):

                self.save_file(
                    content=content,
                    source_url=final_url,
                    file_type="pdf",
                    content_type=content_type,
                    depth=depth
                )

                logger.info(
                    "[PDF] %s",
                    final_url
                )

                return

            # -----------------------------------------------
            # JSON
            # -----------------------------------------------

            if (
                "json" in content_type
                or is_api
            ):

                match = QID_URL_RE.search(
                    final_url
                )

                qid = (
                    match.group(1)
                    if match
                    else None
                )

                self.save_file(
                    content=content,
                    source_url=final_url,
                    file_type="json",
                    content_type=content_type,
                    depth=depth,
                    qid=qid
                )

                logger.info(
                    "[WIKIDATA] %s",
                    final_url
                )

                return

            # -----------------------------------------------
            # HTML
            # -----------------------------------------------

            is_html = (
                "text/html" in content_type
                or "application/xhtml+xml" in content_type
            )

            if not is_html:

                self.skipped.append({
                    "url": final_url,
                    "reason": (
                        f"Type non pris en charge : {content_type}"
                    )
                })

                return

            qid = (
                extract_qid(content)
                if is_wikipedia(final_url)
                else None
            )

            entry = self.save_file(
                content=content,
                source_url=final_url,
                file_type="html",
                content_type=content_type,
                depth=depth,
                qid=qid
            )

            # Ne récupérer les fiches Wikidata que pour les
            # pages Wikipédia pertinentes pour le projet.
            if is_wikipedia(final_url):

                soup = BeautifulSoup(
                    content,
                    "html.parser"
                )

                title = (
                    soup.title.get_text(" ", strip=True)
                    if soup.title
                    else ""
                )

                page_text = soup.get_text(
                    " ",
                    strip=True
                )[:12000]

                relevant_page = is_relevant(
                    final_url,
                    title + " " + page_text
                )

                if relevant_page:
                    self.enqueue_entity(
                        qid,
                        depth
                    )

            links = self.extract_links(
                content,
                final_url
            )

            added = self.enqueue_links(
                links,
                depth + 1
            )

            logger.info(
                "[HTML] %s",
                final_url
            )

            logger.info(
                "[QID] %s",
                qid or "-"
            )

            logger.info(
                "[LIENS] %s nouveau(x) lien(s)",
                added
            )

        except requests.RequestException as exc:

            logger.warning(
                "[ERREUR HTTP] %s : %s",
                normalized,
                exc
            )

            self.failed.append({
                "url": normalized,
                "error": str(exc),
                "timestamp": now_utc()
            })

        except Exception as exc:

            logger.warning(
                "[ERREUR] %s : %s",
                normalized,
                exc
            )

            self.failed.append({
                "url": normalized,
                "error": str(exc),
                "timestamp": now_utc()
            })

        finally:

            if response is not None:
                response.close()

    # ========================================================
    # RAPPORT
    # ========================================================

    def build_report(self):

        return {

            "generated_at": now_utc(),

            "corpus_version": CORPUS_VERSION,

            "max_pages": self.max_pages,

            "max_depth": self.max_depth,

            "max_links_per_page": self.max_links_per_page,

            "delay_seconds": self.delay,

            "processed_urls": len(self.visited),

            "previous_manifest_entries": len(
                self.previous_files
            ),

            "new_manifest_entries": self.new_entries,

            "total_manifest_entries": len(
                self.manifest
            ),

            "downloaded_physical_files": self.downloaded_files,

            "reused_cached_files": self.reused_files,

            "entries_by_type": dict(
                self.counters
            ),

            "failed_count": len(self.failed),

            "skipped_count": len(self.skipped),

            "failed": self.failed,

            "skipped": self.skipped

        }

    # ========================================================
    # FINALISATION
    # ========================================================

    def finalize(self):

        logger.info(
            "[INFO] Préparation du manifeste..."
        )

        # self.manifest contient les anciennes entrées
        # plus les nouvelles entrées.
        #
        # upload_json effectue automatiquement :
        # - création d'un fichier temporaire ;
        # - sauvegarde de l'ancien manifeste ;
        # - renommage du nouveau manifeste ;
        # - restauration en cas d'échec.

        self.hdfs.upload_json(
            self.manifest,
            HDFS_MANIFEST
        )

        report = self.build_report()

        report_path = (
            f"{HDFS_PROCESSED}/crawler_report_{timestamp()}.json"
        )

        self.hdfs.upload_json(
            report,
            report_path
        )

        # Cette copie est également publiée de façon sécurisée.
        self.hdfs.upload_json(
            report,
            f"{HDFS_PROCESSED}/crawler_run_report.json"
        )

        logger.info(
            "[HDFS] Manifeste publié : %s",
            HDFS_MANIFEST
        )

        logger.info(
            "[HDFS] Rapport : %s",
            report_path
        )

    # ========================================================
    # LANCEMENT
    # ========================================================

    def run(self, seed_urls):

        self.hdfs.ensure_directories()

        for seed in seed_urls:

            self.add_url(
                seed,
                depth=0
            )

        logger.info("")
        logger.info("==========================================")
        logger.info(" DÉMARRAGE DU CRAWLER GRAPHRAG MAROC")
        logger.info("==========================================")

        logger.info(
            "Anciennes entrées : %s",
            len(self.previous_files)
        )

        logger.info(
            "Nouvelles sources de départ : %s",
            len(seed_urls)
        )

        logger.info(
            "Limite URL : %s",
            self.max_pages
        )

        logger.info(
            "Profondeur maximale : %s",
            self.max_depth
        )

        logger.info(
            "Liens par page : %s",
            self.max_links_per_page
        )

        logger.info(
            "HDFS : %s",
            HDFS_BASE
        )

        logger.info("")

        while (
            self.frontier
            and len(self.visited) < self.max_pages
        ):

            url, depth = self.frontier.popleft()

            if url in self.visited:
                continue

            logger.info(
                "[CRAWL] (%s/%s) profondeur %s : %s",
                len(self.visited) + 1,
                self.max_pages,
                depth,
                url
            )

            self.process_url(
                url,
                depth
            )

            self.visited.add(url)

            logger.info("")

        self.finalize()

        logger.info("")
        logger.info("==========================================")
        logger.info(" BILAN DE LA COLLECTE")
        logger.info("==========================================")

        logger.info(
            "URL traitées : %s",
            len(self.visited)
        )

        logger.info(
            "Anciennes entrées conservées : %s",
            len(self.previous_files)
        )

        logger.info(
            "Nouvelles entrées : %s",
            self.new_entries
        )

        logger.info(
            "Total des entrées : %s",
            len(self.manifest)
        )

        logger.info(
            "Fichiers physiques téléchargés : %s",
            self.downloaded_files
        )

        logger.info(
            "Pages HTML réutilisées : %s",
            self.reused_files
        )

        logger.info(
            "Fichiers par type : %s",
            self.counters
        )

        logger.info(
            "Échecs : %s",
            len(self.failed)
        )

        logger.info(
            "URL ignorées : %s",
            len(self.skipped)
        )

        logger.info("==========================================")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Crawler GraphRAG Maroc : collecte incrémentale "
            "des personnalités politiques, entreprises, "
            "partis et institutions."
        )
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max-links-per-page",
        type=int,
        default=30
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        default=SEED_URLS
    )

    args = parser.parse_args()

    if args.max_pages < 1:
        parser.error(
            "--max-pages doit être supérieur à 0."
        )

    if args.max_depth < 0:
        parser.error(
            "--max-depth ne peut pas être négatif."
        )

    if args.max_links_per_page < 1:
        parser.error(
            "--max-links-per-page doit être supérieur à 0."
        )

    if args.delay < 0:
        parser.error(
            "--delay ne peut pas être négatif."
        )

    crawler = Crawler(
        max_pages=args.max_pages,
        delay=args.delay,
        max_depth=args.max_depth,
        max_links_per_page=args.max_links_per_page
    )

    try:

        crawler.run(
            args.seeds
        )

    except KeyboardInterrupt:

        logger.warning(
            "[ARRÊT] Interruption manuelle."
        )

        # On publie les entrées déjà collectées.
        try:
            crawler.finalize()
        except Exception as exc:
            logger.error(
                "[ERREUR] Sauvegarde finale impossible : %s",
                exc
            )

    except Exception as exc:

        logger.exception(
            "[ERREUR FATALE] %s",
            exc
        )

        raise

    finally:

        crawler.session.close()


if __name__ == "__main__":
    main()
