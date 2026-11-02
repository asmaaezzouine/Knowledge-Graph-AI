"""
Configuration et fonctions communes aux scripts du corpus
« liens entre personnalités publiques ».

Utilisé par : audit_corpus.py, filtrer_corpus.py, prepare_selection.py,
preparer_corpus_graphrag.py
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

SOURCE = Path("corpus_json")            # sorties de extract_text.py (hdfs get)
AUDIT_REPORT = Path("audit_report.json")  # produit par audit_corpus.py
MIN_TEXT_LENGTH = 300

TEXT_PATTERNS = ("page_*.json", "article_*.json")
ENTITY_PATTERN = "entity_*.json"

# Pages collectées par le crawler mais sans lien avec le réseau
# personnes / partis / entreprises. Comparées au titre de la page.
TITRES_HORS_PERIMETRE = {
    "armoiries du maroc",
    "rabat",
    "quinquennat",
    "arabe",
    "langues berbères",
    # Concepts généraux, sans personnes ni entreprises
    "scrutin proportionnel plurinominal",
    "chambre basse",
    "non-inscrit",
    "opposition parlementaire",
    "forme juridique",
    "siège social",
    "directeur général",
    "secteur économique",
    "énergie",
    "gaz",
    "média",
    "médias",
    "bien immobilier",
    "immobilier",
    "tourisme",
    "filiale",
    # Pages géographiques trop larges (milliers de lignes, aucune relation utile)
    "maroc",
    "casablanca",
    "agadir",
}

_MOIS = (
    "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|"
    "novembre|décembre"
)
# « 24 septembre », « Octobre 2026 », « 2021 », « 1er mai »...
_TITRE_CALENDRIER = re.compile(
    rf"^(?:\d{{1,2}}(?:er)? )?(?:{_MOIS})(?: \d{{3,4}})?$|^\d{{3,4}}$"
)

# Thèmes recherchés (minuscules, apostrophes droites).
THEMES = {
    "politique": [
        "ministre", "ministère", "gouvernement", "parlement", "député",
        "députée", "chambre des représentants", "conseiller", "parti",
        "élections", "élu", "maire", "wali", "secrétaire général",
        "nomination", "وزير", "الحكومة", "البرلمان", "حزب", "انتخابات",
    ],
    "entreprises_affaires": [
        "entreprise", "société", "holding", "groupe", "actionnaire",
        "administrateur", "conseil d'administration",
        "président-directeur général", "pdg", "directeur général",
        "homme d'affaires", "femme d'affaires", "fondateur", "filiale",
        "capital", "bourse", "شركة", "رجل أعمال",
    ],
    "liens_personnels_publics": [
        "épouse", "époux", "conjoint", "fils de", "fille de", "frère",
        "soeur", "sœur", "famille", "beau-frère", "gendre",
    ],
    "conflits_interets": [
        "conflit d'intérêts", "déclaration de patrimoine", "marché public",
        "marchés publics", "incompatibilité", "cumul", "pantouflage",
        "commission d'enquête",
    ],
    "presence_en_ligne": [
        "compte officiel", "twitter", "facebook", "instagram", "linkedin",
        "site officiel", "réseaux sociaux",
    ],
}


def _compile(words):
    return [
        (w, re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)"))
        for w in words
    ]


_THEME_REGEX = {theme: _compile(words) for theme, words in THEMES.items()}


def normaliser_texte(texte):
    texte = str(texte).replace("\u2019", "'").replace("\xa0", " ").lower()
    return re.sub(r"\s+", " ", texte).strip()


def sha256_texte(texte):
    return hashlib.sha256(normaliser_texte(texte).encode("utf-8")).hexdigest()


def domaine(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def detecter_themes(titre, texte):
    """Retourne {theme: [mots trouvés]} pour les thèmes détectés."""
    contenu = normaliser_texte(f"{titre} {texte}")
    resultat = {}

    for theme, regexes in _THEME_REGEX.items():
        trouves = [mot for mot, rx in regexes if rx.search(contenu)]
        if trouves:
            resultat[theme] = trouves

    return resultat


def lister_textes(source=SOURCE):
    chemins = set()
    for pattern in TEXT_PATTERNS:
        chemins.update(source.glob(pattern))
    return sorted(chemins)


def lister_entites(source=SOURCE):
    return sorted(source.glob(ENTITY_PATTERN))


def charger_document(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    titre = str(data.get("title", "")).strip()
    texte = str(data.get("text", data.get("content", ""))).strip()
    url = str(data.get("url") or data.get("source_url") or "").strip()

    return {
        "id": path.stem,
        "file": path.name,
        "title": titre,
        "author": data.get("author", ""),
        "source": data.get("source") or domaine(url),
        "url": url,
        "qid": data.get("qid", ""),
        "language": data.get("language", ""),
        "text": texte,
        "text_length": len(texte),
    }


def charger_exclusions():
    """Lit les exclusions produites par audit_corpus.py ({fichier: raison})."""
    if not AUDIT_REPORT.exists():
        print(
            f"[ATTENTION] {AUDIT_REPORT} introuvable : "
            "lancez d'abord audit_corpus.py (aucune exclusion appliquée)."
        )
        return {}

    data = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    return data.get("excluded", {})


def est_hors_perimetre(titre):
    """Vrai si le titre est une page de calendrier ou figure dans TITRES_HORS_PERIMETRE."""
    titre_normalise = normaliser_texte(titre)

    if _TITRE_CALENDRIER.match(titre_normalise):
        return True

    return titre_normalise in {
        normaliser_texte(t) for t in TITRES_HORS_PERIMETRE
    }


def charger_qids_exclus():
    """QID des pages hors périmètre (produits par audit_corpus.py)."""
    if not AUDIT_REPORT.exists():
        return set()

    data = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    return set(data.get("excluded_qids", []))