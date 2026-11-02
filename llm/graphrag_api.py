"""
GraphRAG Maroc : API v6.3.2
Base v6.3.1 (correctifs LLM) + diagnostics de chargement intégrés.

CHANGEMENTS v6.3.2 (par rapport à v6.3.1) :
  - Warning au démarrage si GRAPH_LABEL / RELATION_SOURCE sont renseignés
    (cause n°1 de relations perdues : le .env forçait le sous-graphe).
  - Nouveaux endpoints de diagnostic :
      GET /debug/env                 → variables effectives du loader
      GET /debug/chargement-funnel   → compte à chaque étape du loader
      GET /debug/trace-relation/...  → trace une relation précise
  - Documentation inline mise à jour pour rappeler que GRAPH_LABEL et
    RELATION_SOURCE doivent rester VIDES pour charger tout le graphe métier.

Principe :
  1. Le graphe Neo4j est chargé en mémoire (toutes les entités :Entite).
  2. Un routeur d'intentions (règles + fallback LLM) choisit le traitement.
  3. Résultats calculés en Python (déterministe, traçable).
  4. Le LLM (Groq, facultatif) rédige une synthèse à partir des résultats.

PÉRIMÈTRE MÉTIER STRICT :
  - nœuds exclus : Customer, Product, GraphRAGEntity
  - nœuds de preuve : Evidence, GraphRAGMarocEvidence
  - relations exclues : KNOWS, GRAPHRAG_RELATION, PURCHASED
  - relations de traçabilité : EVIDENCE_FOR, EVIDENCE_TARGET
  - aucune donnée n'est jamais modifiée dans Neo4j.

Installation :
    pip install fastapi uvicorn neo4j python-dotenv groq
Lancement :
    uvicorn graph_api:app --host 0.0.0.0 --port 8000 --reload

ATTENTION .env — ces deux variables doivent rester VIDES :
    GRAPH_LABEL=
    RELATION_SOURCE=
Sinon, les nœuds et relations qui ne portent pas ces marqueurs sont
silencieusement exclus (par exemple les ACTIONNAIRE_DE).
"""

import os
import re
import json
import time
import logging
import hmac
import threading
import unicodedata
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Query as Neo4jQuery
from neo4j.exceptions import Neo4jError

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# 1. CONFIGURATION
# ============================================================

VERSION = "6.3.2"
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


def _bool(nom, defaut):
    return os.getenv(nom, defaut).strip().lower() in (
        "1", "true", "oui", "yes"
    )


NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j").strip()

# v6.3.2 : ces deux variables doivent rester VIDES pour charger tout le
# graphe métier. Ne les renseigner que pour restreindre volontairement.
GRAPH_LABEL = os.getenv("GRAPH_LABEL", "").strip()
RELATION_SOURCE = os.getenv("RELATION_SOURCE", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_MODELS = [
    x.strip()
    for x in os.getenv(
        "GROQ_MODELS",
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b,openai/gpt-oss-120b"),
    ).split(",")
    if x.strip()
] or ["openai/gpt-oss-20b"]

GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2500"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
MAX_LLM_LIGNES = int(os.getenv("MAX_LLM_LIGNES", "15"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.0"))
MAX_ENTITES_PROFIL = int(os.getenv("MAX_ENTITES_PROFIL", "3"))
CHEMIN_MAX = int(os.getenv("CHEMIN_MAX", "4"))
CHEMINS_PAR_PAIRE = int(os.getenv("CHEMINS_PAR_PAIRE", "5"))
HUB_DEGRE_MAX = int(os.getenv("HUB_DEGRE_MAX", "150"))
LIMITE_DEFAUT = int(os.getenv("LIMITE_DEFAUT", "40"))
LIMITE_API_MAX = int(os.getenv("LIMITE_API_MAX", "500"))

EXCLURE_ETRANGERS = _bool("EXCLURE_ETRANGERS", "1")

NEO4J_TIMEOUT = float(os.getenv("NEO4J_TIMEOUT", "30"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
CORS_ORIGINS = [
    x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()
]
LIMITE_PREUVES = int(os.getenv("LIMITE_PREUVES", "50"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag")

driver = None
groq_client = None
groq_model_lock = threading.Lock()
groq_model_index = 0

GRAPHE = None
graphe_lock = threading.Lock()
reload_lock = threading.Lock()


# ============================================================
# 2. NORMALISATION
# ============================================================

def normaliser(texte) -> str:
    texte = str(texte or "").lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = texte.replace("_", " ")
    texte = re.sub(r"[^a-z0-9\s-]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def type_normalise(valeur) -> str:
    return normaliser(valeur).upper()


@lru_cache(maxsize=32768)
def motif_expression(expression):
    return re.compile(
        r"(?<![a-z0-9])"
        + re.escape(normaliser(expression))
        + r"s?(?![a-z0-9])"
    )


def contient_expression(texte, expressions) -> bool:
    texte = normaliser(texte)
    return any(motif_expression(x).search(texte) for x in expressions)


def est_qid_brut(label) -> bool:
    return bool(re.fullmatch(r"q[0-9]+", normaliser(label)))


def convertir_liste(valeur):
    if valeur is None:
        return []
    if isinstance(valeur, (list, tuple)):
        return [x for x in valeur if x is not None]
    if isinstance(valeur, str):
        if not valeur.strip():
            return []
        if valeur.strip()[:1] in ("[", "{"):
            try:
                resultat = json.loads(valeur)
                if isinstance(resultat, list):
                    return resultat
                if isinstance(resultat, dict):
                    return [resultat]
            except (json.JSONDecodeError, TypeError):
                pass
        return [valeur]
    return []


def tronquer(valeur, n):
    texte = str(valeur)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


# ============================================================
# 3. VOCABULAIRE
# ============================================================

CATEGORIES = {
    "PERSONNE": {"PERSONNE", "PERSON", "INDIVIDU", "PERSONNE PHYSIQUE",
                 "HUMAIN"},
    "PARTI": {"PARTI", "PARTI POLITIQUE", "POLITICAL PARTY"},
    "INSTITUTION": {
        "INSTITUTION", "ADMINISTRATION", "GOUVERNEMENT",
        "INSTITUTION POLITIQUE", "INSTITUTION PUBLIQUE", "PARLEMENT",
        "ASSEMBLEE", "ASSEMBLEE NATIONALE", "CHAMBRE DES REPRESENTANTS",
        "CHAMBRE DES CONSEILLERS", "COALITION POLITIQUE",
    },
    "ENTREPRISE": {
        "ENTREPRISE", "SOCIETE", "SOCIETE COMMERCIALE",
        "ORGANISATION COMMERCIALE", "GROUPE", "GROUPE ECONOMIQUE",
        "COMPANY", "CORPORATION", "BUSINESS", "BANQUE", "HOLDING",
    },
    "ORGANISATION": {
        "ORGANISATION", "ORGANISATION PATRONALE", "ASSOCIATION",
        "UNIVERSITE", "MEDIA", "ORGANISME", "SYNDICAT",
    },
    "FONCTION": {"FONCTION", "POSTE"},
    "LIEU": {"LIEU", "VILLE", "PAYS"},
    "EVENEMENT": {"EVENEMENT"},
    "ENTITE": {"ENTITE"},
}

CATEGORIE_LIBELLE = {
    "PERSONNE": "Personnes",
    "PARTI": "Partis politiques",
    "INSTITUTION": "Institutions",
    "ENTREPRISE": "Entreprises",
    "ORGANISATION": "Organisations",
    "FONCTION": "Fonctions",
    "LIEU": "Lieux",
    "EVENEMENT": "Événements",
    "ENTITE": "Entités non typées",
    "AUTRE": "Autres",
}

FAMILIAUX = ["PARENT_DE", "FRATRIE_DE", "CONJOINT_DE", "FAMILLE_DE",
             "MARIE_A"]

RELATIONS_CANONIQUES = {
    "PRESIDENT_DE": "PRESIDE",
    "ADHERENT_DE": "MEMBRE_DE",
    "DIRECTEUR_DE": "DIRIGE",
}

POIDS_POLITIQUES = {
    "OCCUPE_FONCTION": 3,
    "PRESIDE": 3,
    "ELU_A": 3,
    "PARTICIPE_AU_GOUVERNEMENT": 3,
    "MEMBRE_DE": 2,
    "CANDIDAT_A": 2,
    "NOMME_A": 2,
    "NOMME_PAR": 2,
    "DIRIGE": 2,
    "FONDATEUR_DE": 2,
    "COALITION_AVEC": 1,
    "PARTICIPE_A": 1,
    "OPPOSE_A": 1,
    "SUCCEDE_A": 1,
}

RELATIONS_POLITIQUES = set(POIDS_POLITIQUES) | {
    "GOUVERNEMENT_SORTANT", "CHEF_SUPREME", "AU_SEIN_DE", "COORDINATEUR_DE",
}

RELATIONS_ECONOMIQUES = {
    "DIRIGE", "FONDATEUR_DE", "ACTIONNAIRE_DE", "PRESIDE",
    "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "FILIALE_DE", "APPARTIENT_A",
    "FINANCE",
}

LABELS_EXCLUS = {"Customer", "Product", "GraphRAGEntity"}
LABELS_PREUVE = {"Evidence", "GraphRAGMarocEvidence"}
LABELS_METIER = {
    "Entite", "Personne", "Entreprise", "Parti", "Institution",
    "Fonction", "Autre",
}
LABEL_PROVENANCE = "GraphRAGMaroc"

RELATIONS_PREUVE = {"EVIDENCE_FOR", "EVIDENCE_TARGET"}
RELATIONS_EXCLUES = {"KNOWS", "GRAPHRAG_RELATION", "PURCHASED"}
RELATIONS_TECHNIQUES = RELATIONS_PREUVE | RELATIONS_EXCLUES

RELATIONS_ENTREPRISE_PERSONNE = {
    "DIRIGE", "FONDATEUR_DE", "ACTIONNAIRE_DE", "PRESIDE",
    "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "MEMBRE_DE", "OCCUPE_FONCTION",
}

EXCLUSIONS_ENTREPRISE_ORGANISATION = [
    "academie", "universite", "ecole", "faculte", "musee", "bibliotheque",
    "societe savante", "association culturelle",
]
EXCLUSIONS_ENTREPRISE_INSTITUTION = [
    "ministere", "parlement", "chambre des representants",
    "chambre des conseillers", "gouvernement",
]

NATURE_ORDRE = {
    "politique": 0, "economique": 1, "familial": 2, "institutionnel": 3,
    "succession": 4, "autre": 5, "non_qualifie": 6,
}

NATURE_LIBELLE = {
    "politique": "politique", "economique": "économique",
    "familial": "familial", "institutionnel": "institutionnel",
    "succession": "succession", "autre": "autre",
    "non_qualifie": "non qualifié",
}

FONCTIONS_POLITIQUES = [
    "ministre", "premier ministre", "chef du gouvernement",
    "cheffe du gouvernement", "president du conseil de gouvernement",
    "depute", "deputee", "parlementaire", "groupe parlementaire",
    "senateur", "president de la republique", "president de la chambre",
    "president du parlement", "chambre des representants",
    "president de region", "president de commune", "maire", "bourgmestre",
    "wali", "gouverneur", "pacha", "ambassadeur", "secretaire d etat",
    "membre du gouvernement", "roi", "chef de l etat", "grand vizir",
    "conseil du trone", "forces armees royales", "chef supreme",
    "secretaire a la defense", "president des etats unis",
    "president de la federation de russie",
    "vice president de la commission europeenne", "depute europeen",
]

FONCTIONS_ECONOMIQUES = [
    "directeur", "directrice", "directeur general", "directrice generale",
    "pdg", "president directeur general", "president du directoire",
    "administrateur", "administratrice", "gerant", "gerante",
    "actionnaire", "entrepreneur", "chef d entreprise",
    "conseil d administration", "holding", "groupe", "societe", "banque",
    "entreprise", "entreprises", "cgem",
    "confederation generale des entreprises", "groupement des petroliers",
    "siger", "directeur financier",
]

ACRONYMES_PARTIS = [
    "pjd", "rni", "pam", "usfp", "istiqlal", "unfp", "pi", "parti",
    "mouvement populaire", "bureau politique",
]

INSTITUTIONS_POLITIQUES = [
    "gouvernement", "ministere", "parlement", "chambre des representants",
    "chambre des conseillers", "assemblee", "conseil des ministres",
    "cabinet royal", "conseil de gouvernement", "presidence",
    "secretariat general du gouvernement", "monarchie", "conseil du trone",
]

MOTS_ENTREPRISE = [
    "societe", "entreprise", "groupe", "group", "holding", "banque", "bank",
    "compagnie", "company", "corporation", "industrie", "industries",
    "assurance", "investment", "investissement", "patronat",
    "confederation generale des entreprises", "sarl", "gaz", "telecom",
]

MOTS_NON_ECONOMIQUES = [
    "ministere", "gouvernement", "parlement", "chambre des", "commune",
    "wilaya", "prefecture", "bureau politique", "parti", "academie",
    "universite", "ecole", "faculte", "institut", "association",
    "federation", "tribunal", "cour des",
]

NATIONALITES_ETRANGERES = [
    "portugais", "portugaise", "timorais", "polonais", "polonaise", "grec",
    "grecque", "camerounais", "camerounaise", "luxembourgeois",
    "luxembourgeoise", "francais", "francaise", "allemand", "allemande",
    "belge", "espagnol", "espagnole", "canadien", "canadienne",
    "autrichien", "autrichienne", "neerlandais", "neerlandaise",
    "americain", "americaine", "britannique", "anglais", "anglaise",
    "italien", "italienne", "suedois", "suedoise", "danois", "danoise",
    "norvegien", "norvegienne", "finlandais", "finlandaise", "irlandais",
    "irlandaise", "suisse", "russe", "chinois", "chinoise", "japonais",
    "japonaise", "turc", "turque", "algerien", "algerienne", "tunisien",
    "tunisienne", "egyptien", "egyptienne", "senegalais", "senegalaise",
    "ivoirien", "ivoirienne", "europeen", "europeenne", "bresilien",
    "bresilienne", "mexicain", "mexicaine", "libanais", "libanaise",
    "saoudien", "saoudienne", "hongrois", "hongroise", "roumain",
    "roumaine", "bulgare", "croate", "israelien", "israelienne", "indien",
    "indienne", "neerlandais", "estonien", "lituanien", "letton",
]

PROFIL_POLITIQUE_DESCRIPTION = [
    "homme politique", "femme politique", "personnalite politique",
    "politicien", "politicienne", "homme d etat", "ministre",
    "premier ministre", "depute", "parlementaire", "senateur",
    "chef du gouvernement", "cheffe du gouvernement",
    "secretaire general du parti", "president du parti", "roi",
]

MOTS_CATEGORIES = {
    "PARTI": ["parti", "partis", "formation politique",
              "formations politiques"],
    "ENTREPRISE": ["entreprise", "societe", "holding", "groupe economique",
                   "banque", "compagnie"],
    "INSTITUTION": ["institution", "ministere", "administration",
                    "organe", "organes"],
    "FONCTION": ["fonction", "poste", "mandat"],
    "ORGANISATION": ["organisation", "association", "syndicat",
                     "universite", "media"],
    "PERSONNE": ["personne", "personnalite", "responsable", "politicien",
                 "acteur", "individu", "homme politique",
                 "femme politique", "dirigeant", "ministre", "depute",
                 "maire", "ambassadeur"],
}

MOTS_ROLES = [
    (["preside", "presider", "president", "presidente", "presidence"],
     ["PRESIDE"]),
    (["dirige", "diriger", "dirigeant", "directeur", "directrice", "pdg",
      "patron", "gerant", "gere"], ["DIRIGE", "PRESIDE"]),
    (["membre", "adherent", "appartient", "appartiennent"], ["MEMBRE_DE"]),
    (["fonde", "fondateur", "fondatrice", "cofondateur"],
     ["FONDATEUR_DE"]),
    (["actionnaire", "detient", "detiennent", "participation", "capital"],
     ["ACTIONNAIRE_DE"]),
    (["administrateur", "conseil d administration"],
     ["ADMINISTRATEUR_DE"]),
    (["employe", "salarie", "travaille", "travaillent"], ["EMPLOYE_PAR"]),
    (["filiale", "maison mere", "societe mere"], ["FILIALE_DE"]),
    (["coalition", "allie", "alliance"], ["COALITION_AVEC"]),
    (["participe au gouvernement", "participent au gouvernement"],
     ["PARTICIPE_AU_GOUVERNEMENT"]),
    (["famille", "familial", "frere", "soeur", "fils", "fille", "pere",
      "epouse", "epoux", "conjoint", "conjointe", "parent", "enfant"],
     FAMILIAUX),
    (["occupe", "occupent", "fonction", "poste", "ministre", "depute",
      "maire", "ambassadeur"], ["OCCUPE_FONCTION"]),
]

TYPES_RELATION_NOMMES = {
    "ACTIONNAIRE_DE": [
        "actionnaire de", "actionnaire d", "actionnariat",
        "actionnaires de", "actionnaires d",
    ],
    "FONDATEUR_DE": [
        "fondateur de", "fondateur d", "fondateur du", "fondateurs de",
        "fondateurs d", "fondateurs du", "fondatrice de", "fondatrice d",
        "fondatrice du", "cofondateur", "co-fondateur",
    ],
    "DIRIGE": ["dirige", "dirigé", "dirigee", "direction de", "dirigent"],
    "PRESIDE": ["preside", "préside", "président de", "présidente de"],
    "OCCUPE_FONCTION": [
        "occupe fonction", "occupe le poste", "occupe le mandat",
        "fonction de", "poste de", "mandat de",
    ],
    "MEMBRE_DE": [
        "membre de", "membre du", "membre des", "adherent de",
        "adhérent de", "appartient au parti",
    ],
    "ADMINISTRATEUR_DE": [
        "administrateur de", "administratrice de",
        "conseil d'administration",
    ],
    "EMPLOYE_PAR": [
        "employe par", "employé par", "salarie de", "salarié de",
        "travaille pour",
    ],
    "FILIALE_DE": ["filiale de", "societe mere", "maison mere"],
    "COALITION_AVEC": ["coalition avec", "allie a", "alliance avec"],
    "ELU_A": ["elu a", "élu à", "elu au", "elu aux"],
    "CANDIDAT_A": [
        "candidat a", "candidat à", "candidat au", "candidature",
    ],
    "PARTICIPE_A": ["participe a", "participation a"],
    "SUCCEDE_A": [
        "succede a", "succède à", "succede au", "succede aux",
    ],
    "PARENT_DE": [
        "parent de", "parente de", "pere de", "mere de", "fils de",
        "fille de", "frere de", "soeur de", "epouse de", "epoux de",
    ],
}


def detecter_types_relation_nommes(question):
    t = normaliser(question)
    trouves = []
    for rel_type, motifs in TYPES_RELATION_NOMMES.items():
        if contient_expression(t, motifs):
            trouves.append(rel_type)
    for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", question):
        if token in TYPES_RELATION_NOMMES and token not in trouves:
            trouves.append(token)
    return trouves


MOTS_FONCTIONS_QUESTION = {
    "ministre": ["ministre", "premier ministre"],
    "depute": ["depute", "parlementaire"],
    "maire": ["maire"],
    "ambassadeur": ["ambassadeur"],
    "chef du gouvernement": ["chef du gouvernement",
                             "cheffe du gouvernement", "premier ministre"],
    "gouverneur": ["gouverneur", "wali"],
}

MOTS_POLITIQUE = [
    "politique", "parti", "gouvernement", "ministre", "depute",
    "parlementaire", "politicien", "elu", "coalition", "maire",
]
MOTS_ECONOMIQUE = [
    "entreprise", "societe", "economique", "actionnaire", "holding",
    "banque", "pdg", "patron", "homme d affaires", "groupe economique",
    "liens avec des entreprises", "affaires",
]
MOTS_CHEMIN = [
    "lien", "relie", "connecte", "connait", "chemin", "commun",
    "ensemble", "indirect", "entre", "rapport",
]
MOTS_CLASSEMENT = [
    "plus connecte", "centralite", "classement", "principaux acteurs",
    "acteurs les plus", "plus influent", "plus importants",
    "les plus connectes", "degre",
]
MOTS_STATS = [
    "combien", "statistique", "nombre de", "nombre d", "taille du graphe",
    "que contient", "contenu du graphe", "resume du graphe",
    "vue d ensemble", "apercu", "couverture", "qualite des donnees",
]
MOTS_DOCUMENTE = [
    "documente", "preuve", "source", "justifie", "extrait", "etaye",
    "verifie", "confirme",
]
MOTS_RESEAUX = ["reseau", "reseaux", "cartographie"]
MOTS_PARCOURS = [
    "parcours", "carriere", "trajectoire", "historique", "biographie",
    "cursus", "evolution",
]
MOTS_CONNEXIONS = [
    "connexion", "directes et indirectes", "relations indirectes",
    "liens indirects", "reseau de relations",
]
MOTS_RELATIONS = ["relation", "lien", "connexion"]

MAX_PARCOURS_PERSONNES = int(os.getenv("MAX_PARCOURS_PERSONNES", "8"))
GROUPE_MAX_PERSONNES = int(os.getenv("GROUPE_MAX_PERSONNES", "40"))

TYPES_PARCOURS = {
    "OCCUPE_FONCTION", "MEMBRE_DE", "PRESIDE", "DIRIGE", "FONDATEUR_DE",
    "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "ACTIONNAIRE_DE", "COALITION_AVEC",
    "PARTICIPE_AU_GOUVERNEMENT", "NOMME_A", "ELU_A", "CANDIDAT_A",
}
TYPES_PARTI = {"MEMBRE_DE", "PRESIDE", "DIRIGE", "FONDATEUR_DE",
               "COALITION_AVEC"}
TYPES_ENTREPRISE = {"DIRIGE", "PRESIDE", "ACTIONNAIRE_DE",
                    "ADMINISTRATEUR_DE", "FONDATEUR_DE", "FILIALE_DE",
                    "EMPLOYE_PAR"}

MOTS_QUESTION_GENERIQUES = {
    "maroc", "marocain", "marocains", "marocaine", "marocaines",
    "politique", "politiques", "parti", "partis", "gouvernement",
    "ministre", "ministres", "entreprise", "entreprises", "societe",
    "institution", "institutions", "personne", "personnes", "quels",
    "quelles", "quel", "quelle", "presente", "donne", "explique", "liste",
    "comment", "combien", "pourquoi", "est", "sont", "que", "qui", "quoi",
    "lesquels", "dans", "graphe", "base", "relations", "relation", "liens",
    "lien", "fonction", "fonctions", "responsables", "responsable",
}

NOMS_TROP_GENERIQUES = {
    "gouvernement", "parlement", "parti", "partis", "maroc", "ministre",
    "roi", "etat", "conseil", "assemblee", "chambre", "fonction",
}

PLURIELS_GENERIQUES = {
    "partis", "entreprises", "institutions", "ministres", "personnes",
    "responsables", "organisations", "societes", "fonctions", "acteurs",
}

MOTS_VIDES = {
    "de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "au",
    "aux", "un", "une", "qui", "est", "etait", "ete", "pour", "dans",
    "par", "sur", "a", "ont",
}

PRESENTATION_EXEMPLES = [
    "Les partis politiques et leurs membres",
    "Les entreprises et leurs dirigeants",
    "Quel est le parcours politique de Salaheddine Mezouar ?",
    "Les connexions directes et indirectes entre les personnalités",
    "Quels responsables politiques marocains ont des liens avec des "
    "entreprises ?",
    "Quels sont les partis politiques présents dans le graphe ?",
    "Que sait-on de Aziz Akhannouch ?",
    "Que sait-on de Q206209 ?",
    "Quel est le lien entre Salaheddine Mezouar et Mohamed Benchaâboun ?",
    "Qui dirige Maroc Telecom ?",
    "Quels sont les acteurs les plus connectés ?",
    "Classement des responsables politiques",
    "Quelles relations sont documentées par des sources ?",
    "Qui était ministre en 2018 ?",
    "Combien d'entreprises contient le graphe ?",
    "Combien de relations ACTIONNAIRE_DE existent dans le graphe ?",
]


# ============================================================
# 4. CLASSIFICATION DES ENTITES ET DES LIENS
# ============================================================

def categorie_type(type_entite) -> str:
    tn = type_normalise(type_entite)
    for cat, ensemble in CATEGORIES.items():
        if tn in ensemble:
            return cat
    if "PARTI" in tn:
        return "PARTI"
    if "PERSON" in tn:
        return "PERSONNE"
    if "ENTREPRISE" in tn or "SOCIETE" in tn:
        return "ENTREPRISE"
    if "INSTITUTION" in tn or "ADMINISTRATION" in tn:
        return "INSTITUTION"
    if "ORGANISATION" in tn or "PATRONAL" in tn:
        return "ORGANISATION"
    if "FONCTION" in tn:
        return "FONCTION"
    return "AUTRE"


def reclasser_entreprise(label) -> str:
    if contient_expression(label, EXCLUSIONS_ENTREPRISE_INSTITUTION):
        return "INSTITUTION"
    if contient_expression(label, EXCLUSIONS_ENTREPRISE_ORGANISATION):
        return "ORGANISATION"
    return "ENTREPRISE"


def categorie_entite(type_entite, labels, label) -> str:
    cat = categorie_type(type_entite)
    if cat in ("AUTRE", "ENTITE"):
        for lab in labels or []:
            c = categorie_type(lab)
            if c not in ("AUTRE", "ENTITE"):
                cat = c
                break
    if cat == "ENTREPRISE":
        cat = reclasser_entreprise(label)
    return cat


def perimetre_personne(entite) -> str:
    pays = normaliser(entite.get("pays"))
    if pays:
        if pays in ("maroc", "ma", "morocco", "marocain", "marocaine"):
            return "marocain"
        return "etranger"
    description = entite.get("description") or ""
    if contient_expression(description, ["maroc", "marocain", "marocaine"]):
        return "marocain"
    if contient_expression(description, NATIONALITES_ETRANGERES):
        return "etranger"
    return "inconnu"


def profil_politique_par_description(description) -> bool:
    return contient_expression(description, PROFIL_POLITIQUE_DESCRIPTION)


def institution_politique(label, description) -> bool:
    if contient_expression(label, INSTITUTIONS_POLITIQUES):
        return True
    return contient_expression(
        description, ["politique", "gouvernement", "parlement"]
    )


def classer_fonction(label, description) -> str:
    texte = normaliser(label)
    if re.search(r"\bou\b", texte):
        return "non_qualifie"
    if contient_expression(texte, FONCTIONS_POLITIQUES):
        return "politique"
    if contient_expression(texte, FONCTIONS_ECONOMIQUES):
        return "economique"
    if contient_expression(texte, ACRONYMES_PARTIS) and contient_expression(
        texte,
        ["president", "secretaire", "premier secretaire", "dirigeant",
         "chef", "commission", "membre"],
    ):
        return "politique"
    desc = normaliser(description)
    if contient_expression(
        desc,
        ["fonction parlementaire", "poste ministeriel", "fonction elective",
         "mandat parlementaire", "membre du gouvernement"],
    ):
        return "politique"
    if contient_expression(desc, FONCTIONS_ECONOMIQUES):
        return "economique"
    return "non_qualifie"


def indices_economiques(label, description) -> bool:
    texte = normaliser(str(label or "") + " " + str(description or ""))
    return contient_expression(texte, MOTS_ENTREPRISE)


def classer_objet(type_relation, objet) -> str:
    cat = objet["categorie"]
    label = objet["label"]
    description = objet.get("description") or ""
    t = type_relation
    if objet["qid_brut"]:
        return "non_qualifie"
    if t in ("ELU_A", "CANDIDAT_A", "COALITION_AVEC",
             "PARTICIPE_AU_GOUVERNEMENT"):
        return "economique" if cat == "ENTREPRISE" else "politique"
    if t in ("ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR",
             "FILIALE_DE"):
        if cat == "ENTREPRISE":
            return "economique"
        if cat == "ORGANISATION" or cat in ("ENTITE", "AUTRE"):
            return (
                "economique"
                if indices_economiques(label, description)
                or "patronal" in type_normalise(objet["type"]).lower()
                else "autre"
            )
        return "autre"
    if cat == "PARTI":
        return "politique"
    if cat == "INSTITUTION":
        return (
            "politique"
            if institution_politique(label, description)
            else "institutionnel"
        )
    if cat == "ENTREPRISE":
        return "economique"
    if cat == "FONCTION":
        return classer_fonction(label, description)
    if cat == "ORGANISATION":
        if "PATRONAL" in type_normalise(objet["type"]) or (
            indices_economiques(label, "")
            and not contient_expression(label, MOTS_NON_ECONOMIQUES)
        ):
            return "economique"
        return "autre"
    if cat in ("ENTITE", "AUTRE"):
        if contient_expression(label, INSTITUTIONS_POLITIQUES):
            return "politique"
        if indices_economiques(label, description):
            return "economique"
        return "non_qualifie"
    return "autre"


def nature_lien(relation, source, cible) -> str:
    t = relation["type"]
    if t in FAMILIAUX:
        return "familial"
    if t in ("SUCCEDE_A", "REMPLACE"):
        return "succession"
    cs, cc = source["categorie"], cible["categorie"]
    if cs == "PERSONNE" and cc != "PERSONNE":
        return classer_objet(t, cible)
    if cc == "PERSONNE" and cs != "PERSONNE":
        return classer_objet(t, source)
    if cs == "PERSONNE" and cc == "PERSONNE":
        return "autre"
    if t == "FILIALE_DE":
        return "economique"
    if cs == "PARTI" and cc == "PARTI":
        return "politique"
    if t in ("COALITION_AVEC", "PARTICIPE_AU_GOUVERNEMENT"):
        return "politique"
    return "autre"


# ============================================================
# 5. QUALITE DES PREUVES ET RELATIONS
# ============================================================

STATUTS_EXCLUS = {"REJETEE", "REJETE", "REJECTED", "DENIED", "REFUSE"}
STATUTS_CONFIRMES = {
    "CONFIRME", "CONFIRMEE", "VERIFIE", "VERIFIEE", "VALIDE", "VALIDEE",
}
STATUTS_ASSERTION = {
    "reported", "asserted", "alleged", "planned", "historical",
    "uncertain",
}
CERTITUDE_NUMERIQUE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def lire_preuves(proprietes):
    resultat = []
    for brut in convertir_liste(
        proprietes.get("evidences") or proprietes.get("preuves")
    ):
        objet = brut
        if isinstance(brut, str) and brut.strip()[:1] == "{":
            try:
                objet = json.loads(brut)
            except (json.JSONDecodeError, TypeError):
                objet = brut
        if isinstance(objet, dict):
            texte = (
                objet.get("text") or objet.get("texte")
                or objet.get("excerpt") or objet.get("evidence")
                or objet.get("extrait") or objet.get("description")
            )
            url = (
                objet.get("source_url") or objet.get("url")
                or objet.get("source")
            )
            resultat.append({"texte": texte, "url": url})
        elif objet:
            resultat.append({"texte": str(objet), "url": None})
    return resultat


def analyser_qualite(proprietes):
    preuves = lire_preuves(proprietes)
    extraits = [p["texte"] for p in preuves if p.get("texte")]
    urls = [
        p["url"] for p in preuves
        if p.get("url") and str(p["url"]).startswith("http")
    ]
    urls += [
        s for s in convertir_liste(proprietes.get("sources"))
        if isinstance(s, str) and s.startswith("http")
    ]
    urls = list(dict.fromkeys(urls))
    wikidata = [u for u in urls if "wikidata.org" in u.lower()]
    autres = [u for u in urls if u not in wikidata]
    alertes = []
    if not extraits:
        alertes.append("Aucun extrait textuel.")
    if not urls:
        alertes.append("Aucune URL source.")
    elif wikidata and not autres:
        alertes.append("Source Wikidata structurée seule.")
    if len(autres) > 1:
        alertes.append("Indépendance des sources non vérifiée.")
    if extraits and autres:
        niveau = "documente_a_examiner"
    elif extraits:
        niveau = "extrait_sans_url"
    elif wikidata and not autres:
        niveau = "source_structuree_a_examiner"
    elif urls:
        niveau = "url_sans_extrait"
    else:
        niveau = "insuffisant"
    return {
        "extraits": extraits,
        "urls": urls,
        "nombre_extraits": len(extraits),
        "nombre_urls": len(urls),
        "nombre_urls_wikidata": len(wikidata),
        "alertes": alertes,
        "niveau_documentation": niveau,
    }


def construire_relation(ligne):
    p = ligne.get("proprietes") or {}
    type_original = str(ligne.get("relation") or "")
    type_relation = RELATIONS_CANONIQUES.get(type_original, type_original)

    brut = str(p.get("status") or p.get("statut") or "").strip()
    brut_norm = type_normalise(brut).replace(" ", "")

    if brut_norm in STATUTS_EXCLUS:
        return None, "rejetee"

    if brut_norm in STATUTS_CONFIRMES:
        statut = "CONFIRME"
    elif brut:
        statut = "A_VERIFIER"
    else:
        statut = "NON_RENSEIGNE"

    confiance, certitude = None, None
    for cle in ("confidence", "confiance", "certainty"):
        valeur = p.get(cle)
        if valeur is None:
            continue
        try:
            confiance = float(valeur)
            break
        except (TypeError, ValueError):
            certitude = str(valeur).lower()
            confiance = CERTITUDE_NUMERIQUE.get(certitude)
            break

    if confiance is not None and confiance < MIN_CONFIDENCE:
        return None, "confiance_basse"

    qualite = analyser_qualite(p)
    debut = p.get("start_date") or p.get("debut")
    fin = p.get("end_date") or p.get("fin")

    return {
        "source_id": ligne.get("source_id"),
        "cible_id": ligne.get("cible_id"),
        "type": type_relation,
        "type_original": type_original,
        "statut": statut,
        "statut_renseigne": bool(brut),
        "nature_assertion": (
            brut.lower() if brut.lower() in STATUTS_ASSERTION else None
        ),
        "confiance": confiance,
        "certitude": certitude,
        "debut": str(debut) if debut else None,
        "fin": str(fin) if fin else None,
        "description": p.get("description"),
        "documents": convertir_liste(p.get("documents")),
        "methodes": convertir_liste(p.get("methodes")),
        "motif": p.get("correction_reason"),
        "qualite": qualite,
        "extraits": qualite["extraits"],
        "urls": qualite["urls"],
        "nature": None,
    }, None


ANNEE_RE = re.compile(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)")


def annees_relation(relation):
    def annee(valeur):
        trouve = ANNEE_RE.search(str(valeur or ""))
        return int(trouve.group(1)) if trouve else None
    return annee(relation.get("debut")), annee(relation.get("fin"))


def relation_dans_periode(relation, annee_min, annee_max):
    debut, fin = annees_relation(relation)
    if debut is None and fin is None:
        return None
    debut = debut if debut is not None else -10**6
    fin = fin if fin is not None else 10**6
    return debut <= annee_max and fin >= annee_min


# ============================================================
# 6. GRAPHE EN MEMOIRE
# ============================================================

def noeud_hors_perimetre(labels) -> bool:
    return bool(set(labels or []) & (LABELS_EXCLUS | LABELS_PREUVE))


def liste_noms(valeur):
    return [str(x) for x in convertir_liste(valeur) if x]


ACRONYME_IGNORES = {"de", "du", "des", "la", "le", "les", "l", "d", "et",
                    "en", "au", "aux", "pour"}


def normaliser_entite(ligne):
    props = ligne.get("props") or {}
    labels = list(ligne.get("labels") or [])

    if noeud_hors_perimetre(labels):
        return None

    def lire(cle):
        valeur = ligne.get(cle)
        return valeur if valeur is not None else props.get(cle)

    label = (
        lire("name") or lire("label") or lire("nom")
        or lire("title") or ""
    )

    identifiant = lire("id")
    if identifiant is None:
        identifiant = props.get("qid")
    if identifiant is None:
        identifiant = props.get("name")
    if identifiant is None and str(label).strip():
        identifiant = str(label)
    if identifiant is None:
        return None

    if not str(label).strip():
        if identifiant:
            label = str(identifiant)
        else:
            return None

    entite = {
        "id": identifiant,
        "label": str(label),
        "type": lire("type") or "",
        "labels": labels,
        "provenance_graphrag": LABEL_PROVENANCE in labels,
        "categorie": categorie_entite(lire("type"), labels, label),
        "description": lire("description") or "",
        "qid": lire("qid"),
        "pays": lire("pays") or lire("nationalite") or lire("country"),
        "qid_brut": est_qid_brut(label),
    }

    noms = set()
    sources = [lire("name"), lire("label")]
    sources += liste_noms(lire("aliases"))
    sources += liste_noms(lire("alias"))

    for n in sources:
        if not n:
            continue
        nn = normaliser(n)
        if nn:
            noms.add(nn)
        for sigle in re.findall(r"\(([A-Za-z]{2,10})\)", str(n)):
            noms.add(normaliser(sigle))

    if entite["categorie"] not in ("PERSONNE", "FONCTION"):
        mots = [
            m for m in normaliser(label).split()
            if m not in ACRONYME_IGNORES
        ]
        if len(mots) >= 3:
            sigle = "".join(m[0] for m in mots)
            if len(sigle) >= 3:
                noms.add(sigle)

    entite["noms"] = sorted(n for n in noms if n)
    return entite


class Graphe:

    def __init__(self):
        self.entites = {}
        self.relations = []
        self.adj = defaultdict(list)
        self.rejets = Counter()
        self.charge_le = None
        self.noeuds_neo4j = None
        self.relations_neo4j = None
        self.entites_neo4j_total = None
        self.techniques_neo4j = {}
        self.noeuds_exclus_neo4j = {}
        self.relations_vers_exclus_neo4j = None

    @classmethod
    def construire(cls, lignes_entites, lignes_relations):
        g = cls()
        for ligne in lignes_entites:
            if noeud_hors_perimetre(ligne.get("labels")):
                g.rejets["entite_hors_perimetre"] += 1
                continue
            e = normaliser_entite(ligne)
            if not e:
                g.rejets["entite_sans_id_ou_label"] += 1
                continue
            if e["id"] in g.entites:
                existant = g.entites[e["id"]]
                if e.get("description") and not existant.get("description"):
                    g.entites[e["id"]] = e
                    g.rejets["entite_doublon_id_remplace"] += 1
                else:
                    g.rejets["entite_doublon_id_ignore"] += 1
                continue
            g.entites[e["id"]] = e

        vus = set()
        for ligne in lignes_relations:
            type_brut = type_normalise(ligne.get("relation")).replace(
                " ", "_"
            )
            if type_brut in RELATIONS_PREUVE:
                g.rejets["relation_preuve"] += 1
                continue
            if type_brut in RELATIONS_EXCLUES:
                g.rejets["relation_exclue"] += 1
                continue

            sid, cid = ligne.get("source_id"), ligne.get("cible_id")
            if sid not in g.entites or cid not in g.entites:
                g.rejets["extremite_hors_perimetre"] += 1
                continue
            if sid == cid:
                g.rejets["boucle"] += 1
                continue

            relation, motif = construire_relation(ligne)
            if relation is None:
                g.rejets[motif] += 1
                continue

            cle = (
                sid, cid, relation["type"], relation["debut"],
                relation["fin"], tuple(relation["extraits"]),
                tuple(relation["urls"]),
            )
            if cle in vus:
                g.rejets["doublon_exact"] += 1
                continue
            vus.add(cle)

            relation["nature"] = nature_lien(
                relation, g.entites[sid], g.entites[cid]
            )
            relation["idx"] = len(g.relations)

            g.relations.append(relation)
            g.adj[sid].append(relation["idx"])
            g.adj[cid].append(relation["idx"])

        g.charge_le = time.strftime("%Y-%m-%dT%H:%M:%S")
        return g

    def degre(self, identifiant):
        return len(self.adj.get(identifiant, []))

    def autre_extremite(self, relation, identifiant):
        if relation["source_id"] == identifiant:
            return self.entites[relation["cible_id"]]
        return self.entites[relation["source_id"]]


def obtenir_graphe():
    with graphe_lock:
        graphe = GRAPHE
    if graphe is None:
        raise HTTPException(
            status_code=503,
            detail="Le graphe n'est pas encore chargé (POST /reload).",
        )
    return graphe


def verifier_perimetre(g):
    violations = []
    for e in g.entites.values():
        if noeud_hors_perimetre(e.get("labels")):
            violations.append(f"entité hors périmètre : {e['label']}")
    for r in g.relations:
        t = type_normalise(r["type_original"]).replace(" ", "_")
        if t in RELATIONS_TECHNIQUES:
            violations.append(f"relation non métier présente : {t}")
        if r["source_id"] not in g.entites or r["cible_id"] not in g.entites:
            violations.append("relation avec extrémité hors graphe métier")
    return {
        "conforme": not violations,
        "nombre_violations": len(violations),
        "violations": violations[:20],
        "labels_exclus": sorted(LABELS_EXCLUS | LABELS_PREUVE),
        "relations_exclues": sorted(RELATIONS_TECHNIQUES),
    }


def bloc_perimetre(g):
    preuve = {t: n for t, n in g.techniques_neo4j.items()
              if t in RELATIONS_PREUVE}
    exclues = {t: n for t, n in g.techniques_neo4j.items()
               if t in RELATIONS_EXCLUES}
    return {
        "entites_metier": {
            "chargees": len(g.entites),
            "par_categorie": dict(
                Counter(e["categorie"] for e in g.entites.values())
            ),
            "avec_label_provenance_graphrag": sum(
                1 for e in g.entites.values() if e.get("provenance_graphrag")
            ),
            "entites_neo4j_label_Entite": g.entites_neo4j_total,
            "ecartees_au_chargement": {
                k: v for k, v in g.rejets.items()
                if k.startswith("entite_")
            },
        },
        "relations_metier": {
            "chargees": len(g.relations),
            "par_nature": dict(Counter(r["nature"] for r in g.relations)),
            "par_statut": dict(Counter(r["statut"] for r in g.relations)),
        },
        "relations_techniques": {
            "preuve_tracabilite": preuve,
            "total_preuve": sum(preuve.values()),
            "exclues_des_analyses": exclues,
            "total_exclues": sum(exclues.values()),
            "ecartees_au_chargement": {
                k: v for k, v in g.rejets.items()
                if k in ("relation_preuve", "relation_exclue")
            },
        },
        "noeuds_exclus_neo4j": g.noeuds_exclus_neo4j,
        "relations_touchant_noeuds_exclus_neo4j":
            g.relations_vers_exclus_neo4j,
        "labels_exclus": sorted(LABELS_EXCLUS),
        "labels_preuve": sorted(LABELS_PREUVE),
        "relations_preuve": sorted(RELATIONS_PREUVE),
        "relations_exclues": sorted(RELATIONS_EXCLUES),
        "label_provenance_non_exclusif": LABEL_PROVENANCE,
    }


def confiance_moyenne(g):
    valeurs = [r["confiance"] for r in g.relations
               if r["confiance"] is not None]
    return round(sum(valeurs) / len(valeurs), 4) if valeurs else None


# ============================================================
# 7. NEO4J
# ============================================================

CYPHER_ENTITES = """
MATCH (e:Entite)
WHERE NOT any(l IN labels(e) WHERE l IN $labels_exclus)
  AND ($label = '' OR $label IN labels(e))
  AND coalesce(e.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(e.id, e.qid) IS NOT NULL
RETURN
    coalesce(e.id, e.qid) AS id,
    labels(e) AS labels,
    properties(e) AS props
"""

CYPHER_RELATIONS = """
MATCH (a:Entite)-[r]->(b:Entite)
WHERE NOT (type(r) IN $types_exclus)
  AND NOT any(l IN labels(a) WHERE l IN $labels_exclus)
  AND NOT any(l IN labels(b) WHERE l IN $labels_exclus)
  AND ($label = '' OR ($label IN labels(a) AND $label IN labels(b)))
  AND ($rel_source = '' OR r.source = $rel_source)
  AND coalesce(a.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(b.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(a.id, a.qid) IS NOT NULL
  AND coalesce(b.id, b.qid) IS NOT NULL
RETURN
    coalesce(a.id, a.qid) AS source_id,
    coalesce(b.id, b.qid) AS cible_id,
    type(r) AS relation,
    properties(r) AS proprietes
"""

CYPHER_COMPTES = """
CALL { MATCH (n) RETURN count(n) AS noeuds }
CALL { MATCH ()-[r]->() RETURN count(r) AS relations }
RETURN noeuds, relations
"""

CYPHER_ENTITES_TOTAL = """
MATCH (e:Entite) RETURN count(e) AS n
"""

CYPHER_TECHNIQUES = """
MATCH ()-[r]->()
WHERE type(r) IN $types_exclus
RETURN type(r) AS type, count(r) AS n
"""

CYPHER_NOEUDS_EXCLUS = """
MATCH (n)
UNWIND [l IN labels(n) WHERE l IN $labels_exclus] AS label
RETURN label, count(DISTINCT n) AS n
"""

CYPHER_REL_VERS_EXCLUS = """
MATCH (a)-[r]->(b)
WHERE any(l IN labels(a) + labels(b) WHERE l IN $labels_exclus)
RETURN count(r) AS n
"""

CYPHER_AUDIT = """
MATCH (e:Entite)
RETURN
    count(e) AS entites_total,
    sum(CASE WHEN any(l IN labels(e) WHERE l IN $labels_exclus)
             THEN 1 ELSE 0 END) AS entites_avec_label_exclu,
    sum(CASE WHEN $label <> '' AND $label IN labels(e)
             THEN 1 ELSE 0 END) AS entites_avec_label
"""

CYPHER_AUDIT_RELATIONS = """
MATCH (:Entite)-[r]->(:Entite)
RETURN
    count(r) AS relations_total,
    sum(CASE WHEN type(r) IN $types_exclus
             THEN 1 ELSE 0 END) AS relations_techniques_ou_exclues,
    sum(CASE WHEN $rel_source <> '' AND r.source = $rel_source
             THEN 1 ELSE 0 END) AS relations_avec_source
"""

CYPHER_PREUVES_ENTITE = """
MATCH (ev)-[r:EVIDENCE_FOR|EVIDENCE_TARGET]-(e:Entite)
WHERE coalesce(e.id, e.qid) = $id
  AND any(l IN labels(ev) WHERE l IN $labels_preuve)
RETURN type(r) AS role, properties(ev) AS preuve
LIMIT $limite
"""


def params_perimetre():
    return {
        "label": GRAPH_LABEL,
        "rel_source": RELATION_SOURCE,
        "labels_exclus": sorted(LABELS_EXCLUS | LABELS_PREUVE),
        "labels_preuve": sorted(LABELS_PREUVE),
        "types_exclus": sorted(RELATIONS_TECHNIQUES),
    }


CONTROLES_CYPHER = [
    {
        "nom": "noeuds_exclus_par_label",
        "objectif": "Nombre de nœuds portant un label exclu.",
        "attendu": "Customer, Product, GraphRAGEntity (indicatif).",
        "cypher": CYPHER_NOEUDS_EXCLUS.strip(),
    },
    {
        "nom": "entites_avec_label_exclu",
        "objectif": "Nœuds :Entite portant aussi un label exclu.",
        "attendu": "À comparer avec entite_hors_perimetre dans /stats.",
        "cypher": (
            "MATCH (e:Entite)\n"
            "WHERE any(l IN labels(e) WHERE l IN $labels_exclus)\n"
            "RETURN labels(e) AS labels, count(e) AS n ORDER BY n DESC"
        ),
    },
    {
        "nom": "entites_metier_chargees",
        "objectif": "Nombre d'entités métier lues par l'API.",
        "attendu": "Égal à perimetre.entites_metier.chargees.",
        "cypher": (
            "MATCH (e:Entite)\n"
            "WHERE NOT any(l IN labels(e) WHERE l IN $labels_exclus)\n"
            "  AND ($label = '' OR $label IN labels(e))\n"
            "  AND coalesce(e.statut_entite, '') <> 'FUSIONNEE'\n"
            "  AND coalesce(e.id, e.qid) IS NOT NULL\n"
            "RETURN count(e) AS entites_metier"
        ),
    },
    {
        "nom": "relations_metier_lues",
        "objectif": "Relations lues par l'API (avant rejets Python).",
        "attendu": "≥ relations_metier.chargees.",
        "cypher": (
            "MATCH (a:Entite)-[r]->(b:Entite)\n"
            "WHERE NOT (type(r) IN $types_exclus)\n"
            "  AND NOT any(l IN labels(a) WHERE l IN $labels_exclus)\n"
            "  AND NOT any(l IN labels(b) WHERE l IN $labels_exclus)\n"
            "  AND ($label = '' OR ($label IN labels(a) "
            "AND $label IN labels(b)))\n"
            "  AND ($rel_source = '' OR r.source = $rel_source)\n"
            "  AND coalesce(a.statut_entite, '') <> 'FUSIONNEE'\n"
            "  AND coalesce(b.statut_entite, '') <> 'FUSIONNEE'\n"
            "RETURN count(r) AS relations_metier"
        ),
    },
    {
        "nom": "relations_techniques_par_type",
        "objectif": "Relations non métier.",
        "attendu": "Jamais comptées dans les classements.",
        "cypher": CYPHER_TECHNIQUES.strip(),
    },
    {
        "nom": "relations_touchant_noeuds_exclus",
        "objectif": "Relations dont une extrémité est un nœud exclu.",
        "attendu": "Ignorées par l'API.",
        "cypher": CYPHER_REL_VERS_EXCLUS.strip(),
    },
    {
        "nom": "entites_sans_id",
        "objectif": "Nœuds :Entite sans id ni qid.",
        "attendu": "Idéalement 0.",
        "cypher": (
            "MATCH (e:Entite) WHERE e.id IS NULL AND e.qid IS NULL\n"
            "RETURN count(e) AS entites_sans_id"
        ),
    },
    {
        "nom": "doublons_id",
        "objectif": "Identifiants portés par plusieurs nœuds.",
        "attendu": "Idéalement aucun.",
        "cypher": (
            "MATCH (e:Entite)\n"
            "WITH coalesce(e.id, e.qid) AS id, count(e) AS n\n"
            "WHERE id IS NOT NULL AND n > 1\n"
            "RETURN id, n ORDER BY n DESC LIMIT 20"
        ),
    },
]

INDEX_RECOMMANDES = [
    "CREATE INDEX entite_id IF NOT EXISTS FOR (e:Entite) ON (e.id)",
    "CREATE INDEX entite_qid IF NOT EXISTS FOR (e:Entite) ON (e.qid)",
]


def get_driver():
    global driver
    if driver is None:
        if not NEO4J_URI or not NEO4J_PASSWORD:
            raise RuntimeError(
                "NEO4J_URI ou NEO4J_PASSWORD absent du fichier .env"
            )
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=NEO4J_TIMEOUT,
            max_transaction_retry_time=NEO4J_TIMEOUT,
        )
    return driver


def executer(cypher, params=None):
    try:
        with get_driver().session(database=NEO4J_DATABASE) as session:
            return [
                r.data() for r in session.run(
                    Neo4jQuery(cypher, timeout=NEO4J_TIMEOUT),
                    params or {},
                )
            ]
    except Neo4jError as exc:
        logger.exception("Erreur Neo4j")
        raise HTTPException(
            status_code=500,
            detail={"erreur": "Erreur Neo4j", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur d'exécution Neo4j")
        raise HTTPException(status_code=500, detail=str(exc))


def charger_graphe():
    global GRAPHE

    logger.info(
        "Chargement du graphe (GRAPH_LABEL=%r, RELATION_SOURCE=%r)...",
        GRAPH_LABEL or "(vide)", RELATION_SOURCE or "(vide)",
    )

    if GRAPH_LABEL or RELATION_SOURCE:
        logger.warning(
            "ATTENTION : GRAPH_LABEL et/ou RELATION_SOURCE sont "
            "renseignés dans le .env. Le périmètre sera restreint et "
            "des relations pourraient être silencieusement exclues. "
            "Vider ces variables pour charger tout le graphe métier."
        )

    parametres = params_perimetre()
    entites = executer(CYPHER_ENTITES, parametres)
    relations = executer(CYPHER_RELATIONS, parametres)

    nouveau = Graphe.construire(entites, relations)

    try:
        comptes = executer(CYPHER_COMPTES)
        if comptes:
            nouveau.noeuds_neo4j = comptes[0].get("noeuds")
            nouveau.relations_neo4j = comptes[0].get("relations")
    except Exception:
        logger.warning("Comptage global Neo4j indisponible.", exc_info=True)

    try:
        total = executer(CYPHER_ENTITES_TOTAL)
        nouveau.entites_neo4j_total = total[0]["n"] if total else None
        nouveau.techniques_neo4j = {
            r["type"]: r["n"] for r in executer(CYPHER_TECHNIQUES, parametres)
        }
        nouveau.noeuds_exclus_neo4j = {
            r["label"]: r["n"]
            for r in executer(CYPHER_NOEUDS_EXCLUS, parametres)
        }
        vers = executer(CYPHER_REL_VERS_EXCLUS, parametres)
        nouveau.relations_vers_exclus_neo4j = vers[0]["n"] if vers else None
    except Exception:
        logger.warning("Comptages du périmètre indisponibles.",
                       exc_info=True)

    controle = verifier_perimetre(nouveau)
    if not controle["conforme"]:
        logger.error("PÉRIMÈTRE NON CONFORME : %s", controle["violations"])

    with graphe_lock:
        GRAPHE = nouveau

    logger.info(
        "Graphe chargé : %s entités, %s relations (base : %s nœuds, "
        "%s relations ; rejets : %s).",
        len(nouveau.entites), len(nouveau.relations),
        nouveau.noeuds_neo4j, nouveau.relations_neo4j,
        dict(nouveau.rejets),
    )
    return nouveau


charger_index = charger_graphe


# ============================================================
# 8. DETECTION DES ENTITES DANS LA QUESTION
# ============================================================

QID_RE = re.compile(r"(?<![A-Za-z0-9])[Qq][0-9]{2,}(?![A-Za-z0-9])")


def detecter_entites(g, question, limite=6):
    texte = normaliser(question)
    mots_question = set(texte.split())
    qids = {q.upper() for q in QID_RE.findall(question)}

    capitalises = set()
    for jeton in re.findall(r"[^\W\d_][\w'’-]*", question, re.UNICODE):
        if jeton[:1].isupper():
            nj = normaliser(jeton)
            if len(nj) >= 4 and nj not in MOTS_QUESTION_GENERIQUES:
                capitalises.add(nj)

    candidats = []

    for e in g.entites.values():
        if qids and (
            str(e.get("qid") or "").upper() in qids
            or str(e["id"]).upper() in qids
        ):
            candidats.append({
                **e, "score": 200,
                "nom_trouve": normaliser(e.get("qid") or e["id"]),
            })
            continue

        if e["qid_brut"]:
            continue

        meilleur, nom_trouve = 0, None

        for nom in e["noms"]:
            if nom in NOMS_TROP_GENERIQUES or len(nom) < 3:
                continue

            if motif_expression(nom).search(texte):
                score = 100 + len(nom)
                nom_trouve = nom if score > meilleur else nom_trouve
                meilleur = max(meilleur, score)
                continue

            mots_nom = [m for m in nom.split() if len(m) > 2]
            if len(mots_nom) >= 2:
                communs = [m for m in mots_nom if m in mots_question]
                if len(communs) >= 2 and len(communs) / len(mots_nom) >= 0.6:
                    score = int(80 * len(communs) / len(mots_nom))
                    if score > meilleur:
                        meilleur, nom_trouve = score, nom

            for mot in nom.split():
                if mot in capitalises and len(mot) >= 5:
                    if 55 > meilleur:
                        meilleur, nom_trouve = 55, nom

        if meilleur:
            candidats.append({
                **e, "score": meilleur, "nom_trouve": nom_trouve,
            })

    if not candidats:
        return []

    if any(c["score"] >= 100 for c in candidats):
        candidats = [c for c in candidats if c["score"] >= 100]
        longs = [c["nom_trouve"] for c in candidats]
        candidats = [
            c for c in candidats
            if not any(
                c["nom_trouve"] != autre
                and motif_expression(c["nom_trouve"]).search(autre)
                for autre in longs
            )
        ]
    else:
        candidats = [c for c in candidats if c["score"] >= 55]

    candidats.sort(
        key=lambda c: (c["score"], g.degre(c["id"])), reverse=True
    )
    return candidats[:limite]


# ============================================================
# 9. ANALYSE DE LA QUESTION
# ============================================================

def extraire_periode(question):
    annees = [int(x) for x in ANNEE_RE.findall(question)]
    if not annees:
        return None
    annees = annees[:2]
    return (min(annees), max(annees))


def domaine_fonction(texte, cle):
    trouve = re.search(
        r"(?<![a-z0-9])" + re.escape(normaliser(cle))
        + r"s? (?:de la|de l|du|des|de|d) ([a-z0-9-]{4,})",
        texte,
    )
    if trouve and trouve.group(1) not in MOTS_VIDES:
        return trouve.group(1)
    return None


def analyser_question(question):
    t = normaliser(question)

    categories = [
        cat for cat, mots in MOTS_CATEGORIES.items()
        if contient_expression(t, mots)
    ]

    roles = []
    for mots, types in MOTS_ROLES:
        if contient_expression(t, mots):
            roles.extend(types)

    types_nommes = detecter_types_relation_nommes(question)
    types_relation = list(dict.fromkeys(roles + types_nommes))

    fonctions = [
        cle for cle, mots in MOTS_FONCTIONS_QUESTION.items()
        if contient_expression(t, mots)
    ]

    domaines = {}
    for cle in fonctions:
        domaine = domaine_fonction(t, cle)
        if domaine:
            domaines[cle] = domaine

    return {
        "periode": extraire_periode(question),
        "categories": categories,
        "types_relation": types_relation,
        "types_relation_explicites": types_nommes,
        "fonctions": fonctions,
        "domaines_fonction": domaines,
        "politique": contient_expression(t, MOTS_POLITIQUE),
        "economique": contient_expression(t, MOTS_ECONOMIQUE),
        "chemin": contient_expression(t, MOTS_CHEMIN),
        "classement": contient_expression(t, MOTS_CLASSEMENT),
        "connectivite": contient_expression(
            t, ["plus connecte", "les plus connectes", "connectivite",
                "centralite", "degre"],
        ),
        "stats": contient_expression(t, MOTS_STATS),
        "comptage": contient_expression(t, ["combien", "nombre de",
                                            "nombre d"]),
        "documente": contient_expression(t, MOTS_DOCUMENTE),
        "reseaux": contient_expression(t, MOTS_RESEAUX),
        "parcours": contient_expression(t, MOTS_PARCOURS),
        "connexions": contient_expression(t, MOTS_CONNEXIONS),
        "relations_mot": contient_expression(t, MOTS_RELATIONS),
        "personne_explicite": contient_expression(
            t,
            ["personne", "personnalite", "responsable", "acteur",
             "individu", "politicien", "homme politique",
             "femme politique"],
        ),
        "etrangers_inclus": contient_expression(
            t, ["international", "etranger", "monde"]
        ),
    }


def decider_intention(a, entites):
    cats = set(a["categories"])
    autres = cats - {"PERSONNE"}
    types = set(a["types_relation"])
    types_nommes = set(a.get("types_relation_explicites") or [])

    if types_nommes and not entites:
        if a["comptage"] or (a["stats"] and not a["classement"]):
            return "comptage_relation"
        return "relations_roles"

    if a["comptage"] and not entites and types:
        return "comptage_relation"

    if entites:
        if a["parcours"]:
            return "parcours"
        if len(entites) >= 2 and a["chemin"]:
            return "chemins"
        return "entite"

    if a["parcours"]:
        return "parcours"

    if a["classement"] and a["stats"]:
        return "stats"

    if a["classement"]:
        if (
            a["politique"] or ("PERSONNE" in cats and not a["connectivite"])
        ) and not a["economique"] and not autres:
            return "classement_politique"
        return "classement"

    if a["connexions"]:
        return "connexions"

    if (
        cats & {"PARTI", "ENTREPRISE"}
        and not a["personne_explicite"]
        and not a["fonctions"]
        and types <= (TYPES_PARTI | TYPES_ENTREPRISE)
        and not (a["politique"] and a["economique"])
        and not (a["chemin"] and len(autres) >= 2)
    ):
        return "liste_categorie"

    if a["politique"] and a["economique"]:
        if "PERSONNE" in cats or a["fonctions"]:
            return "personnes_politiques_economiques"
        if len(autres) >= 2:
            return "relations_categories"
        if not cats:
            return "relations_nature"
        return "personnes_politiques_economiques"

    if "PERSONNE" in cats and (
        a["economique"] or "ENTREPRISE" in cats
    ):
        return "personnes_economiques"

    if a["economique"] and types & {
        "ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR",
    } and not (cats - {"ENTREPRISE", "PERSONNE"}) \
            and (a["personne_explicite"] or "PERSONNE" in cats):
        return "personnes_economiques"

    if ("PERSONNE" in cats and a["politique"]) or (
        a["fonctions"] and not a["comptage"]
    ):
        return "personnes_politiques"

    if "PERSONNE" in cats and "OCCUPE_FONCTION" in types:
        return "personnes_fonctions"

    if a["chemin"] and autres and "PERSONNE" not in cats:
        return "relations_categories"

    if a["documente"]:
        return "documentees"

    if a["reseaux"]:
        return "apercu_reseaux"

    if a["stats"] and not autres:
        return "stats"

    if "PARTI" in cats and types <= TYPES_PARTI:
        return "liste_categorie"

    if "ENTREPRISE" in cats and types <= TYPES_ENTREPRISE:
        return "liste_categorie"

    if a["relations_mot"] and (a["politique"] or a["economique"]) \
            and not cats:
        return "relations_nature"

    if types and cats != {"FONCTION"}:
        return "relations_roles"

    if cats:
        return "liste_categorie"

    if a["periode"]:
        return "relations_roles"

    if types_nommes:
        return "relations_roles"

    return "hors_perimetre"


# ============================================================
# 10. VUES ET TABLEAUX
# ============================================================

def periode_texte(relation):
    debut, fin = relation["debut"], relation["fin"]
    if not debut and not fin:
        return "non renseignée"
    return f"{debut or '?'} → {fin or '?'}"


def confiance_texte(relation):
    if relation["confiance"] is not None:
        texte = f"{relation['confiance']:.2f}"
        if relation["certitude"]:
            texte += f" ({relation['certitude']})"
        return texte
    return "non renseignée"


def statut_texte(relation):
    texte = relation["statut"]
    if relation["nature_assertion"]:
        texte += f" ({relation['nature_assertion']})"
    return texte


def preuve_texte(relation):
    if relation["extraits"]:
        return tronquer(relation["extraits"][0], 110)
    if relation["qualite"]["nombre_urls_wikidata"] and not [
        u for u in relation["urls"] if "wikidata.org" not in u.lower()
    ]:
        return "aucun extrait (Wikidata structuré seul)"
    return "aucun extrait"


def sources_texte(relation):
    urls = relation["urls"]
    if not urls:
        return "aucune URL"
    liens = []
    for url in urls[:2]:
        domaine = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        liens.append(f"[{domaine}]({url})" if "(" not in url
                     and ")" not in url else domaine)
    reste = len(urls) - 2
    return " ".join(liens) + (f" +{reste}" if reste > 0 else "")


def relation_api(g, relation):
    s = g.entites[relation["source_id"]]
    c = g.entites[relation["cible_id"]]
    return {
        "source_id": s["id"], "source": s["label"],
        "type_source": s["type"],
        "cible_id": c["id"], "cible": c["label"],
        "type_cible": c["type"],
        "relation": relation["type"],
        "relation_originale": relation["type_original"],
        "nature": relation["nature"],
        "statut": relation["statut"],
        "statut_renseigne": relation["statut_renseigne"],
        "confiance": relation["confiance"],
        "debut": relation["debut"], "fin": relation["fin"],
        "description": relation["description"],
        "preuves": relation["extraits"],
        "sources": relation["urls"],
        "documents": relation["documents"],
        "methodes": relation["methodes"],
        "motif_correction": relation["motif"],
        "qualite_preuve": relation["qualite"],
    }


def entite_api(e):
    return {
        "id": e["id"], "qid": e.get("qid"), "label": e["label"],
        "type": e["type"], "categorie": e["categorie"],
        "description": e["description"], "pays": e.get("pays"),
        "provenance_graphrag": e.get("provenance_graphrag", False),
    }


def ligne_relation(g, relation):
    s = g.entites[relation["source_id"]]
    c = g.entites[relation["cible_id"]]
    return {
        "Source": f"{s['label']} ({s['type']})",
        "Relation": relation["type"],
        "Cible": f"{c['label']} ({c['type']})",
        "Nature": NATURE_LIBELLE.get(relation["nature"], relation["nature"]),
        "Période": periode_texte(relation),
        "Statut": statut_texte(relation),
        "Confiance": confiance_texte(relation),
        "Preuve": preuve_texte(relation),
        "Sources": sources_texte(relation),
    }


COLONNES_RELATION = [
    "Source", "Relation", "Cible", "Nature", "Période", "Statut",
    "Confiance", "Preuve", "Sources",
]


def ligne_profil(g, relation, ref_id):
    autre = g.autre_extremite(relation, ref_id)
    sortant = relation["source_id"] == ref_id
    return {
        "Nature": NATURE_LIBELLE.get(relation["nature"], relation["nature"]),
        "Sens": "→" if sortant else "←",
        "Relation": relation["type"],
        "Entité liée": f"{autre['label']} ({autre['type']})",
        "Période": periode_texte(relation),
        "Statut": statut_texte(relation),
        "Confiance": confiance_texte(relation),
        "Preuve": preuve_texte(relation),
        "Sources": sources_texte(relation),
    }


COLONNES_PROFIL = [
    "Nature", "Sens", "Relation", "Entité liée", "Période", "Statut",
    "Confiance", "Preuve", "Sources",
]


def tableau(titre, colonnes, lignes, total=None):
    return {
        "titre": titre,
        "colonnes": colonnes,
        "total": len(lignes) if total is None else total,
        "lignes": lignes[:LIMITE_API_MAX],
    }


def nouveau_resultat(intention, titre, tableaux, total, notes=None,
                     relations=None, extra=None):
    resultat = {
        "intention": intention,
        "titre": titre,
        "total": total,
        "tableaux": tableaux,
        "notes": notes or [],
        "relations": relations or [],
    }
    if extra:
        resultat.update(extra)
    return resultat


def filtrer_periode(relations, periode, notes):
    if not periode:
        return relations
    gardees, sans_date = [], 0
    for r in relations:
        ok = relation_dans_periode(r, *periode)
        if ok is True:
            gardees.append(r)
        elif ok is None:
            sans_date += 1
    libelle = (
        str(periode[0]) if periode[0] == periode[1]
        else f"{periode[0]}–{periode[1]}"
    )
    notes.append(
        f"Filtre temporel {libelle} : {len(gardees)} relation(s) datée(s) "
        f"retenue(s). {sans_date} relation(s) sans date ont été écartées "
        "car leur période est inconnue."
    )
    return gardees


def trier_relations(relations):
    return sorted(
        relations,
        key=lambda r: (
            NATURE_ORDRE.get(r["nature"], 9), r["type"],
            -(r["confiance"] or 0),
        ),
    )


def voisins_par_type(g, identifiant, types=None, categories=None):
    sortie = []
    for idx in g.adj.get(identifiant, []):
        r = g.relations[idx]
        if types and r["type"] not in types:
            continue
        autre = g.autre_extremite(r, identifiant)
        if categories and autre["categorie"] not in categories:
            continue
        sortie.append((r, autre))
    return sortie


def noms_courts(paires, n=6, avec_periode=True):
    vus, noms = set(), []
    for r, autre in paires:
        if autre["id"] in vus:
            continue
        vus.add(autre["id"])
        texte = autre["label"]
        if avec_periode and (r["debut"] or r["fin"]):
            texte += f" ({r['debut'] or '?'}→{r['fin'] or '?'})"
        noms.append(texte)
    if not noms:
        return "—"
    reste = len(noms) - n
    return ", ".join(noms[:n]) + (f" +{reste}" if reste > 0 else "")


def liste_courte(elements, n=3, separateur=" ; "):
    uniques = list(dict.fromkeys(e for e in elements if e))
    if not uniques:
        return "—"
    reste = len(uniques) - n
    return separateur.join(uniques[:n]) + (f" +{reste}" if reste > 0 else "")


def nb_uniques(paires):
    return len({autre["id"] for _, autre in paires})


# ============================================================
# 11. TRAITEMENTS PAR INTENTION
# ============================================================

def h_entite(g, a, entites):
    cibles = entites[:MAX_ENTITES_PROFIL]
    tableaux, notes, relations_api = [], [], []
    total = 0

    ambigus = [e for e in entites if e["score"] < 100]

    if len(ambigus) > 1:
        notes.append(
            "Plusieurs entités correspondent au nom cité : "
            + "; ".join(
                f"{e['label']} ({e['type']}"
                + (f", {tronquer(e['description'], 60)}"
                   if e["description"] else "")
                + ")"
                for e in ambigus[:5]
            )
            + ". Précisez le nom complet pour cibler une seule entité."
        )

    if len(entites) > len(cibles):
        notes.append(
            f"{len(entites)} entités détectées : seules les "
            f"{len(cibles)} premières sont détaillées."
        )

    for e in cibles:
        relations = [g.relations[i] for i in g.adj.get(e["id"], [])]
        total_brut = len(relations)

        if a["types_relation"]:
            relations = [
                r for r in relations if r["type"] in a["types_relation"]
            ]

        if a["documente"]:
            relations = [r for r in relations if r["extraits"]]

        if a["economique"] and not a["politique"]:
            relations = [r for r in relations if r["nature"] == "economique"]
        elif a["politique"] and not a["economique"]:
            relations = [r for r in relations if r["nature"] == "politique"]

        relations = filtrer_periode(relations, a["periode"], notes)
        relations = trier_relations(relations)

        if not relations:
            disponibles = Counter(
                g.relations[i]["type"] for i in g.adj.get(e["id"], [])
            )
            texte = (
                f"Aucune relation correspondante pour {e['label']}."
            )
            if disponibles:
                texte += " Types disponibles : " + ", ".join(
                    f"{t} ({n})" for t, n in disponibles.most_common()
                ) + "."
            else:
                texte += " Aucune relation n'est enregistrée pour cette entité."
            notes.append(texte)
            continue

        total += len(relations)

        nb = Counter(r["nature"] for r in relations)
        resume = ", ".join(
            f"{n} {NATURE_LIBELLE.get(k, k)}" for k, n in nb.most_common()
        )

        description = (
            f" — {tronquer(e['description'], 120)}"
            if e["description"] else ""
        )

        tableaux.append(tableau(
            f"{e['label']} ({e['type']}){description} — {resume}",
            COLONNES_PROFIL,
            [ligne_profil(g, r, e["id"]) for r in relations],
        ))
        relations_api.extend(relation_api(g, r) for r in relations)

        if "DIRIGE" in a["types_relation"]:
            notes_dirige = (
                "« Diriger » recherche les relations DIRIGE et PRESIDE ; "
                "chaque relation garde son propre type, ses dates et ses "
                "preuves."
            )
            if notes_dirige not in notes:
                notes.append(notes_dirige)

        if total_brut > len(relations) and (
            a["types_relation"] or a["documente"]
            or (a["economique"] != a["politique"])
        ):
            notes.append(
                f"{e['label']} : {len(relations)} relation(s) sur "
                f"{total_brut} correspondent au filtre demandé."
            )

    return nouveau_resultat(
        "entite",
        "Relations des entités citées dans la question",
        tableaux, total, notes, relations_api,
    )


def noeud_intermediaire_ok(g, identifiant, cible_id):
    if identifiant == cible_id:
        return True
    e = g.entites[identifiant]
    if e["qid_brut"] or e["categorie"] in ("LIEU", "EVENEMENT"):
        return False
    return g.degre(identifiant) <= HUB_DEGRE_MAX


def plus_courts_chemins(g, a_id, b_id, periode=None, exclure_direct=False,
                        profondeur=None):
    prof_max = profondeur or CHEMIN_MAX
    dist = {a_id: 0}
    preds = defaultdict(list)
    file = deque([a_id])

    while file:
        u = file.popleft()
        if u == b_id or dist[u] >= prof_max:
            continue
        voisins = defaultdict(list)
        for idx in g.adj.get(u, []):
            relation = g.relations[idx]
            if periode and relation_dans_periode(relation, *periode) \
                    is not True:
                continue
            v = (
                relation["cible_id"] if relation["source_id"] == u
                else relation["source_id"]
            )
            if exclure_direct and {u, v} == {a_id, b_id}:
                continue
            if not noeud_intermediaire_ok(g, v, b_id):
                continue
            voisins[v].append(idx)
        for v, indices in voisins.items():
            if v not in dist:
                dist[v] = dist[u] + 1
                preds[v].append((u, indices))
                file.append(v)
            elif dist[v] == dist[u] + 1:
                preds[v].append((u, indices))

    if b_id not in dist:
        return []

    chemins = []

    def remonter(noeud, acc):
        if len(chemins) >= CHEMINS_PAR_PAIRE:
            return
        if noeud == a_id:
            chemins.append(list(reversed(acc)))
            return
        for precedent, indices in preds[noeud]:
            remonter(precedent, acc + [(precedent, noeud, indices)])

    remonter(b_id, [])
    return chemins


def h_chemins(g, a, entites):
    cibles = [e for e in entites if e["score"] >= 100][:4] or entites[:4]
    tableaux, notes, relations_api = [], [], []
    lignes, details = [], []
    paires = 0
    sans_chemin = []

    for i in range(len(cibles)):
        for j in range(i + 1, len(cibles)):
            e1, e2 = cibles[i], cibles[j]
            paires += 1
            chemins = plus_courts_chemins(
                g, e1["id"], e2["id"], a["periode"]
            )
            if chemins and len(chemins[0]) == 1:
                chemins = chemins + plus_courts_chemins(
                    g, e1["id"], e2["id"], a["periode"],
                    exclure_direct=True,
                )
            if not chemins:
                sans_chemin.append(f"{e1['label']} ↔ {e2['label']}")
                continue
            for chemin in chemins:
                morceaux = [g.entites[chemin[0][0]]["label"]]
                statuts, natures = set(), set()
                for (u, v, indices) in chemin:
                    textes = []
                    for idx in indices:
                        r = g.relations[idx]
                        fleche = "→" if r["source_id"] == u else "←"
                        textes.append(f"{r['type']} {fleche}")
                        statuts.add(statut_texte(r))
                        natures.add(
                            NATURE_LIBELLE.get(r["nature"], r["nature"])
                        )
                        details.append({
                            "Paire": f"{e1['label']} ↔ {e2['label']}",
                            **ligne_relation(g, r),
                        })
                        relations_api.append(relation_api(g, r))
                    morceaux.append(f"—[{', '.join(textes)}]—")
                    morceaux.append(g.entites[v]["label"])
                lignes.append({
                    "Paire": f"{e1['label']} ↔ {e2['label']}",
                    "Longueur": len(chemin),
                    "Type": "direct" if len(chemin) == 1 else "indirect",
                    "Chemin": " ".join(morceaux),
                    "Natures": ", ".join(sorted(natures)),
                    "Statuts": ", ".join(sorted(statuts)),
                })

    if lignes:
        tableaux.append(tableau(
            "Chemins les plus courts (directs et indirects)",
            ["Paire", "Longueur", "Type", "Chemin", "Natures", "Statuts"],
            lignes,
        ))
        vus, uniques = set(), []
        for d in details:
            cle = tuple(d.items())
            if cle not in vus:
                vus.add(cle)
                uniques.append(d)
        tableaux.append(tableau(
            "Détail des relations de chaque chemin",
            ["Paire"] + COLONNES_RELATION, uniques,
        ))

    if sans_chemin:
        notes.append(
            "Aucun chemin de longueur ≤ " + str(CHEMIN_MAX)
            + " dans les données pour : " + "; ".join(sans_chemin)
            + ". Information insuffisante pour établir un lien."
        )

    if a["periode"]:
        notes.append(
            "Chemins restreints aux relations datées dans la période "
            "demandée."
        )

    notes.append(
        "Relation directe = relation enregistrée entre les deux entités. "
        "Un chemin indirect (via une ou plusieurs entités intermédiaires) "
        "n'est ni une relation directe ni la preuve d'un contrôle, d'une "
        "propriété ou d'une influence."
    )

    if len(entites) < 2 or paires == 0:
        notes.append("Deux entités distinctes sont nécessaires.")

    return nouveau_resultat(
        "chemins", "Liens entre les entités citées", tableaux,
        len(lignes), notes, relations_api,
    )


def liens_qualifies(g, identifiant):
    pol, eco = [], []
    for idx in g.adj.get(identifiant, []):
        relation = g.relations[idx]
        if relation["nature"] not in ("politique", "economique"):
            continue
        if g.autre_extremite(relation, identifiant)["categorie"] \
                == "PERSONNE":
            continue
        (pol if relation["nature"] == "politique" else eco).append(relation)
    return pol, eco


def appliquer_periode(relations, periode, compteur):
    if not periode:
        return relations
    gardees = []
    for r in relations:
        ok = relation_dans_periode(r, *periode)
        if ok is True:
            gardees.append(r)
        elif ok is None:
            compteur[0] += 1
    return gardees


def separer_types(types, exiger_pol, exiger_eco):
    if not types:
        return None, None
    if exiger_pol and exiger_eco:
        eco = [t for t in types if t in (
            "ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR",
            "FILIALE_DE", "DIRIGE", "PRESIDE", "FONDATEUR_DE",
        )]
        pol = [t for t in types if t in (
            "OCCUPE_FONCTION", "MEMBRE_DE", "COALITION_AVEC",
            "PARTICIPE_AU_GOUVERNEMENT",
        )]
        return (pol or None), (eco or None)
    if exiger_pol:
        return types, None
    return None, types


def fonction_correspond(g, relation, a):
    if relation["type"] != "OCCUPE_FONCTION":
        return False
    cible = g.entites[relation["cible_id"]]
    source = g.entites[relation["source_id"]]
    objet = cible if source["categorie"] == "PERSONNE" else source
    label = normaliser(objet["label"])
    for cle in a["fonctions"]:
        if not contient_expression(label, MOTS_FONCTIONS_QUESTION[cle]):
            continue
        domaine = a["domaines_fonction"].get(cle)
        if domaine and domaine not in label.split():
            continue
        return True
    return False


def resume_fiche(g, f):
    pid = f["personne"]["id"]
    fonctions, partis, entreprises = [], [], []
    for r in f["pol"] + f["eco"]:
        autre = g.autre_extremite(r, pid)
        if r["type"] == "OCCUPE_FONCTION":
            texte = autre["label"]
            if r["debut"] or r["fin"]:
                texte += f" ({r['debut'] or '?'}→{r['fin'] or '?'})"
            fonctions.append(texte)
        elif autre["categorie"] == "PARTI":
            partis.append(f"{autre['label']} ({r['type']})")
    for r in f["eco"]:
        autre = g.autre_extremite(r, pid)
        entreprises.append(
            autre["label"] if r["type"] == "OCCUPE_FONCTION"
            else f"{autre['label']} ({r['type']})"
        )
    return fonctions, partis, entreprises


def tableau_entites_liees(g, fiches, titre, cles):
    groupes = {}
    for f in fiches:
        for cle in cles:
            for r in f[cle]:
                autre = g.autre_extremite(r, f["personne"]["id"])
                groupe = groupes.setdefault(
                    autre["id"], {"e": autre, "liens": [], "natures": set()}
                )
                groupe["liens"].append((f["personne"]["label"], r["type"]))
                groupe["natures"].add(
                    NATURE_LIBELLE.get(r["nature"], r["nature"])
                )
    lignes = []
    for groupe in sorted(
        groupes.values(),
        key=lambda x: (-len({n for n, _ in x["liens"]}), x["e"]["label"]),
    ):
        personnes = {n for n, _ in groupe["liens"]}
        detail = list(dict.fromkeys(
            f"{n} ({t})" for n, t in groupe["liens"]
        ))
        lignes.append({
            "Entité": groupe["e"]["label"],
            "Catégorie": CATEGORIE_LIBELLE.get(
                groupe["e"]["categorie"], groupe["e"]["categorie"]
            ),
            "Nature": ", ".join(sorted(groupe["natures"])),
            "Personnes": len(personnes),
            "Personnes liées (relation)": liste_courte(detail, 6, ", "),
        })
    return tableau(
        titre,
        ["Entité", "Catégorie", "Nature", "Personnes",
         "Personnes liées (relation)"],
        lignes,
    )


def h_personnes(g, a, exiger_pol, exiger_eco, limite):
    types_pol, types_eco = separer_types(
        a["types_relation"], exiger_pol, exiger_eco
    )
    fiches, exclues, notes = [], [], []
    exclure_etrangers = EXCLURE_ETRANGERS and not a["etrangers_inclus"]
    sans_date_total = [0]

    for e in g.entites.values():
        if e["categorie"] != "PERSONNE":
            continue
        pol, eco = liens_qualifies(g, e["id"])

        if a["fonctions"]:
            pol = [r for r in pol if fonction_correspond(g, r, a)]
        elif types_pol:
            pol = [r for r in pol if r["type"] in types_pol]

        if types_eco:
            eco = [r for r in eco if r["type"] in types_eco]

        pol = appliquer_periode(pol, a["periode"], sans_date_total)
        eco = appliquer_periode(eco, a["periode"], sans_date_total)

        retenu_par = "relation_politique"
        if exiger_pol and not pol:
            if (
                not a["fonctions"] and not types_pol and not a["periode"]
                and profil_politique_par_description(e["description"])
            ):
                retenu_par = "description"
            else:
                continue

        if exiger_eco and not eco:
            continue

        if not exiger_pol:
            retenu_par = "relation_economique"

        perimetre = perimetre_personne(e)
        if exclure_etrangers and perimetre == "etranger":
            exclues.append(e["label"])
            continue

        fiches.append({
            "personne": e, "pol": trier_relations(pol),
            "eco": trier_relations(eco), "retenu_par": retenu_par,
            "perimetre": perimetre,
        })

    fiches.sort(
        key=lambda f: (len(f["pol"]) + len(f["eco"]), f["personne"]["label"]),
        reverse=True,
    )

    if exclues:
        notes.append(
            f"{len(exclues)} personne(s) exclue(s) car leur description "
            "indique une nationalité étrangère : "
            + ", ".join(exclues[:15])
            + ("…" if len(exclues) > 15 else "")
            + ". Passez `EXCLURE_ETRANGERS=0` pour les inclure."
        )

    inconnus = sum(1 for f in fiches if f["perimetre"] == "inconnu")
    if inconnus:
        notes.append(
            f"{inconnus} personne(s) dont la nationalité n'est pas établie "
            "par les données : elles ne sont pas présentées comme "
            "marocaines avec certitude."
        )

    if a["periode"]:
        notes.append(
            "Filtre temporel appliqué ; "
            f"{sans_date_total[0]} lien(s) sans date, correspondant par "
            "ailleurs à la question, ont été écarté(s)."
        )

    if any(f["retenu_par"] == "description" for f in fiches):
        notes.append(
            "« description » : personne retenue parce que sa description "
            "indique un profil politique, sans lien politique enregistré."
        )

    notes.append(
        "Personnes triées par nombre de liens qualifiés. Avoir un lien "
        "politique et un lien économique ne prouve ni influence ni conflit "
        "d'intérêts."
    )

    lignes_resume = []
    for f in fiches:
        fonctions, partis, entreprises = resume_fiche(g, f)
        lignes_resume.append({
            "Personne": f["personne"]["label"],
            "Description": tronquer(f["personne"]["description"] or "—", 80),
            "Périmètre": {
                "marocain": "marocain (d'après les données)",
                "etranger": "étranger",
                "inconnu": "nationalité non établie",
            }[f["perimetre"]],
            "Fonctions": liste_courte(fonctions, 3),
            "Partis": liste_courte(partis, 2),
            "Entreprises / organisations": liste_courte(entreprises, 3),
            "Liens politiques": len(f["pol"]),
            "Liens économiques": len(f["eco"]),
            "Retenue par": f["retenu_par"].replace("_", " "),
        })

    resume = tableau(
        "Personnes retenues",
        ["Personne", "Description", "Périmètre", "Fonctions", "Partis",
         "Entreprises / organisations", "Liens politiques",
         "Liens économiques", "Retenue par"],
        lignes_resume,
    )

    relations_api = []

    def detail(cle, titre):
        lignes = []
        for f in fiches[:limite]:
            for r in f[cle]:
                lignes.append({
                    "Personne": f["personne"]["label"],
                    **ligne_profil(g, r, f["personne"]["id"]),
                })
                relations_api.append(relation_api(g, r))
        return tableau(titre, ["Personne"] + COLONNES_PROFIL, lignes)

    tableaux = []
    solo_eco = exiger_eco and not exiger_pol
    cles = ("eco",) if solo_eco else (
        ("pol", "eco") if exiger_pol and exiger_eco else ("pol",)
    )

    groupes = tableau_entites_liees(
        g, fiches,
        "Entreprises et organisations liées, avec leurs dirigeants, "
        "actionnaires et autres rôles" if solo_eco
        else "Partis, institutions, fonctions et entreprises liés",
        cles,
    )

    if solo_eco:
        tableaux += [groupes, resume]
    else:
        tableaux += [resume, groupes]

    if exiger_pol:
        tableaux.append(detail("pol", "Relations politiques (détail)"))
    if exiger_eco or not exiger_pol:
        tableaux.append(detail("eco", "Relations économiques (détail)"))

    notes.insert(
        0,
        f"{len(fiches)} personne(s) reliée(s) à {groupes['total']} entité(s)."
    )

    intention = (
        "personnes_politiques_economiques" if exiger_pol and exiger_eco
        else "personnes_politiques" if exiger_pol
        else "personnes_economiques"
    )

    return nouveau_resultat(
        intention,
        "Personnes avec des liens "
        + ("politiques et économiques" if exiger_pol and exiger_eco
           else "politiques" if exiger_pol else "économiques"),
        tableaux, len(fiches), notes, relations_api,
        extra={"exclues": exclues},
    )


def h_personnes_fonctions(g, a, limite):
    fiches, exclues, notes = [], [], []
    sans_date = [0]
    exclure_etrangers = EXCLURE_ETRANGERS and not a["etrangers_inclus"]

    for e in g.entites.values():
        if e["categorie"] != "PERSONNE":
            continue
        liens = [
            g.relations[i] for i in g.adj.get(e["id"], [])
            if g.relations[i]["type"] == "OCCUPE_FONCTION"
        ]
        liens = appliquer_periode(liens, a["periode"], sans_date)
        if not liens:
            continue
        perimetre = perimetre_personne(e)
        if exclure_etrangers and perimetre == "etranger":
            exclues.append(e["label"])
            continue
        fiches.append((e, trier_relations(liens), perimetre))

    fiches.sort(key=lambda f: (len(f[1]), f[0]["label"]), reverse=True)

    lignes = [{
        "Personne": e["label"],
        "Description": tronquer(e["description"] or "—", 80),
        "Périmètre": "étranger" if p == "etranger" else (
            "marocain (d'après les données)" if p == "marocain"
            else "nationalité non établie"
        ),
        "Fonctions": len(liens),
        "Liste des fonctions": liste_courte([
            g.autre_extremite(r, e["id"])["label"]
            + (f" ({r['debut'] or '?'}→{r['fin'] or '?'})"
               if r["debut"] or r["fin"] else "")
            for r in liens
        ], 4),
    } for e, liens, p in fiches]

    detail, relations_api = [], []
    for e, liens, _ in fiches[:limite]:
        for r in liens:
            detail.append({"Personne": e["label"],
                           **ligne_profil(g, r, e["id"])})
            relations_api.append(relation_api(g, r))

    if exclues:
        notes.append(
            f"{len(exclues)} personne(s) exclue(s) (nationalité étrangère "
            "d'après la description)."
        )
    if a["periode"]:
        notes.append(
            f"Filtre temporel appliqué ; {sans_date[0]} fonction(s) sans "
            "date écartée(s)."
        )
    notes.append(
        "Les fonctions sont celles enregistrées par OCCUPE_FONCTION, "
        "quelle que soit leur nature."
    )

    return nouveau_resultat(
        "personnes_fonctions", "Personnalités et leurs fonctions",
        [
            tableau("Personnalités et fonctions",
                    ["Personne", "Description", "Périmètre", "Fonctions",
                     "Liste des fonctions"], lignes),
            tableau("Détail des fonctions",
                    ["Personne"] + COLONNES_PROFIL, detail),
        ],
        len(fiches), notes, relations_api, extra={"exclues": exclues},
    )


def repartition_relations(g, identifiant, n=3):
    compte = Counter(g.relations[i]["type"]
                     for i in g.adj.get(identifiant, []))
    return ", ".join(f"{t}×{k}" for t, k in compte.most_common(n)) or "—"


def ligne_parti(g, e):
    i = e["id"]
    dirigeants = voisins_par_type(g, i, ["PRESIDE", "DIRIGE"], {"PERSONNE"})
    fondateurs = voisins_par_type(g, i, ["FONDATEUR_DE"], {"PERSONNE"})
    membres = voisins_par_type(g, i, ["MEMBRE_DE"], {"PERSONNE"})
    coalitions = voisins_par_type(g, i, ["COALITION_AVEC"], {"PARTI"})
    return {
        "Parti": e["label"],
        "Description": tronquer(e["description"] or "—", 80),
        "Présidents / dirigeants": noms_courts(dirigeants, 4),
        "Fondateurs": noms_courts(fondateurs, 3),
        "Membres": nb_uniques(membres),
        "Exemples de membres": noms_courts(membres, 6, False),
        "Coalitions": noms_courts(coalitions, 4, False),
        "Relations": g.degre(i),
    }


COLONNES_PARTI = [
    "Parti", "Description", "Présidents / dirigeants", "Fondateurs",
    "Membres", "Exemples de membres", "Coalitions", "Relations",
]


def ligne_entreprise(g, e):
    i = e["id"]
    filiales = voisins_par_type(g, i, ["FILIALE_DE"], None)
    return {
        "Entreprise": e["label"],
        "Description": tronquer(e["description"] or "—", 80),
        "Dirigeants / présidents": noms_courts(
            voisins_par_type(g, i, ["DIRIGE", "PRESIDE"], {"PERSONNE"}), 4
        ),
        "Actionnaires": noms_courts(
            voisins_par_type(g, i, ["ACTIONNAIRE_DE"], {"PERSONNE"}), 4,
            False,
        ),
        "Fondateurs": noms_courts(
            voisins_par_type(g, i, ["FONDATEUR_DE"], {"PERSONNE"}), 3, False
        ),
        "Administrateurs": noms_courts(
            voisins_par_type(g, i, ["ADMINISTRATEUR_DE"], {"PERSONNE"}), 3,
            False,
        ),
        "Filiales / maison mère": noms_courts(filiales, 4, False),
        "Relations": g.degre(i),
    }


COLONNES_ENTREPRISE = [
    "Entreprise", "Description", "Dirigeants / présidents", "Actionnaires",
    "Fondateurs", "Administrateurs", "Filiales / maison mère", "Relations",
]


def ligne_institution(g, e):
    i = e["id"]
    personnes = voisins_par_type(g, i, None, {"PERSONNE"})
    roles = Counter(r["type"] for r, _ in personnes)
    autres = voisins_par_type(g, i, None, {"INSTITUTION", "ORGANISATION",
                                           "PARTI"})
    return {
        "Nom": e["label"],
        "Type": e["type"],
        "Description": tronquer(e["description"] or "—", 80),
        "Personnes liées": nb_uniques(personnes),
        "Rôles": ", ".join(f"{t}×{n}" for t, n in roles.most_common(3))
        or "—",
        "Exemples de personnes": noms_courts(personnes, 5, False),
        "Autres institutions / organisations liées": noms_courts(
            autres, 4, False
        ),
        "Relations": g.degre(i),
    }


COLONNES_INSTITUTION = [
    "Nom", "Type", "Description", "Personnes liées", "Rôles",
    "Exemples de personnes", "Autres institutions / organisations liées",
    "Relations",
]


def ligne_fonction(g, e):
    titulaires = voisins_par_type(g, e["id"], ["OCCUPE_FONCTION"],
                                  {"PERSONNE"})
    return {
        "Fonction": e["label"],
        "Description": tronquer(e["description"] or "—", 80),
        "Titulaires": nb_uniques(titulaires),
        "Titulaires (période)": noms_courts(titulaires, 5),
        "Relations": g.degre(e["id"]),
    }


COLONNES_FONCTION = [
    "Fonction", "Description", "Titulaires", "Titulaires (période)",
    "Relations",
]


def ligne_personne_liste(g, e):
    fonctions, partis, eco = [], [], []
    for r, autre in voisins_par_type(g, e["id"]):
        if autre["categorie"] == "PERSONNE":
            continue
        if r["type"] == "OCCUPE_FONCTION":
            fonctions.append(autre["label"])
        elif autre["categorie"] == "PARTI":
            partis.append(autre["label"])
        if r["nature"] == "economique":
            eco.append(autre["label"])
    return {
        "Nom": e["label"],
        "Description": tronquer(e["description"] or "—", 80),
        "Fonctions": liste_courte(fonctions, 3),
        "Partis": liste_courte(partis, 2),
        "Entreprises / organisations": liste_courte(eco, 3),
        "Relations": g.degre(e["id"]),
    }


COLONNES_PERSONNE = [
    "Nom", "Description", "Fonctions", "Partis",
    "Entreprises / organisations", "Relations",
]


def ligne_generique(g, e):
    return {
        "Nom": e["label"], "Type": e["type"],
        "Description": tronquer(e["description"] or "—", 100),
        "Relations": g.degre(e["id"]),
        "Principaux types": repartition_relations(g, e["id"]),
    }


COLONNES_GENERIQUE = ["Nom", "Type", "Description", "Relations",
                      "Principaux types"]

VUES_CATEGORIES = {
    "PARTI": (COLONNES_PARTI, ligne_parti, "Partis politiques et leurs "
              "membres", TYPES_PARTI),
    "ENTREPRISE": (COLONNES_ENTREPRISE, ligne_entreprise,
                   "Entreprises et leurs dirigeants", TYPES_ENTREPRISE),
    "INSTITUTION": (COLONNES_INSTITUTION, ligne_institution,
                    "Institutions", None),
    "ORGANISATION": (COLONNES_INSTITUTION, ligne_institution,
                     "Organisations", None),
    "FONCTION": (COLONNES_FONCTION, ligne_fonction, "Fonctions", None),
    "PERSONNE": (COLONNES_PERSONNE, ligne_personne_liste, "Personnes",
                 None),
}


def tri_vue(ligne):
    return (
        ligne.get("Membres") or ligne.get("Titulaires")
        or ligne.get("Personnes liées") or 0,
        ligne.get("Relations") or 0,
    )


def h_liste_categorie(g, a, limite):
    cats = list(a["categories"]) or ["PERSONNE"]
    if len(cats) > 1 and "PERSONNE" in cats:
        cats = [c for c in cats if c != "PERSONNE"]

    tableaux, notes, relations_api = [], [], []
    total_global = 0

    for cat in cats:
        entites = [
            e for e in g.entites.values()
            if e["categorie"] == cat and not e["qid_brut"]
        ]
        colonnes, fabrique, titre, types_detail = VUES_CATEGORIES.get(
            cat, (COLONNES_GENERIQUE, ligne_generique,
                  CATEGORIE_LIBELLE.get(cat, cat), None)
        )
        lignes = [fabrique(g, e) for e in entites]
        lignes.sort(key=lambda l: (tri_vue(l), str(next(iter(l.values())))),
                    reverse=True)
        total_global += len(lignes)
        tableaux.append(tableau(
            f"{titre} ({len(lignes)}, triés par nombre de liens)",
            colonnes, lignes,
        ))

        if types_detail:
            detail = []
            for e in sorted(entites, key=lambda e: -g.degre(e["id"])):
                for r, _ in voisins_par_type(
                    g, e["id"], types_detail, {"PERSONNE"} | (
                        {"PARTI", "ENTREPRISE", "ORGANISATION"}
                        if cat == "ENTREPRISE" else {"PARTI"}
                    )
                ):
                    detail.append({"Entité": e["label"],
                                   **ligne_profil(g, r, e["id"])})
                    relations_api.append(relation_api(g, r))
            if detail:
                tableaux.append(tableau(
                    f"{titre} : détail des relations (statuts, dates, "
                    "preuves)",
                    ["Entité"] + COLONNES_PROFIL, detail,
                ))

        non_resolues = sum(
            1 for e in g.entites.values()
            if e["categorie"] == cat and e["qid_brut"]
        )
        if non_resolues:
            notes.append(
                f"{non_resolues} entité(s) {cat} sans libellé exploitable "
                "(identifiant Wikidata brut) non listée(s)."
            )

    if "PERSONNE" in cats:
        notes.append(
            "Cette liste de personnes n'est pas filtrée par nationalité."
        )
    if "ENTREPRISE" in cats:
        notes.append(
            "Les entités portant le label Entreprise mais dont le libellé "
            "désigne une académie, une université, un musée ou un "
            "ministère sont reclassées (organisation ou institution)."
        )
    notes.append(
        "Les listes sont exhaustives sur le graphe chargé ; le détail "
        "affiché est limité par le paramètre `limite`."
    )

    return nouveau_resultat(
        "liste_categorie",
        "Liste par catégorie d'entités", tableaux, total_global, notes,
        relations_api[:LIMITE_API_MAX],
    )


def h_relations_categories(g, a, limite):
    cats = set(a["categories"])
    relations = []
    for r in g.relations:
        cs = g.entites[r["source_id"]]["categorie"]
        cc = g.entites[r["cible_id"]]["categorie"]
        if len(cats) == 1:
            if cs in cats and cc in cats:
                relations.append(r)
        elif cs in cats and cc in cats and cs != cc:
            relations.append(r)

    notes = []
    relations = filtrer_periode(relations, a["periode"], notes)
    relations = trier_relations(relations)

    tableaux = [tableau(
        "Relations directes entre ces catégories",
        COLONNES_RELATION, [ligne_relation(g, r) for r in relations],
    )]

    via = Counter()
    exemples = {}
    for e in g.entites.values():
        if e["categorie"] != "PERSONNE":
            continue
        voisins = {}
        for i in g.adj.get(e["id"], []):
            autre = g.autre_extremite(g.relations[i], e["id"])
            if autre["categorie"] in cats:
                voisins[autre["id"]] = autre
        ids = sorted(voisins, key=str)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if len(cats) > 1 and (
                    voisins[ids[x]]["categorie"]
                    == voisins[ids[y]]["categorie"]
                ):
                    continue
                cle = (ids[x], ids[y])
                via[cle] += 1
                exemples.setdefault(cle, []).append(e["label"])

    lignes_via = [{
        "Entité A": g.entites[x]["label"],
        "Entité B": g.entites[y]["label"],
        "Personnes en commun": n,
        "Exemples": ", ".join(exemples[(x, y)][:3]),
    } for (x, y), n in via.most_common()]

    if lignes_via:
        tableaux.append(tableau(
            "Liens indirects : entités reliées à une même personne",
            ["Entité A", "Entité B", "Personnes en commun", "Exemples"],
            lignes_via,
        ))

    notes.append(
        "Deux entités reliées à une même personne ne sont pas "
        "directement liées entre elles : ce n'est pas un lien de contrôle."
    )
    if not relations:
        notes.append(
            "Aucune relation directe enregistrée entre ces catégories."
        )

    return nouveau_resultat(
        "relations_categories", "Relations entre catégories d'entités",
        tableaux, len(relations), notes,
        [relation_api(g, r) for r in relations[:LIMITE_API_MAX]],
    )


def h_classement(g, a, limite):
    cats = set(a["categories"])
    entites = [
        e for e in g.entites.values()
        if not e["qid_brut"] and (
            e["categorie"] in cats if cats
            else e["categorie"] not in ("AUTRE", "ENTITE")
        )
    ]
    entites.sort(key=lambda e: (g.degre(e["id"]), e["label"]), reverse=True)
    lignes = [{
        "Rang": i,
        "Nom": e["label"],
        "Type": e["type"],
        "Relations": g.degre(e["id"]),
        "Principaux types": repartition_relations(g, e["id"]),
    } for i, e in enumerate(entites, start=1) if g.degre(e["id"]) > 0]

    return nouveau_resultat(
        "classement", "Acteurs les plus connectés",
        [tableau("Classement par nombre de relations", [
            "Rang", "Nom", "Type", "Relations", "Principaux types",
        ], lignes)], len(lignes),
        ["Le nombre de relations mesure la connectivité dans le graphe, "
         "pas l'importance ni l'influence."],
    )


def classement_politique(g, periode=None, inclure_etrangers=False):
    fiches = []
    sans_date = [0]
    for e in g.entites.values():
        if e["categorie"] != "PERSONNE" or e["qid_brut"]:
            continue
        perimetre = perimetre_personne(e)
        if EXCLURE_ETRANGERS and not inclure_etrangers \
                and perimetre == "etranger":
            continue
        pol, _ = liens_qualifies(g, e["id"])
        pol = [r for r in pol if r["type"] in POIDS_POLITIQUES]
        pol = appliquer_periode(pol, periode, sans_date)
        if not pol:
            continue
        compteur = Counter(r["type"] for r in pol)
        score = sum(POIDS_POLITIQUES[t] * n for t, n in compteur.items())
        fiches.append({
            "personne": e, "score": score, "relations": pol,
            "par_type": dict(compteur), "perimetre": perimetre,
        })
    fiches.sort(
        key=lambda f: (-f["score"], -len(f["relations"]),
                       f["personne"]["label"])
    )
    return fiches, sans_date[0]


def h_classement_politique(g, a, limite):
    fiches, sans_date = classement_politique(
        g, a["periode"], a["etrangers_inclus"]
    )
    lignes = [{
        "Rang": i,
        "Personne": f["personne"]["label"],
        "Description": tronquer(f["personne"]["description"] or "—", 80),
        "Périmètre": {
            "marocain": "marocain (d'après les données)",
            "etranger": "étranger",
            "inconnu": "nationalité non établie",
        }[f["perimetre"]],
        "Score politique": f["score"],
        "Relations politiques": len(f["relations"]),
        "Détail par type": ", ".join(
            f"{t}×{n}" for t, n in sorted(
                f["par_type"].items(), key=lambda kv: -kv[1]
            )
        ),
    } for i, f in enumerate(fiches, start=1)]

    relations_api = []
    for f in fiches[:limite]:
        relations_api.extend(relation_api(g, r) for r in f["relations"])

    notes = [
        "Score = somme pondérée des relations politiques directes : "
        + ", ".join(f"{t} ×{p}" for t, p in sorted(
            POIDS_POLITIQUES.items(), key=lambda kv: -kv[1]
        ))
        + ".",
        "Les relations familiales, scolaires, géographiques, judiciaires "
        "et techniques ne comptent pas.",
        "Ce score mesure la présence de relations politiques dans le "
        "graphe, pas l'influence politique réelle.",
    ]
    if a["periode"]:
        notes.append(
            f"Filtre temporel appliqué ; {sans_date} lien(s) sans date "
            "écarté(s)."
        )

    return nouveau_resultat(
        "classement_politique", "Classement politique pondéré",
        [tableau("Responsables politiques classés par score",
                 ["Rang", "Personne", "Description", "Périmètre",
                  "Score politique", "Relations politiques",
                  "Détail par type"], lignes)],
        len(lignes), notes, relations_api,
    )


def relations_personnes_entreprises(g, inclure_etrangers=False):
    trouvees = []
    for r in g.relations:
        if r["type"] not in RELATIONS_ENTREPRISE_PERSONNE:
            continue
        s, c = g.entites[r["source_id"]], g.entites[r["cible_id"]]
        if s["categorie"] == "PERSONNE" and c["categorie"] == "ENTREPRISE":
            personne, entreprise = s, c
        elif c["categorie"] == "PERSONNE" and s["categorie"] == "ENTREPRISE":
            personne, entreprise = c, s
        else:
            continue
        if entreprise["qid_brut"] or personne["qid_brut"]:
            continue
        if EXCLURE_ETRANGERS and not inclure_etrangers \
                and perimetre_personne(personne) == "etranger":
            continue
        trouvees.append((personne, entreprise, r))
    trouvees.sort(
        key=lambda t: (-(t[2]["confiance"] or 0), t[0]["label"])
    )
    return trouvees


def tableau_perimetre(g):
    b = bloc_perimetre(g)

    def v(x):
        return x if x is not None else "non calculé"

    lignes = [
        {"Indicateur": "Entités métier chargées",
         "Valeur": b["entites_metier"]["chargees"]},
        {"Indicateur": "Nœuds :Entite dans Neo4j",
         "Valeur": v(b["entites_metier"]["entites_neo4j_label_Entite"])},
        {"Indicateur": "Relations métier chargées",
         "Valeur": b["relations_metier"]["chargees"]},
        {"Indicateur": "Relations de preuve (EVIDENCE_*, traçabilité)",
         "Valeur": b["relations_techniques"]["total_preuve"]},
        {"Indicateur": "Relations exclues (KNOWS, GRAPHRAG_RELATION, "
                       "PURCHASED)",
         "Valeur": b["relations_techniques"]["total_exclues"]},
        {"Indicateur": "Nœuds exclus (label → nombre)",
         "Valeur": ", ".join(
             f"{k}: {n}" for k, n in b["noeuds_exclus_neo4j"].items()
         ) or "aucun"},
        {"Indicateur": "Relations touchant un nœud exclu",
         "Valeur": v(b["relations_touchant_noeuds_exclus_neo4j"])},
    ]

    return tableau("Périmètre métier : métier / techniques / exclus",
                   ["Indicateur", "Valeur"], lignes)


def h_stats(g, a, limite):
    cats = Counter(e["categorie"] for e in g.entites.values())
    types = Counter(r["type"] for r in g.relations)
    natures = Counter(r["nature"] for r in g.relations)
    statuts = Counter(r["statut"] for r in g.relations)

    avec_extrait = sum(1 for r in g.relations if r["extraits"])
    avec_date = sum(1 for r in g.relations if r["debut"] or r["fin"])
    avec_statut = sum(1 for r in g.relations if r["statut_renseigne"])
    wikidata_seul = sum(
        1 for r in g.relations
        if r["qualite"]["niveau_documentation"]
        == "source_structuree_a_examiner"
    )
    non_resolues = sum(1 for e in g.entites.values() if e["qid_brut"])

    n = len(g.relations) or 1
    moyenne = confiance_moyenne(g)

    def valeur_neo4j(v):
        return v if v is not None else "non calculé"

    tableaux = [
        tableau("Couverture : base Neo4j et mémoire", ["Indicateur",
                                                       "Valeur"], [
            {"Indicateur": "Nœuds dans Neo4j (toute la base)",
             "Valeur": valeur_neo4j(g.noeuds_neo4j)},
            {"Indicateur": "Relations dans Neo4j (toute la base)",
             "Valeur": valeur_neo4j(g.relations_neo4j)},
            {"Indicateur": "Entités chargées en mémoire",
             "Valeur": len(g.entites)},
            {"Indicateur": "Relations chargées en mémoire",
             "Valeur": len(g.relations)},
            {"Indicateur": "Relations ignorées au chargement",
             "Valeur": ", ".join(f"{k}: {v}" for k, v in
                                 g.rejets.items()) or "aucune"},
        ]),
        tableau("Entités par catégorie", ["Catégorie", "Nombre"], [
            {"Catégorie": CATEGORIE_LIBELLE.get(c, c), "Nombre": k}
            for c, k in cats.most_common()
        ]),
        tableau("Relations par type", ["Type", "Nombre"], [
            {"Type": t, "Nombre": k} for t, k in types.most_common()
        ]),
        tableau("Relations par nature", ["Nature", "Nombre"], [
            {"Nature": NATURE_LIBELLE.get(c, c), "Nombre": k}
            for c, k in natures.most_common()
        ]),
        tableau("Relations par statut", ["Statut", "Nombre"], [
            {"Statut": s, "Nombre": k} for s, k in statuts.most_common()
        ]),
        tableau("Qualité de la documentation", ["Indicateur", "Valeur"], [
            {"Indicateur": "Relations avec extrait textuel",
             "Valeur": f"{avec_extrait} ({100 * avec_extrait // n} %)"},
            {"Indicateur": "Relations avec source Wikidata seule",
             "Valeur": f"{wikidata_seul} ({100 * wikidata_seul // n} %)"},
            {"Indicateur": "Relations avec au moins une date",
             "Valeur": f"{avec_date} ({100 * avec_date // n} %)"},
            {"Indicateur": "Relations avec statut renseigné",
             "Valeur": f"{avec_statut} ({100 * avec_statut // n} %)"},
            {"Indicateur": "Relations CONFIRME",
             "Valeur": statuts.get("CONFIRME", 0)},
            {"Indicateur": "Relations NON_RENSEIGNE",
             "Valeur": statuts.get("NON_RENSEIGNE", 0)},
            {"Indicateur": "Confiance moyenne (relations renseignées)",
             "Valeur": moyenne if moyenne is not None else "non renseignée"},
            {"Indicateur": "Entités sans libellé exploitable",
             "Valeur": non_resolues},
        ]),
    ]

    tableaux.insert(0, tableau_perimetre(g))

    if a.get("classement"):
        top = h_classement(g, {"categories": []}, 15)["tableaux"][0]
        top["lignes"] = top["lignes"][:15]
        top["titre"] = "Entités les plus connectées (15 premières)"
        tableaux.append(top)

    return nouveau_resultat(
        "stats", "Statistiques du graphe", tableaux,
        len(g.relations),
        [f"{len(g.entites)} entités et {len(g.relations)} relations "
         "actives chargées.",
         "NON_RENSEIGNE (aucun statut enregistré) est distinct de "
         "CONFIRME et de A_VERIFIER."],
    )


def h_documentees(g, a, limite):
    notes = []
    relations = [r for r in g.relations if r["extraits"]]
    sur = len(g.relations)
    relations = filtrer_periode(relations, a["periode"], notes)
    relations.sort(
        key=lambda r: (len(r["extraits"]), r["confiance"] or 0),
        reverse=True,
    )

    par_type = defaultdict(lambda: [0, 0, 0, 0])
    for r in g.relations:
        ligne = par_type[r["type"]]
        ligne[0] += 1
        if r["extraits"]:
            ligne[1] += 1
        if [u for u in r["urls"] if "wikidata.org" not in u.lower()]:
            ligne[2] += 1
        if r["qualite"]["niveau_documentation"] \
                == "source_structuree_a_examiner":
            ligne[3] += 1

        couverture = tableau(
        "Couverture documentaire par type de relation",
        ["Type", "Relations", "Avec extrait", "Avec URL hors Wikidata",
         "Wikidata structuré seul"],
        [{"Type": t, "Relations": v[0], "Avec extrait": v[1],
          "Avec URL hors Wikidata": v[2], "Wikidata structuré seul": v[3]}
         for t, v in sorted(par_type.items(), key=lambda kv: -kv[1][0])],
    )

    notes.append(
        f"{len(relations)} relation(s) sur {sur} disposent d'au moins un "
        "extrait textuel. Un extrait n'est pas une validation : une "
        "relation reste A_VERIFIER ou NON_RENSEIGNE tant qu'aucune "
        "vérification n'est enregistrée."
    )

    return nouveau_resultat(
        "documentees", "Relations documentées par un extrait textuel",
        [couverture,
         tableau("Relations avec extrait (les plus documentées d'abord)",
                 COLONNES_RELATION,
                 [ligne_relation(g, r) for r in relations])],
        len(relations), notes,
        [relation_api(g, r) for r in relations[:LIMITE_API_MAX]],
    )


def h_relations_roles(g, a, limite):
    cats = set(a["categories"]) - {"PERSONNE"}
    types = a["types_relation"]
    notes = []
    relations = []

    for r in g.relations:
        if types and r["type"] not in types:
            continue
        s, c = g.entites[r["source_id"]], g.entites[r["cible_id"]]
        if cats and not (s["categorie"] in cats or c["categorie"] in cats):
            continue
        if a["fonctions"] and not fonction_correspond(g, r, a):
            continue
        relations.append(r)

    relations = filtrer_periode(relations, a["periode"], notes)
    relations = trier_relations(relations)

    return nouveau_resultat(
        "relations_roles",
        "Relations de type " + (", ".join(types) if types else "toutes"),
        [tableau("Relations correspondantes", COLONNES_RELATION,
                 [ligne_relation(g, r) for r in relations])],
        len(relations), notes,
        [relation_api(g, r) for r in relations[:LIMITE_API_MAX]],
    )


def h_comptage_relation(g, a, limite):
    """Compte les relations d'un type nommé, sur le périmètre chargé.

    N'active jamais le mode chemin. Retourne le NOMBRE DE RELATIONS (pas
    le nombre de personnes ni d'entités). Affiche la répartition par type
    et le détail limité.
    """
    types_demandes = set(a["types_relation"]) or set(
        TYPES_RELATION_NOMMES.keys()
    )

    compte = Counter()
    entites_impliquees = set()
    relations_correspondantes = []

    for r in g.relations:
        if r["type"] in types_demandes:
            compte[r["type"]] += 1
            entites_impliquees.add(r["source_id"])
            entites_impliquees.add(r["cible_id"])
            relations_correspondantes.append(r)

    total = sum(compte.values())

    lignes = [
        {"Type de relation": t, "Nombre": n}
        for t, n in compte.most_common()
    ] or [
        {"Type de relation": t, "Nombre": 0}
        for t in sorted(types_demandes)
    ]

    detail = [
        ligne_relation(g, r)
        for r in relations_correspondantes[:limite]
    ]

    notes = [
        f"Périmètre : graphe métier chargé "
        f"({len(g.entites)} entités, {len(g.relations)} relations "
        "en mémoire, exclusions appliquées).",
        f"Types recherchés : {', '.join(sorted(types_demandes))}.",
        "Le statut (CONFIRME / A_VERIFIER / NON_RENSEIGNE) est indiqué "
        "relation par relation : aucun filtre implicite par statut n'est "
        "appliqué au comptage.",
        f"{len(entites_impliquees)} entité(s) distincte(s) impliquée(s).",
        "Le total retourné est le nombre de RELATIONS, distinct du nombre "
        "d'entités.",
    ]

    if not total:
        notes.insert(
            0,
            "Aucune relation de ce type n'a été retournée par le graphe "
            "interrogé. Cela ne prouve pas que la relation n'existe pas "
            "dans la réalité : elle peut être absente du périmètre chargé "
            "ou non documentée.",
        )

    return nouveau_resultat(
        "comptage_relation",
        "Comptage des relations " + ", ".join(sorted(types_demandes)),
        [
            tableau(
                "Nombre de relations par type",
                ["Type de relation", "Nombre"],
                lignes, total=total,
            ),
            tableau(
                "Détail des relations (limité)",
                COLONNES_RELATION,
                detail, total=total,
            ),
        ],
        total,
        notes,
        [relation_api(g, r) for r in relations_correspondantes
         [:LIMITE_API_MAX]],
        extra={
            "nombre_entites_distinctes": len(entites_impliquees),
            "types_demandes": sorted(types_demandes),
            "perimetre": "sous_graphe_metier_charge",
        },
    )


def h_apercu(g, a, limite, reseaux=True):
    sous = {**a, "types_relation": [], "fonctions": [], "periode": None}

    partis = [
        ligne_parti(g, e) for e in g.entites.values()
        if e["categorie"] == "PARTI" and not e["qid_brut"]
    ]
    partis.sort(key=lambda l: (l["Membres"], l["Relations"]), reverse=True)

    alliances = [
        r for r in g.relations
        if r["type"] in ("COALITION_AVEC", "PARTICIPE_AU_GOUVERNEMENT")
    ]

    personnes = h_personnes(g, sous, True, False, limite)
    top_personnes = personnes["tableaux"][0]
    top_personnes["lignes"] = top_personnes["lignes"][:10]
    top_personnes["titre"] = (
        "Personnalités politiques les plus reliées (10 premières)"
    )

    tableaux = [
        tableau("Partis et leurs membres (réseaux partisans)",
                COLONNES_PARTI, partis),
        tableau("Coalitions et participations au gouvernement",
                COLONNES_RELATION,
                [ligne_relation(g, r) for r in trier_relations(alliances)]),
        top_personnes,
    ]

    return nouveau_resultat(
        "apercu_reseaux", "Réseaux politiques du graphe", tableaux,
        len(partis),
        ["Les réseaux sont lus ici à travers les partis (membres, "
         "dirigeants), les coalitions, la participation au gouvernement et "
         "les personnalités les plus reliées.",
         "Une connexion n'est pas une preuve d'influence."],
    )


def h_hors_perimetre(g):
    return nouveau_resultat(
        "hors_perimetre", "Question non rattachée au graphe", [], 0,
        ["Information insuffisante : la question ne correspond à aucune "
         "recherche possible dans le graphe (aucune entité reconnue, "
         "aucune catégorie, aucun type de relation).",
         "Exemples de questions prises en charge : "
         + " | ".join(PRESENTATION_EXEMPLES)],
    )


def cle_chronologique(relation):
    debut, fin = annees_relation(relation)
    reference = debut if debut is not None else fin
    return (reference is None, reference or 0, relation["type"])


COLONNES_PARCOURS = [
    "Début", "Fin", "Relation", "Entité", "Nature", "Statut",
    "Confiance", "Preuve", "Sources",
]


def lignes_parcours(g, e, a, notes):
    relations = []
    for i in g.adj.get(e["id"], []):
        r = g.relations[i]
        if r["type"] in FAMILIAUX:
            continue
        if e["categorie"] == "PERSONNE" and r["type"] not in TYPES_PARCOURS:
            continue
        relations.append(r)

    relations = filtrer_periode(relations, a["periode"], notes)
    relations.sort(key=cle_chronologique)

    lignes = []
    for r in relations:
        autre = g.autre_extremite(r, e["id"])
        lignes.append({
            "Début": r["debut"] or "—",
            "Fin": r["fin"] or "—",
            "Relation": r["type"],
            "Entité": f"{autre['label']} ({autre['type']})",
            "Nature": NATURE_LIBELLE.get(r["nature"], r["nature"]),
            "Statut": statut_texte(r),
            "Confiance": confiance_texte(r),
            "Preuve": preuve_texte(r),
            "Sources": sources_texte(r),
        })
    return relations, lignes


def h_parcours(g, a, entites):
    notes, tableaux, relations_api = [], [], []

    if entites:
        cibles = entites[:MAX_ENTITES_PROFIL]
    else:
        candidats = []
        for e in g.entites.values():
            if e["categorie"] != "PERSONNE":
                continue
            if EXCLURE_ETRANGERS and perimetre_personne(e) == "etranger":
                continue
            pol, _ = liens_qualifies(g, e["id"])
            if not pol:
                continue
            datees = sum(
                1 for i in g.adj.get(e["id"], [])
                if g.relations[i]["type"] in TYPES_PARCOURS
                and (g.relations[i]["debut"] or g.relations[i]["fin"])
            )
            candidats.append((datees, g.degre(e["id"]), e))

        candidats.sort(key=lambda c: (c[0], c[1]), reverse=True)
        cibles = [c[2] for c in candidats[:MAX_PARCOURS_PERSONNES]]

        notes.append(
            f"Aucune personne n'est citée : parcours des "
            f"{len(cibles)} personne(s) ayant des liens politiques et le "
            "plus de relations datées. Citez un nom pour un parcours précis."
        )

    total = 0
    for e in cibles:
        relations, lignes = lignes_parcours(g, e, a, notes)
        if not lignes:
            notes.append(f"Aucune relation de parcours pour {e['label']}.")
            continue
        total += 1
        sans_date = sum(1 for r in relations if not (r["debut"] or r["fin"]))
        if sans_date:
            notes.append(
                f"{e['label']} : {sans_date} relation(s) sans date, placée(s) "
                "en fin de parcours (période non renseignée)."
            )
        tableaux.append(tableau(
            f"Parcours de {e['label']}"
            + (f" — {tronquer(e['description'], 90)}"
               if e["description"] else ""),
            COLONNES_PARCOURS, lignes,
        ))
        relations_api.extend(relation_api(g, r) for r in relations)

    notes.append(
        "Chronologie reconstituée à partir des dates enregistrées dans le "
        "graphe ; elle ne prouve pas qu'une fonction soit encore exercée."
    )

    return nouveau_resultat(
        "parcours", "Parcours (chronologie des fonctions et appartenances)",
        tableaux, total, notes, relations_api,
    )


def h_connexions(g, a, limite):
    categories = ("PARTI", "INSTITUTION", "ENTREPRISE", "ORGANISATION")
    groupes, notes = [], []

    for e in g.entites.values():
        if e["categorie"] not in categories or e["qid_brut"]:
            continue
        paires = [
            (r, p) for r, p in voisins_par_type(g, e["id"], None,
                                                 {"PERSONNE"})
            if r["type"] not in FAMILIAUX
        ]
        personnes = {p["id"]: p for _, p in paires}
        if len(personnes) >= 2:
            groupes.append((e, personnes, paires))

    groupes.sort(key=lambda x: -len(x[1]))

    lignes_orgs = [{
        "Entité": e["label"],
        "Catégorie": CATEGORIE_LIBELLE.get(e["categorie"], e["categorie"]),
        "Personnes reliées": len(personnes),
        "Exemples": noms_courts(paires, 5, False),
    } for e, personnes, paires in groupes]

    commun = defaultdict(list)
    ignorees = 0
    for e, personnes, _ in groupes:
        if len(personnes) > GROUPE_MAX_PERSONNES:
            ignorees += 1
            continue
        ids = sorted(personnes, key=str)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                commun[(ids[x], ids[y])].append(e["label"])

    paires_triees = sorted(
        commun.items(),
        key=lambda kv: (-len(kv[1]), g.entites[kv[0][0]]["label"]),
    )

    lignes_paires = [{
        "Personne A": g.entites[x]["label"],
        "Personne B": g.entites[y]["label"],
        "Entités en commun": len(entites_communes),
        "Détail": liste_courte(entites_communes, 5, ", "),
    } for (x, y), entites_communes in paires_triees]

    directs = [
        r for r in g.relations
        if g.entites[r["source_id"]]["categorie"] == "PERSONNE"
        and g.entites[r["cible_id"]]["categorie"] == "PERSONNE"
    ]

    if ignorees:
        notes.append(
            f"{ignorees} entité(s) reliant plus de {GROUPE_MAX_PERSONNES} "
            "personnes ne sont pas utilisées pour former des paires."
        )

    notes.append(
        "Connexion indirecte = deux personnes reliées à une même entité. "
        "Ce n'est ni une relation directe ni la preuve d'un contrôle."
    )

    return nouveau_resultat(
        "connexions", "Connexions directes et indirectes entre personnes",
        [
            tableau("Liens directs entre personnes", COLONNES_RELATION,
                    [ligne_relation(g, r) for r in trier_relations(directs)]),
            tableau("Entités qui relient plusieurs personnes (connexions "
                    "indirectes)",
                    ["Entité", "Catégorie", "Personnes reliées", "Exemples"],
                    lignes_orgs),
            tableau("Paires de personnes ayant des entités en commun",
                    ["Personne A", "Personne B", "Entités en commun",
                     "Détail"], lignes_paires),
        ],
        len(directs) + len(lignes_paires), notes,
        [relation_api(g, r) for r in directs[:LIMITE_API_MAX]],
    )


def h_relations_nature(g, a, limite):
    if a["politique"] and not a["economique"]:
        natures = ["politique"]
    elif a["economique"] and not a["politique"]:
        natures = ["economique"]
    else:
        natures = ["politique", "economique"]

    notes = []
    relations = [r for r in g.relations if r["nature"] in natures]
    relations = filtrer_periode(relations, a["periode"], notes)

    compte = Counter((r["nature"], r["type"]) for r in relations)

    tableaux = [tableau(
        "Répartition par nature et par type de relation",
        ["Nature", "Type", "Nombre"],
        [{"Nature": NATURE_LIBELLE[n], "Type": t, "Nombre": k}
         for (n, t), k in sorted(compte.items(),
                                 key=lambda kv: (kv[0][0], -kv[1]))],
    )]

    for nature in natures:
        sous = [r for r in relations if r["nature"] == nature]
        sous.sort(key=lambda r: (len(r["extraits"]), r["confiance"] or 0),
                  reverse=True)
        tableaux.append(tableau(
            f"Relations {NATURE_LIBELLE[nature]}s (les mieux documentées "
            "d'abord)",
            COLONNES_RELATION, [ligne_relation(g, r) for r in sous],
        ))

    notes.append(
        "La nature (politique / économique) est déduite du type et de la "
        "cible de chaque relation ; les liens non qualifiés ne sont pas "
        "comptés. Une relation n'est pas une preuve d'influence."
    )

    return nouveau_resultat(
        "relations_nature", "Relations "
        + " et ".join(NATURE_LIBELLE[n] + "s" for n in natures),
        tableaux, len(relations), notes,
        [relation_api(g, r) for r in relations[:LIMITE_API_MAX]],
    )


# ============================================================
# 12. ORCHESTRATION
# ============================================================

def nom_generique(label) -> bool:
    mots = normaliser(label).split()
    return bool(mots) and mots[0] in PLURIELS_GENERIQUES


LLM_INTENT_PROMPT = """Tu es un classificateur d'intention pour une API
de graphe de connaissances politiques et économiques sur le Maroc.

Réponds UNIQUEMENT en JSON valide :
{
  "intent": "recherche_entite|recherche_relation|actionnariat|
             recherche_globale_relation|comptage_relation|
             recherche_chemin|recherche_personne|recherche_entreprise|
             recherche_politique|recherche_economique|
             relations_categories|stats|hors_perimetre",
  "relation_types": ["ACTIONNAIRE_DE"],
  "entities": ["Aziz Akhannouch", "Akwa Group"],
  "person_explicit": true,
  "economic": true,
  "political": false,
  "global_search": false,
  "count": false,
  "path": false,
  "confidence": 0.85
}

Règles :
- Deux entités citées → intent recherche_relation.
- « Combien de relations X » → intent comptage_relation.
- « Quelles relations X existent » → intent recherche_globale_relation.
- « Quel chemin relie A à B » → intent recherche_chemin.
- Aucune entité, aucun type → hors_perimetre.
"""

LLM_INTENT_MAP = {
    "recherche_entite": "entite",
    "recherche_relation": "entite",
    "actionnariat": "entite",
    "recherche_globale_relation": "relations_roles",
    "comptage_relation": "comptage_relation",
    "recherche_chemin": "chemins",
    "recherche_personne": "personnes_politiques",
    "recherche_entreprise": "liste_categorie",
    "recherche_politique": "personnes_politiques",
    "recherche_economique": "personnes_economiques",
    "relations_categories": "relations_categories",
    "stats": "stats",
    "hors_perimetre": "hors_perimetre",
}


def classifier_intention_llm(question):
    if groq_client is None:
        return None
    try:
        completion = groq_client.chat.completions.create(
            model=obtenir_modele_prioritaire(),
            messages=[
                {"role": "system", "content": LLM_INTENT_PROMPT},
                {"role": "user", "content": question[:2000]},
            ],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        if not completion.choices:
            return None
        contenu = completion.choices[0].message.content or "{}"
        return json.loads(contenu)
    except Exception as exc:
        logger.warning("LLM intent classification échouée : %s", exc)
        return None


def construire_resultat(question, limite=LIMITE_DEFAUT):
    g = obtenir_graphe()
    a = analyser_question(question)

    entites = detecter_entites(g, question)

    entites = [
        e for e in entites
        if not (
            e["score"] < 100 and e["categorie"] in ("PARTI", "INSTITUTION")
        )
        and not nom_generique(e["label"])
    ]

    intention = decider_intention(a, entites)

    if intention == "hors_perimetre" and groq_client is not None:
        llm = classifier_intention_llm(question)
        if llm and llm.get("intent") and llm["intent"] != "hors_perimetre":
            mapped = LLM_INTENT_MAP.get(llm["intent"])
            if mapped:
                intention = mapped
                logger.info(
                    "Intent LLM (fallback) : %s -> %s",
                    llm["intent"], intention,
                )

            if not entites and llm.get("entities"):
                for nom in llm["entities"][:4]:
                    trouve = detecter_entites(g, nom, 1)
                    entites.extend(trouve)

            if llm.get("relation_types"):
                a["types_relation"] = list(dict.fromkeys(
                    a["types_relation"] + llm["relation_types"]
                ))
                a["types_relation_explicites"] = list(dict.fromkeys(
                    a.get("types_relation_explicites", [])
                    + llm["relation_types"]
                ))

            if llm.get("count"):
                a["comptage"] = True
            if llm.get("path"):
                a["chemin"] = True

    if entites and intention in ("hors_perimetre", "comptage_relation"):
        if len(entites) >= 2 and a["chemin"]:
            intention = "chemins"
        elif intention == "hors_perimetre":
            intention = "entite"

    if intention == "chemins":
        resultat = h_chemins(g, a, entites)
    elif intention == "comptage_relation":
        resultat = h_comptage_relation(g, a, limite)
    elif intention == "entite":
        resultat = h_entite(g, a, entites)
    elif intention == "personnes_politiques_economiques":
        resultat = h_personnes(g, a, True, True, limite)
    elif intention == "personnes_politiques":
        resultat = h_personnes(g, a, True, False, limite)
    elif intention == "personnes_economiques":
        resultat = h_personnes(g, a, False, True, limite)
    elif intention == "personnes_fonctions":
        resultat = h_personnes_fonctions(g, a, limite)
    elif intention == "parcours":
        resultat = h_parcours(g, a, entites)
    elif intention == "connexions":
        resultat = h_connexions(g, a, limite)
    elif intention == "relations_nature":
        resultat = h_relations_nature(g, a, limite)
    elif intention == "classement_politique":
        resultat = h_classement_politique(g, a, limite)
    elif intention == "classement":
        resultat = h_classement(g, a, limite)
    elif intention == "relations_categories":
        resultat = h_relations_categories(g, a, limite)
    elif intention == "documentees":
        resultat = h_documentees(g, a, limite)
    elif intention == "apercu_reseaux":
        resultat = h_apercu(g, a, limite, reseaux=True)
    elif intention == "stats":
        resultat = h_stats(g, a, limite)
    elif intention == "relations_roles":
        resultat = h_relations_roles(g, a, limite)
    elif intention == "liste_categorie":
        resultat = h_liste_categorie(g, a, limite)
    else:
        resultat = h_hors_perimetre(g)

    resultat["entites_detectees"] = [
        {"id": e["id"], "label": e["label"], "type": e["type"],
         "score": e["score"]}
        for e in entites
    ]
    resultat["analyse"] = a
    resultat["graphe"] = {
        "entites": len(g.entites),
        "relations": len(g.relations),
        "noeuds_neo4j": g.noeuds_neo4j,
        "relations_neo4j": g.relations_neo4j,
        "perimetre": "metier",
        "labels_exclus": sorted(LABELS_EXCLUS),
        "relations_exclues": sorted(RELATIONS_EXCLUES),
        "charge_le": g.charge_le,
    }
    return resultat


# ============================================================
# 13. RENDU MARKDOWN
# ============================================================

def cellule(valeur, n=140):
    if valeur in (None, "", []):
        return "—"
    texte = str(valeur).replace("|", "/").replace("\n", " ").strip()
    return texte if n is None or len(texte) <= n else tronquer(texte, n)


def rendre_tableau(t, limite):
    lignes = t["lignes"][:limite]
    sortie = [
        f"### {t['titre']}",
        f"_{len(lignes)} ligne(s) affichée(s) sur {t['total']}._"
        if t["total"] > len(lignes) else f"_{t['total']} ligne(s)._",
        "",
        "| " + " | ".join(t["colonnes"]) + " |",
        "|" + "|".join(["---"] * len(t["colonnes"])) + "|",
    ]
    for ligne in lignes:
        sortie.append(
            "| " + " | ".join(
                cellule(ligne.get(c), 500 if c == "Sources" else 140)
                for c in t["colonnes"]
            ) + " |"
        )
    return "\n".join(sortie)


def rendre_markdown(resultat, limite):
    parties = []
    for t in resultat["tableaux"]:
        if t["lignes"]:
            parties.append(rendre_tableau(t, limite))
    if resultat["notes"]:
        parties.append(
            "### Notes et limites\n"
            + "\n".join(f"- {n}" for n in resultat["notes"])
        )
    return "\n\n".join(parties)


# ============================================================
# 14. SYNTHESE GROQ (FACULTATIVE)
# ============================================================

SYSTEM_PROMPT = """
Tu es l'assistant intelligent de GraphRAG Maroc, une application de
recherche et d'analyse sur les personnalités, institutions, partis,
entreprises et relations politiques et économiques liées au Maroc.

SOURCE DE VÉRITÉ : uniquement le graphe Neo4j chargé par l'API.
Tu ne dois JAMAIS compléter avec des connaissances générales.
Ne fabrique jamais une entité, une relation, une date, une fonction,
un pourcentage, une source, un statut ou un niveau de confiance.

RÈGLES ABSOLUES :
- Ne transforme JAMAIS un statut null/absent en A_VERIFIER ni en VERIFIE.
- Ne transforme JAMAIS confidence=0.9 en « vérifié ».
- ACTIONNAIRE_DE ne prouve ni pourcentage, ni contrôle effectif, ni
  situation actuelle.
- OCCUPE_FONCTION ne prouve pas que la fonction est encore exercée.
- Un chemin indirect n'est ni une relation directe ni un contrôle.
- Le nombre de relations n'est pas une mesure d'influence.
- Un lien politique + un lien économique ≠ conflit d'intérêts.
- Un score technique n'est jamais une réalité sociale ou politique.

STATUTS (présentation API) :
- VERIFIE       → « Vérifié »
- A_VERIFIER    → « À vérifier »
- null/absent   → « Non renseigné » (ni vérifié, ni à vérifier)

PÉRIMÈTRE :
- Graphe métier : entités :Entite (hors Customer, Product,
  GraphRAGEntity, Evidence, GraphRAGMarocEvidence).
- Relations métier : toutes sauf KNOWS, GRAPHRAG_RELATION, PURCHASED,
  EVIDENCE_FOR, EVIDENCE_TARGET.
- EVIDENCE_* = traçabilité uniquement.

TA TÂCHE, en français :
1. « Synthèse » : 120 mots MAXIMUM qui répond à la question UNIQUEMENT
   avec les résultats fournis. Reprends le TOTAL EXACT sans le modifier.
2. « Limites » : 5 puces MAXIMUM.

INTERDICTIONS DE FORME :
- N'écris AUCUN tableau (ils sont ajoutés automatiquement après toi).
- Ne recopie pas les lignes une par une.
- Ne cite que des entités présentes dans les résultats.

SITUATIONS DE RÉSULTAT VIDE :
A. Entité introuvable → « Cette entité n'a pas été retrouvée dans le
   sous-graphe interrogé. »
B. Entité trouvée, relation absente → « L'entité a été retrouvée, mais
   aucune relation directe correspondant au type demandé n'a été
   retournée par le graphe interrogé. »
C. Relation A_VERIFIER → « présente mais marquée A_VERIFIER, non
   confirmée. »
D. Relation sans statut → « présente, statut de vérification non
   renseigné. »

Ne confonds JAMAIS : zéro résultat ≠ absence de preuve ≠ preuve
d'absence.

Si le total est 0 ou les données insuffisantes, écris « Information
insuffisante » et explique ce qui manque.
Si seule une partie des résultats t'est fournie, dis-le.
"""


def obtenir_modele_prioritaire():
    with groq_model_lock:
        return GROQ_MODELS[groq_model_index]


def definir_modele_prioritaire(index):
    global groq_model_index
    with groq_model_lock:
        groq_model_index = index % len(GROQ_MODELS)


def ordre_modeles():
    with groq_model_lock:
        depart = groq_model_index
    return [(depart + i) % len(GROQ_MODELS) for i in range(len(GROQ_MODELS))]


def extraire_code_erreur(exc):
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def delai_reessai(exc, defaut=15):
    trouve = re.search(
        r"try again in (?:(\d+)m)?\s*([0-9.]+)s", str(exc)
    )
    if trouve:
        return int(trouve.group(1) or 0) * 60 + float(trouve.group(2)) + 1
    return defaut


def contexte_llm(resultat, nb_lignes, taille_max):
    def compact(t, n):
        return {
            "titre": t["titre"],
            "total": t["total"],
            "colonnes": t["colonnes"],
            "lignes": [
                [tronquer(cellule(l.get(c)), 90)
                 for c in t["colonnes"] if c != "Sources"]
                for l in t["lignes"][:n]
            ],
        }

    while True:
        charge = {
            "intention": resultat["intention"],
            "TOTAL_EXACT": resultat["total"],
            "entites_detectees": [
                {"label": e["label"], "type": e.get("type", "")}
                for e in resultat.get("entites_detectees", [])
            ][:6],
            "notes": [tronquer(n, 300) for n in resultat["notes"][:6]],
            "tableaux": [
                compact(t, nb_lignes) for t in resultat["tableaux"]
                if t["lignes"]
            ][:3],
        }
        texte = json.dumps(
            charge, ensure_ascii=False, separators=(",", ":"), default=str
        )
        if len(texte) <= taille_max or nb_lignes <= 1:
            return texte
        nb_lignes = max(1, nb_lignes // 2)


def appel_groq(modele, question, contexte, max_tokens):
    extra = {}
    if "gpt-oss" in modele:
        extra["extra_body"] = {"reasoning_effort": "low"}
        max_tokens = max(max_tokens, 2500)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "QUESTION :\n" + question[:2000]
                + "\n\nRÉSULTATS CALCULÉS :\n" + contexte
            ),
        },
    ]

    try:
        completion = groq_client.chat.completions.create(
            model=modele,
            messages=messages,
            temperature=GROQ_TEMPERATURE,
            max_tokens=max_tokens,
            top_p=0.9,
            **extra,
        )
    except TypeError:
        completion = groq_client.chat.completions.create(
            model=modele,
            messages=messages,
            temperature=GROQ_TEMPERATURE,
            max_tokens=max_tokens,
            **extra,
        )

    if not completion.choices:
        raise RuntimeError("Réponse Groq vide.")

    choix = completion.choices[0]
    texte = (choix.message.content or "").strip()

    if not texte:
        raise RuntimeError("Le modèle a retourné une réponse vide.")

    if getattr(choix, "finish_reason", None) == "length":
        texte += (
            "\n\n_Synthèse tronquée par la limite de tokens ; les tableaux "
            "ci-dessous sont complets._"
        )

    return texte


def generer_synthese(question, resultat):
    if groq_client is None:
        return None, ["GROQ_API_KEY non configurée"]

    erreurs = []
    niveaux = [
        (MAX_LLM_LIGNES, MAX_CONTEXT_CHARS),
        (max(3, MAX_LLM_LIGNES // 2), min(MAX_CONTEXT_CHARS, 3500)),
        (2, 1800),
    ]

    for index in ordre_modeles():
        modele = GROQ_MODELS[index]

        for nb_lignes, taille in niveaux:
            contexte = contexte_llm(resultat, nb_lignes, taille)
            reessaye = False

            while True:
                try:
                    texte = appel_groq(
                        modele, question, contexte, GROQ_MAX_TOKENS
                    )
                    definir_modele_prioritaire(index)
                    return texte, erreurs

                except Exception as exc:
                    code = extraire_code_erreur(exc)
                    logger.warning(
                        "Groq %s HTTP %s : %s", modele, code,
                        str(exc)[:250],
                    )
                    erreurs.append(f"{modele} : HTTP {code or 'inconnu'}")

                    if code == 429 and not reessaye:
                        attente = delai_reessai(exc)
                        if attente <= 12:
                            reessaye = True
                            time.sleep(attente)
                            continue
                    break

            if not erreurs or "HTTP 413" not in erreurs[-1]:
                break

    return None, erreurs


def synthese_locale(resultat):
    if resultat["intention"] == "hors_perimetre":
        return "Information insuffisante pour répondre à cette question."

    if resultat["total"] == 0:
        return (
            "Information insuffisante : aucun résultat correspondant n'a "
            "été trouvé dans le graphe. Cela ne prouve pas que la relation "
            "n'existe pas : elle peut être absente ou non documentée."
        )

    return (
        f"**Résultat : {resultat['titre']}.** {resultat['total']} élément(s) "
        "trouvé(s) dans le graphe. Les tableaux ci-dessous donnent le "
        "détail, avec statuts, dates, confiance et sources."
    )


def composer_reponse(question, resultat, limite):
    erreurs = []

    if resultat["total"] == 0 or resultat["intention"] == "hors_perimetre":
        synthese = synthese_locale(resultat)
    else:
        synthese, erreurs = generer_synthese(question, resultat)
        if synthese is None:
            synthese = synthese_locale(resultat)

    corps = rendre_markdown(resultat, limite)
    reponse = synthese + ("\n\n---\n\n" + corps if corps else "")
    return reponse, erreurs


# ============================================================
# 15. FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, driver

    logger.info("Démarrage de GraphRAG API %s", VERSION)

    try:
        get_driver().verify_connectivity()
        logger.info("Connexion Neo4j réussie.")

        if GROQ_API_KEY and Groq is not None:
            groq_client = Groq(
                api_key=GROQ_API_KEY, max_retries=0, timeout=60.0
            )
            logger.info("Client Groq initialisé : %s", GROQ_MODELS)
        else:
            logger.warning(
                "Groq indisponible (clé absente ou paquet non installé) : "
                "synthèse locale seule."
            )

        charger_graphe()

    except Exception:
        logger.exception("Erreur au démarrage")
        if driver is not None:
            driver.close()
            driver = None
        raise

    yield

    if driver is not None:
        driver.close()
        driver = None

    groq_client = None
    logger.info("Arrêt de GraphRAG API.")


app = FastAPI(
    title="GraphRAG Maroc API",
    description=(
        "Exploration du graphe des relations politiques, économiques et "
        "institutionnelles au Maroc. Résultats calculés sur l'ensemble du "
        "graphe, avec sources et statuts."
    ),
    version=VERSION,
    lifespan=lifespan,
)


if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )


def exiger_admin(x_admin_token: Optional[str] = Header(None)):
    if not ADMIN_TOKEN:
        return
    if not x_admin_token or not hmac.compare_digest(
        x_admin_token, ADMIN_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Jeton admin invalide.")


ADMIN = [Depends(exiger_admin)]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    limite: int = Field(
        LIMITE_DEFAUT, ge=1, le=LIMITE_API_MAX,
        description="Nombre maximal de lignes affichées par tableau.",
    )


@app.get("/")
def accueil():
    return {
        "application": "GraphRAG Maroc API",
        "version": VERSION,
        "statut": "en ligne",
        "documentation": "/docs",
        "neo4j_database": NEO4J_DATABASE,
        "graph_label": GRAPH_LABEL or "(tous les nœuds :Entite)",
        "relation_source": RELATION_SOURCE or "(aucun filtre)",
        "modeles_groq": GROQ_MODELS,
        "modele_prioritaire": obtenir_modele_prioritaire(),
        "endpoints": [
            "/health", "/stats", "/ask", "/entites", "/entite/{id}",
            "/entite/recherche/{nom}", "/politique/classement",
            "/relations/personnes-entreprises",
            "/relations/entre/{nom_a}/{nom_b}", "/debug/graphe",
            "/debug/env", "/debug/chargement-funnel",
            "/debug/trace-relation/{src}/{cib}/{type}",
            "/admin/recharger",
        ],
        "exemples": PRESENTATION_EXEMPLES,
        "avertissement": (
            "Si GRAPH_LABEL ou RELATION_SOURCE sont renseignés dans le "
            ".env, certaines relations peuvent être silencieusement "
            "exclues. Vider ces variables pour charger tout le graphe "
            "métier."
        ),
    }


@app.get("/health")
def health():
    try:
        get_driver().verify_connectivity()
        with graphe_lock:
            g = GRAPHE
        return {
            "status": "ok",
            "neo4j": "connecté",
            "groq": (
                "configuré" if groq_client is not None else "non configuré"
            ),
            "graphe_charge": g is not None,
            "entites": len(g.entites) if g else 0,
            "relations": len(g.relations) if g else 0,
            "noeuds_neo4j": g.noeuds_neo4j if g else None,
            "relations_neo4j": g.relations_neo4j if g else None,
            "version": VERSION,
        }
    except Exception as exc:
        logger.exception("Health check échoué")
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/stats")
def statistiques():
    g = obtenir_graphe()
    return {
        "version": VERSION,
        "perimetre": bloc_perimetre(g),
        "neo4j": {
            "noeuds_total_base": g.noeuds_neo4j,
            "relations_total_base": g.relations_neo4j,
            "entites_chargees": len(g.entites),
            "relations_chargees": len(g.relations),
        },
        "nombre_entites": len(g.entites),
        "nombre_relations": len(g.relations),
        "personnes": sum(
            1 for e in g.entites.values() if e["categorie"] == "PERSONNE"
        ),
        "entreprises": sum(
            1 for e in g.entites.values() if e["categorie"] == "ENTREPRISE"
        ),
        "entites_par_categorie": dict(
            Counter(e["categorie"] for e in g.entites.values())
        ),
        "relations_par_type": dict(Counter(r["type"] for r in g.relations)),
        "relations_par_nature": dict(
            Counter(r["nature"] for r in g.relations)
        ),
        "relations_par_statut": dict(
            Counter(r["statut"] for r in g.relations)
        ),
        "confiance_moyenne": confiance_moyenne(g),
        "relations_avec_confiance": sum(
            1 for r in g.relations if r["confiance"] is not None
        ),
        "relations_avec_extrait": sum(1 for r in g.relations if r["extraits"]),
        "relations_ignorees_au_chargement": dict(g.rejets),
        "charge_le": g.charge_le,
        "base_neo4j": NEO4J_DATABASE,
        "modeles_groq": GROQ_MODELS,
        "modele_prioritaire": obtenir_modele_prioritaire(),
    }


@app.get("/entites")
def lister_entites(
    recherche: str = Query(..., min_length=1),
    limite: int = Query(10, ge=1, le=100),
):
    g = obtenir_graphe()
    resultats = detecter_entites(g, recherche, limite)
    return {
        "recherche": recherche,
        "nombre_resultats": len(resultats),
        "resultats": [
            {"id": e["id"], "qid": e.get("qid"), "label": e["label"],
             "type": e["type"], "categorie": e["categorie"],
             "score": e["score"], "description": e["description"],
             "nombre_relations": g.degre(e["id"])}
            for e in resultats
        ],
    }


@app.get("/entite/recherche/{nom}")
def recherche_entite(nom: str, limite: int = Query(5, ge=1, le=50)):
    g = obtenir_graphe()
    resultats = detecter_entites(g, nom, limite)

    if not resultats:
        recherche = normaliser(nom)
        if len(recherche) >= 3:
            trouves = [
                {**e, "score": 40} for e in g.entites.values()
                if not e["qid_brut"]
                and any(recherche in n for n in e["noms"])
            ]
            trouves.sort(key=lambda e: g.degre(e["id"]), reverse=True)
            resultats = trouves[:limite]

    if not resultats:
        raise HTTPException(
            status_code=404, detail="Aucune entité correspondante."
        )

    return {
        "recherche": nom,
        "nombre_resultats": len(resultats),
        "resultats": [
            {**entite_api(e), "score": e["score"],
             "nombre_relations": g.degre(e["id"])}
            for e in resultats
        ],
    }


@app.get("/entite/{entite_id}")
def profil_entite(entite_id: str):
    g = obtenir_graphe()

    if entite_id not in g.entites:
        qid = entite_id.upper()
        trouve = next(
            (e for e in g.entites.values()
             if str(e.get("qid") or "").upper() == qid), None
        )
        if trouve is None:
            raise HTTPException(status_code=404, detail="Entité inconnue.")
        entite_id = trouve["id"]

    e = g.entites[entite_id]
    relations = trier_relations(
        [g.relations[i] for i in g.adj.get(entite_id, [])]
    )

    reponse = {
        "entite": entite_api(e),
        "perimetre": (
            perimetre_personne(e) if e["categorie"] == "PERSONNE" else None
        ),
        "nombre_relations": len(relations),
        "relations": [relation_api(g, r) for r in relations],
    }

    if e["categorie"] == "PERSONNE":
        reponse["relations_politiques"] = [
            relation_api(g, r) for r in relations
            if r["nature"] == "politique"
        ]
        reponse["relations_entreprises_directes"] = [
            relation_api(g, r) for p, ent, r
            in relations_personnes_entreprises(g, True)
            if p["id"] == entite_id
        ]

    return reponse


@app.get("/politique/classement")
def api_classement_politique(
    limite: int = Query(30, ge=1, le=LIMITE_API_MAX),
    inclure_etrangers: bool = Query(False),
):
    g = obtenir_graphe()
    fiches, _ = classement_politique(g, None, inclure_etrangers)
    return {
        "total": len(fiches),
        "affiches": min(len(fiches), limite),
        "criteres": (
            "Classement fondé sur les relations politiques directes "
            "pondérées. Les relations familiales, scolaires, "
            "géographiques, judiciaires et techniques sont exclues. "
            "Ce score mesure la présence de relations politiques dans le "
            "graphe, pas l'influence réelle."
        ),
        "poids": POIDS_POLITIQUES,
        "resultats": [{
            "rang": i,
            "personne": entite_api(f["personne"]),
            "perimetre": f["perimetre"],
            "score_politique": f["score"],
            "nombre_relations_politiques": len(f["relations"]),
            "relations_par_type": f["par_type"],
        } for i, f in enumerate(fiches[:limite], start=1)],
    }


@app.get("/relations/personnes-entreprises")
def api_relations_personnes_entreprises(
    limite: int = Query(30, ge=1, le=LIMITE_API_MAX),
    inclure_etrangers: bool = Query(False),
):
    g = obtenir_graphe()
    trouvees = relations_personnes_entreprises(g, inclure_etrangers)
    return {
        "total": len(trouvees),
        "affiches": min(len(trouvees), limite),
        "precision": (
            "Seules les relations DIRECTES entre une personne et une "
            "entreprise sont présentées. Les chemins indirects ne sont pas "
            "assimilés à des liens de propriété ou de contrôle."
        ),
        "resultats": [{
            "personne": entite_api(p),
            "entreprise": entite_api(ent),
            "relation": relation_api(g, r),
            "nature": "DIRECTE",
        } for p, ent, r in trouvees[:limite]],
    }


def _chemin_api(g, chemin):
    entites = [g.entites[chemin[0][0]]["label"]]
    relations = []
    for (_, v, indices) in chemin:
        entites.append(g.entites[v]["label"])
        relations.extend(relation_api(g, g.relations[i]) for i in indices)
    return {
        "type": "DIRECT" if len(chemin) == 1 else "INDIRECT",
        "longueur": len(chemin),
        "entites": entites,
        "relations": relations,
    }


@app.get("/relations/entre/{nom_a}/{nom_b}")
def api_relations_entre(
    nom_a: str, nom_b: str,
    profondeur: int = Query(CHEMIN_MAX, ge=1, le=6),
):
    g = obtenir_graphe()
    trouve_a = detecter_entites(g, nom_a, 1)
    trouve_b = detecter_entites(g, nom_b, 1)

    if not trouve_a or not trouve_b:
        raise HTTPException(
            status_code=404,
            detail="Une ou plusieurs entités sont introuvables.",
        )

    a, b = trouve_a[0], trouve_b[0]

    if a["id"] == b["id"]:
        raise HTTPException(
            status_code=400, detail="Les deux noms désignent la même entité."
        )

    directes = [
        g.relations[i] for i in g.adj.get(a["id"], [])
        if g.autre_extremite(g.relations[i], a["id"])["id"] == b["id"]
    ]
    indirects = plus_courts_chemins(
        g, a["id"], b["id"], None, exclure_direct=True, profondeur=profondeur
    )

    return {
        "entite_a": entite_api(a),
        "entite_b": entite_api(b),
        "relations_directes": [relation_api(g, r) for r in directes],
        "nombre_relations_directes": len(directes),
        "chemins_indirects": [_chemin_api(g, c) for c in indirects],
        "nombre_chemins_indirects": len(indirects),
        "avertissement": (
            "Un chemin indirect indique une connexion dans le graphe. Il "
            "ne prouve pas à lui seul une relation de contrôle, de "
            "propriété, de financement ou d'influence."
        ),
    }


def _recharger():
    if not reload_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Un rechargement est déjà en cours."
        )
    try:
        return _recharger_interne()
    finally:
        reload_lock.release()


def _recharger_interne():
    try:
        g = charger_graphe()
        return {
            "message": "Graphe rechargé en mémoire.",
            "nombre_entites": len(g.entites),
            "nombre_relations": len(g.relations),
            "noeuds_neo4j": g.noeuds_neo4j,
            "relations_neo4j": g.relations_neo4j,
            "relations_ignorees": dict(g.rejets),
            "neo4j_modifie": False,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur rechargement")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reload", dependencies=ADMIN)
def recharger_index():
    return _recharger()


@app.post("/admin/recharger", dependencies=ADMIN)
def admin_recharger():
    return _recharger()


@app.get("/debug/graphe", dependencies=ADMIN)
def debug_graphe():
    p = params_perimetre()
    entites = executer(CYPHER_AUDIT, p)
    relations = executer(CYPHER_AUDIT_RELATIONS, p)

    with graphe_lock:
        g = GRAPHE

    return {
        "version": VERSION,
        "perimetre_configure": {
            "GRAPH_LABEL": GRAPH_LABEL or "(vide : tous les :Entite)",
            "RELATION_SOURCE": RELATION_SOURCE or "(vide : aucun filtre)",
        },
        "neo4j": {
            **(entites[0] if entites else {}),
            **(relations[0] if relations else {}),
            "noeuds_total_base": g.noeuds_neo4j if g else None,
            "relations_total_base": g.relations_neo4j if g else None,
        },
        "memoire": {
            "entites": len(g.entites) if g else 0,
            "relations": len(g.relations) if g else 0,
            "relations_ignorees": dict(g.rejets) if g else {},
            "charge_le": g.charge_le if g else None,
        },
        "perimetre_metier": bloc_perimetre(g) if g else None,
        "controle_perimetre": verifier_perimetre(g) if g else None,
        "conseil": (
            "Vérifiez que 'entites' en mémoire est proche de "
            "'entites_total'. Si 'entites_avec_label' ou "
            "'relations_avec_source' est bien inférieur au total, "
            "GRAPH_LABEL / RELATION_SOURCE doivent rester vides dans le "
            ".env."
        ),
    }


@app.get("/debug/env", dependencies=ADMIN)
def debug_env():
    """v6.3.2 : variables d'environnement effectives du loader."""
    return {
        "version": VERSION,
        "GRAPH_LABEL": {
            "valeur": GRAPH_LABEL or "(vide → tous les :Entite)",
            "impact": (
                "Filtre e.label IN labels(e). Si non vide, seuls les "
                "nœuds portant ce label sont chargés."
            ),
        },
        "RELATION_SOURCE": {
            "valeur": RELATION_SOURCE or "(vide → aucun filtre)",
            "impact": (
                "Filtre r.source = valeur. Si non vide, seules les "
                "relations portant r.source sont chargées."
            ),
        },
        "conclusion": (
            "Pour charger tout le graphe métier, laisser GRAPH_LABEL et "
            "RELATION_SOURCE VIDES dans le .env."
        ),
        "pistes": [
            "Vérifier .env avec : grep -E 'GRAPH_LABEL|RELATION_SOURCE' .env",
            "Un GRAPH_LABEL='GraphRAGMaroc' exclut Aziz Akhannouch "
            "(labels: Entite, Personne) et Akwa Group (Entite, Entreprise).",
        ],
    }


# --- v6.3.2 : diagnostic de chargement étape par étape ---

CYPHER_FUNNEL_ENTITES_TOTAL = """
MATCH (e:Entite) RETURN count(e) AS n
"""

CYPHER_FUNNEL_ENTITES_APRES_LABELS = """
MATCH (e:Entite)
WHERE NOT any(l IN labels(e) WHERE l IN $labels_exclus)
RETURN count(e) AS n
"""

CYPHER_FUNNEL_ENTITES_APRES_FUSIONNEE = """
MATCH (e:Entite)
WHERE NOT any(l IN labels(e) WHERE l IN $labels_exclus)
  AND coalesce(e.statut_entite, '') <> 'FUSIONNEE'
RETURN count(e) AS n
"""

CYPHER_FUNNEL_ENTITES_APRES_ID = """
MATCH (e:Entite)
WHERE NOT any(l IN labels(e) WHERE l IN $labels_exclus)
  AND coalesce(e.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(e.id, e.qid) IS NOT NULL
RETURN count(e) AS n
"""

CYPHER_FUNNEL_ACTIONNAIRE_TOTAL = """
MATCH ()-[r:ACTIONNAIRE_DE]->() RETURN count(r) AS n
"""

CYPHER_FUNNEL_ACTIONNAIRE_ENTITE = """
MATCH (a:Entite)-[r:ACTIONNAIRE_DE]->(b:Entite)
RETURN count(r) AS n
"""

CYPHER_FUNNEL_ACTIONNAIRE_FUSIONNEE = """
MATCH (a:Entite)-[r:ACTIONNAIRE_DE]->(b:Entite)
WHERE coalesce(a.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(b.statut_entite, '') <> 'FUSIONNEE'
RETURN count(r) AS n
"""

CYPHER_FUNNEL_ACTIONNAIRE_ID = """
MATCH (a:Entite)-[r:ACTIONNAIRE_DE]->(b:Entite)
WHERE coalesce(a.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(b.statut_entite, '') <> 'FUSIONNEE'
  AND coalesce(a.id, a.qid) IS NOT NULL
  AND coalesce(b.id, b.qid) IS NOT NULL
RETURN count(r) AS n
"""


def _compter(cypher, params=None):
    try:
        r = executer(cypher, params or {})
        return r[0]["n"] if r else 0
    except Exception as exc:
        logger.warning("Compter a échoué : %s", exc)
        return None


@app.get("/debug/chargement-funnel", dependencies=ADMIN)
def debug_chargement_funnel():
    """
    v6.3.2 : compte, à chaque étape du loader, combien d'entités et de
    ACTIONNAIRE_DE survivent. Révèle immédiatement le filtre fautif.
    """
    p = params_perimetre()

    etapes_entites = {
        "1_entite_total": _compter(CYPHER_FUNNEL_ENTITES_TOTAL),
        "2_apres_filtre_labels_exclus":
            _compter(CYPHER_FUNNEL_ENTITES_APRES_LABELS, p),
        "3_apres_statut_non_fusionnee":
            _compter(CYPHER_FUNNEL_ENTITES_APRES_FUSIONNEE, p),
        "4_apres_id_qid_non_null":
            _compter(CYPHER_FUNNEL_ENTITES_APRES_ID, p),
    }

    with graphe_lock:
        g = GRAPHE

    etapes_entites["5_chargees_en_memoire"] = (
        len(g.entites) if g else 0
    )
    etapes_entites["rejets_python"] = dict(g.rejets) if g else {}

    actionnaire = {
        "1_neo4j_total": _compter(CYPHER_FUNNEL_ACTIONNAIRE_TOTAL),
        "2_apres_endpoints_entite":
            _compter(CYPHER_FUNNEL_ACTIONNAIRE_ENTITE, p),
        "3_apres_statut_non_fusionnee":
            _compter(CYPHER_FUNNEL_ACTIONNAIRE_FUSIONNEE, p),
        "4_apres_id_non_null":
            _compter(CYPHER_FUNNEL_ACTIONNAIRE_ID, p),
        "5_en_memoire": (
            sum(1 for r in g.relations if r["type"] == "ACTIONNAIRE_DE")
            if g else 0
        ),
    }

    diag = []
    e2 = etapes_entites.get("2_apres_filtre_labels_exclus")
    e3 = etapes_entites.get("3_apres_statut_non_fusionnee")
    e4 = etapes_entites.get("4_apres_id_qid_non_null")
    e5 = etapes_entites.get("5_chargees_en_memoire")
    total_e = etapes_entites.get("1_entite_total")

    if total_e is not None and e2 is not None and e2 < total_e:
        diag.append(
            f"Perte au filtre labels_exclus : {total_e - e2} nœuds."
        )
    if e2 is not None and e3 is not None and e3 < e2:
        diag.append(
            f"Perte de {e2 - e3} nœuds à cause de "
            "statut_entite = 'FUSIONNEE'."
        )
    if e3 is not None and e4 is not None and e4 < e3:
        diag.append(
            f"Perte de {e3 - e4} nœuds à cause de id/qid null."
        )
    if e4 is not None and e5 is not None and e5 < e4:
        diag.append(
            f"Perte de {e4 - e5} nœuds côté Python "
            "(label manquant, collision d'id)."
        )

    a3 = actionnaire.get("3_apres_statut_non_fusionnee")
    a4 = actionnaire.get("4_apres_id_non_null")
    a5 = actionnaire.get("5_en_memoire")
    a1 = actionnaire.get("1_neo4j_total")
    if a1 is not None and a3 is not None and a3 < a1:
        diag.append(
            f"{a1 - a3} ACTIONNAIRE_DE perdues à cause de "
            "statut_entite='FUSIONNEE' sur un endpoint."
        )
    if a3 is not None and a4 is not None and a4 < a3:
        diag.append(
            f"{a3 - a4} ACTIONNAIRE_DE perdues à cause de id/qid null."
        )
    if a4 is not None and a5 is not None and a5 < a4:
        diag.append(
            f"{a4 - a5} ACTIONNAIRE_DE perdues côté Python "
            "(rejet dans construire_relation)."
        )

    return {
        "version": VERSION,
        "parametres": {
            "GRAPH_LABEL": GRAPH_LABEL or "(vide)",
            "RELATION_SOURCE": RELATION_SOURCE or "(vide)",
        },
        "entites_funnel": etapes_entites,
        "actionnaire_de_funnel": actionnaire,
        "diagnostic": diag or ["Aucune perte détectée."],
        "conseil": (
            "Si GRAPH_LABEL ou RELATION_SOURCE sont non vides, c'est la "
            "cause principale. Sinon, suivre la ligne de perte indiquée "
            "dans diagnostic."
        ),
    }


@app.get("/debug/trace-relation/{source_id}/{cible_id}/{type_rel}",
         dependencies=ADMIN)
def debug_trace_relation(source_id: str, cible_id: str, type_rel: str):
    """
    v6.3.2 : trace une relation spécifique à travers chaque étape du
    loader. Permet de savoir pourquoi une relation précise n'arrive pas
    en mémoire.
    """
    # 1. Existe-t-elle dans Neo4j ?
    cypher = (
        "MATCH (a)-[r:`" + type_rel + "`]->(b)\n"
        "WHERE coalesce(a.id, a.qid) = $src\n"
        "  AND coalesce(b.id, b.qid) = $cib\n"
        "RETURN a.label AS source, labels(a) AS labels_source,\n"
        "       a.statut_entite AS statut_source,\n"
        "       b.label AS cible, labels(b) AS labels_cible,\n"
        "       b.statut_entite AS statut_cible,\n"
        "       type(r) AS type, properties(r) AS props,\n"
        "       count(r) AS occurrences"
    )
    try:
        lignes = executer(cypher, {"src": source_id, "cib": cible_id})
    except Exception as exc:
        raise HTTPException(400, f"Requête invalide : {exc}")

    with graphe_lock:
        g = GRAPHE

    source_present = source_id in g.entites if g else False
    cible_present = cible_id in g.entites if g else False
    relation_presente = (
        any(
            r["type"] == type_rel
            and r["source_id"] == source_id
            and r["cible_id"] == cible_id
            for r in g.relations
        )
        if g else False
    )

    if not lignes or lignes[0]["occurrences"] == 0:
        conclusion = "Relation absente de Neo4j."
    elif not source_present and not cible_present:
        conclusion = (
            "Relation dans Neo4j, mais les 2 endpoints sont absents de "
            "la mémoire. Voir /debug/chargement-funnel."
        )
    elif not source_present:
        conclusion = f"Endpoint source ({source_id}) absent de la mémoire."
    elif not cible_present:
        conclusion = f"Endpoint cible ({cible_id}) absent de la mémoire."
    elif not relation_presente:
        conclusion = (
            "Les 2 endpoints sont en mémoire, mais la relation a été "
            "rejetée par construire_relation(). Vérifier les props."
        )
    else:
        conclusion = "Relation chargée correctement."

    return {
        "source_id": source_id,
        "cible_id": cible_id,
        "type": type_rel,
        "neo4j": lignes,
        "memoire": {
            "source_present": source_present,
            "cible_present": cible_present,
            "relation_presente": relation_presente,
        },
        "conclusion": conclusion,
        "parametres": {
            "GRAPH_LABEL": GRAPH_LABEL or "(vide)",
            "RELATION_SOURCE": RELATION_SOURCE or "(vide)",
        },
    }


@app.get("/debug/classification", dependencies=ADMIN)
def debug_classification(
    limite: int = Query(100, ge=1, le=1000),
    nature: Optional[str] = Query(None),
):
    g = obtenir_graphe()
    lignes = []
    for r in g.relations:
        if nature and r["nature"] != nature:
            continue
        s, c = g.entites[r["source_id"]], g.entites[r["cible_id"]]
        lignes.append({
            "source": s["label"], "type_source": s["type"],
            "relation": r["type"],
            "cible": c["label"], "type_cible": c["type"],
            "nature": r["nature"],
        })
    return {"total": len(lignes), "lignes": lignes[:limite]}


@app.get("/debug/types", dependencies=ADMIN)
def debug_types():
    g = obtenir_graphe()
    entites = Counter(
        (e["type"] or "(vide)", e["categorie"]) for e in g.entites.values()
    )
    relations = Counter(
        (r["type_original"], r["type"]) for r in g.relations
    )
    non_resolues = sum(1 for e in g.entites.values() if e["qid_brut"])
    return {
        "entites_par_type_brut": [
            {"type_brut": t, "categorie_api": c, "nombre": n}
            for (t, c), n in entites.most_common()
        ],
        "relations_par_type": [
            {"type_brut": t, "type_api": c, "nombre": n}
            for (t, c), n in relations.most_common()
        ],
        "entites_sans_libelle_exploitable": non_resolues,
        "relations_ignorees_au_chargement": dict(g.rejets),
        "conseil": (
            "Si un type brut apparaît en catégorie AUTRE, ajoutez-le dans "
            "CATEGORIES (graph_api.py)."
        ),
    }


@app.get("/debug/personnes", dependencies=ADMIN)
def debug_personnes(
    limite: int = Query(60, ge=1, le=500),
    nom: Optional[str] = Query(None),
):
    g = obtenir_graphe()
    lignes = []
    bilan = Counter()

    for e in g.entites.values():
        if e["categorie"] != "PERSONNE":
            continue
        if nom and normaliser(nom) not in normaliser(e["label"]):
            continue

        natures = Counter()
        details = []
        for idx in g.adj.get(e["id"], []):
            r = g.relations[idx]
            autre = g.autre_extremite(r, e["id"])
            natures[r["nature"]] += 1
            details.append(
                f"{r['type']} → {autre['label']} "
                f"[{autre['type']}→{autre['categorie']}] = {r['nature']}"
            )

        pol, eco = liens_qualifies(g, e["id"])
        perimetre = perimetre_personne(e)

        if pol and eco:
            raison = "RETENUE"
        elif not g.adj.get(e["id"]):
            raison = "aucune relation"
        elif not pol and not eco:
            raison = "aucun lien politique ni économique qualifié"
        elif not pol:
            raison = "lien économique seulement"
        else:
            raison = "lien politique seulement"

        if raison == "RETENUE" and EXCLURE_ETRANGERS \
                and perimetre == "etranger":
            raison = "exclue (nationalité étrangère)"

        bilan[raison] += 1
        lignes.append({
            "personne": e["label"],
            "description": tronquer(e["description"], 80),
            "perimetre": perimetre,
            "raison": raison,
            "natures": dict(natures),
            "liens": details[:8],
        })

    lignes.sort(
        key=lambda l: (l["raison"] != "RETENUE", -sum(l["natures"].values()))
    )

    return {
        "personnes": sum(bilan.values()),
        "bilan": dict(bilan),
        "lignes": lignes[:limite],
    }


@app.get("/debug/controles", dependencies=ADMIN)
def debug_controles(
    executer_requetes: bool = Query(
        False, description="Exécute les requêtes (lecture seule)."
    ),
):
    p = params_perimetre()
    controles = []
    for c in CONTROLES_CYPHER:
        item = dict(c)
        if executer_requetes:
            item["resultat"] = json_safe_simple(executer(c["cypher"], p))
        controles.append(item)
    return {
        "parametres": {
            k: p[k] for k in ("label", "labels_exclus", "labels_preuve",
                              "types_exclus")
        },
        "controles": controles,
        "index_recommandes": INDEX_RECOMMANDES,
        "note": (
            "Toutes les requêtes sont en lecture seule. Les index sont "
            "proposés à titre indicatif et ne sont jamais créés par l'API."
        ),
    }


def json_safe_simple(valeur):
    try:
        json.dumps(valeur, default=str)
        return json.loads(json.dumps(valeur, default=str))
    except Exception:
        return str(valeur)


@app.get("/entite/{entite_id}/preuves")
def preuves_entite(
    entite_id: str, limite: int = Query(LIMITE_PREUVES, ge=1, le=200)
):
    g = obtenir_graphe()
    if entite_id not in g.entites:
        raise HTTPException(status_code=404, detail="Entité métier inconnue.")

    lignes = executer(CYPHER_PREUVES_ENTITE, {
        "id": entite_id, "labels_preuve": sorted(LABELS_PREUVE),
        "limite": limite,
    })

    return {
        "entite": entite_api(g.entites[entite_id]),
        "nombre_preuves": len(lignes),
        "preuves": [
            {"role": l["role"], "preuve": json_safe_simple(l["preuve"])}
            for l in lignes
        ],
        "avertissement": (
            "Liens de traçabilité uniquement, jamais comptés comme "
            "relations politiques ou économiques."
        ),
    }


def traiter_question(question, limite):
    question = question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="La question ne peut pas être vide."
        )

    logger.info("Question reçue : %s", question)

    try:
        resultat = construire_resultat(question, limite)
        reponse, erreurs = composer_reponse(question, resultat, limite)

        logger.info(
            "Intention : %s | total : %s",
            resultat["intention"], resultat["total"],
        )

        return {
            "version": VERSION,
            "type_de_recherche": resultat["intention"],
            "question": question,
            "reponse": reponse,
            "entites_detectees": resultat["entites_detectees"],
            "nombre_relations": len(resultat["relations"]),
            "relations": resultat["relations"],
            "resultat": {
                "titre": resultat["titre"],
                "total": resultat["total"],
                "tableaux": resultat["tableaux"],
                "notes": resultat["notes"],
            },
            "analyse_question": resultat["analyse"],
            "graphe": resultat["graphe"],
            "modele_groq_prioritaire": obtenir_modele_prioritaire(),
            "erreurs_llm": erreurs,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur pendant le traitement de la question")
        raise HTTPException(
            status_code=500,
            detail={"erreur": "Erreur pendant le traitement",
                    "message": str(exc)},
        )


@app.post("/ask")
def poser_question(request: AskRequest):
    return traiter_question(request.question, request.limite)


@app.get("/ask")
def poser_question_get(
    question: str = Query(..., min_length=2, max_length=2000),
    limite: int = Query(LIMITE_DEFAUT, ge=1, le=LIMITE_API_MAX),
):
    return traiter_question(question, limite)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)