"""
GraphRAG Maroc : extension plateforme (phase 7 + évaluation)

Branchement (à la fin de graphrag_api.py, avant `if __name__ == "__main__":`) :

    import sys, extras
    extras.register(app, sys.modules[__name__])

Fournit :
  - authentification par rôles (lecteur < analyste < admin), jetons HS256
  - journal d'audit (mémoire + fichier JSONL) avec rétention
  - supervision (latences, erreurs, Neo4j, Groq, graphe)
  - qualité des données, sources, sous-graphe, détail d'une relation
  - évaluation GraphRAG vs Cypher direct (précision, rappel, F1, latence)
  - description de l'architecture

Aucune route existante n'est modifiée. Avec AUTH_REQUIRED=0 (défaut), rien
n'est bloqué : le comportement de l'API reste celui d'avant.
"""

import os
import re
import json
import time
import hmac
import base64
import hashlib
import secrets
import logging
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("graphrag.extras")
BASE_DIR = Path(__file__).resolve().parent
ctx = None  # module graphrag_api, injecté par register()


def _bool(nom, defaut):
    return os.getenv(nom, defaut).strip().lower() in ("1", "true", "oui", "yes")


# ============================================================
# 1. CONFIGURATION
# ============================================================

AUTH_REQUIRED = _bool("AUTH_REQUIRED", "0")
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_TTL_MIN = int(os.getenv("JWT_TTL_MIN", "480"))
AUTH_DEFAULT_ROLE = os.getenv("AUTH_DEFAULT_ROLE", "admin").strip()
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]
AUDIT_FILE = Path(os.getenv("AUDIT_FILE", str(BASE_DIR / "audit.jsonl")))
AUDIT_RETENTION_JOURS = int(os.getenv("AUDIT_RETENTION_JOURS", "90"))
AUDIT_MEMOIRE = int(os.getenv("AUDIT_MEMOIRE", "2000"))
EVAL_FILE = Path(os.getenv("EVAL_FILE", str(BASE_DIR / "eval_runs.jsonl")))

ROLES = {"lecteur": 1, "analyste": 2, "admin": 3}
DEMARRAGE = time.time()


def _lire_utilisateurs():
    """AUTH_USERS="admin:motdepasse:admin,ana:motdepasse:analyste"."""
    users = {}

    for bloc in os.getenv("AUTH_USERS", "").split(","):
        bloc = bloc.strip()
        if not bloc:
            continue

        morceaux = bloc.split(":", 2)

        if len(morceaux) != 3 or morceaux[2].strip() not in ROLES:
            logger.warning("AUTH_USERS : entrée ignorée (format invalide).")
            continue

        users[morceaux[0].strip()] = {
            "mot_de_passe": morceaux[1], "role": morceaux[2].strip(),
        }

    return users


UTILISATEURS = _lire_utilisateurs()

if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    if AUTH_REQUIRED:
        logger.warning(
            "JWT_SECRET absent : secret aléatoire, les sessions seront "
            "invalidées à chaque redémarrage."
        )


# ============================================================
# 2. JETONS (HS256, bibliothèque standard)
# ============================================================

def _b64(donnees: bytes) -> str:
    return base64.urlsafe_b64encode(donnees).rstrip(b"=").decode()


def _unb64(texte: str) -> bytes:
    return base64.urlsafe_b64decode(texte + "=" * (-len(texte) % 4))


def _signer(message: str) -> str:
    return _b64(hmac.new(JWT_SECRET.encode(), message.encode(),
                         hashlib.sha256).digest())


def creer_jeton(utilisateur, role):
    entete = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    charge = _b64(json.dumps({
        "sub": utilisateur, "role": role,
        "exp": int(time.time()) + JWT_TTL_MIN * 60,
    }).encode())
    return f"{entete}.{charge}.{_signer(entete + '.' + charge)}"


def lire_jeton(jeton):
    try:
        entete, charge, signature = jeton.split(".")

        if not hmac.compare_digest(signature, _signer(entete + "." + charge)):
            return None

        donnees = json.loads(_unb64(charge))

        if donnees.get("exp", 0) < time.time() or donnees.get("role") \
                not in ROLES:
            return None

        return donnees

    except Exception:
        return None


# ============================================================
# 3. AUDIT ET METRIQUES
# ============================================================

AUDIT = deque(maxlen=AUDIT_MEMOIRE)
_audit_lock = threading.Lock()
LATENCES = defaultdict(lambda: deque(maxlen=500))
STATUTS = Counter()
ERREURS = deque(maxlen=50)
ECHECS_CONNEXION = {}


def _charger_audit():
    if not AUDIT_FILE.exists():
        return

    try:
        with AUDIT_FILE.open(encoding="utf-8") as f:
            for ligne in deque(f, maxlen=AUDIT_MEMOIRE):
                try:
                    AUDIT.append(json.loads(ligne))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.warning("Lecture du journal d'audit impossible.")


def auditer(evenement):
    AUDIT.append(evenement)

    try:
        with _audit_lock, AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evenement, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Écriture du journal d'audit impossible.")


def purger_audit():
    """Supprime les entrées plus anciennes que la rétention. Retourne n."""

    limite = time.time() - AUDIT_RETENTION_JOURS * 86400
    gardees, supprimees = [], 0

    with _audit_lock:
        if AUDIT_FILE.exists():
            with AUDIT_FILE.open(encoding="utf-8") as f:
                for ligne in f:
                    try:
                        if json.loads(ligne).get("t", 0) >= limite:
                            gardees.append(ligne)
                        else:
                            supprimees += 1
                    except json.JSONDecodeError:
                        supprimees += 1

            AUDIT_FILE.write_text("".join(gardees), encoding="utf-8")

    restantes = [e for e in AUDIT if e.get("t", 0) >= limite]
    AUDIT.clear()
    AUDIT.extend(restantes)
    return supprimees


def percentile(valeurs, p):
    if not valeurs:
        return None

    s = sorted(valeurs)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 1)


def chemin_normalise(chemin):
    return re.sub(r"^(/entite|/graphe/relation)/[^/]+$", r"\1/{id}", chemin)


# ============================================================
# 4. GARDE (authentification, rôles, audit)
# ============================================================

PUBLICS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/auth/login"}


def role_requis(chemin):
    if chemin in PUBLICS:
        return None

    if chemin.startswith("/admin") or chemin == "/reload":
        return "admin"

    if chemin.startswith("/evaluation") or chemin.startswith("/debug") \
            or chemin == "/supervision":
        return "analyste"

    return "lecteur"


async def garde(request: Request, call_next):
    debut = time.perf_counter()
    chemin = request.url.path
    utilisateur, role, question = "anonyme", AUTH_DEFAULT_ROLE, None

    if request.method == "POST" and chemin == "/ask":
        corps = await request.body()

        async def rejouer():
            return {"type": "http.request", "body": corps,
                    "more_body": False}

        request._receive = rejouer

        try:
            question = str(json.loads(corps).get("question", ""))[:300]
        except Exception:
            question = None
    elif request.method == "GET" and chemin == "/ask":
        question = request.query_params.get("question", "")[:300]

    requis = None if request.method == "OPTIONS" else role_requis(chemin)
    refus = None

    if requis and AUTH_REQUIRED:
        en_tete = request.headers.get("authorization", "")
        jeton = lire_jeton(en_tete[7:]) if en_tete.lower().startswith(
            "bearer "
        ) else None

        if jeton is None:
            refus = (401, "Authentification requise ou jeton expiré.")
        else:
            utilisateur, role = jeton["sub"], jeton["role"]

            if ROLES[role] < ROLES[requis]:
                refus = (403, f"Rôle « {requis} » requis.")
    elif requis:
        utilisateur = "local"

    if refus:
        reponse = JSONResponse(status_code=refus[0],
                               content={"detail": refus[1]})
    else:
        try:
            reponse = await call_next(request)
        except Exception as exc:
            logger.exception("Erreur non gérée")
            reponse = JSONResponse(status_code=500,
                                   content={"detail": str(exc)})

    ms = (time.perf_counter() - debut) * 1000

    if request.method != "OPTIONS":
        cle = f"{request.method} {chemin_normalise(chemin)}"
        LATENCES[cle].append(ms)
        STATUTS[f"{reponse.status_code // 100}xx"] += 1

        if reponse.status_code >= 500:
            ERREURS.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "chemin": chemin,
                "statut": reponse.status_code,
            })

        if chemin != "/health":
            evenement = {
                "t": time.time(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "utilisateur": utilisateur, "role": role,
                "methode": request.method, "chemin": chemin,
                "statut": reponse.status_code, "ms": round(ms, 1),
                "ip": request.client.host if request.client else None,
            }

            if question:
                evenement["question"] = question

            auditer(evenement)

    return reponse


# ============================================================
# 5. ROUTES : ACCES ET ADMINISTRATION
# ============================================================

router = APIRouter()


class Connexion(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=200)


@router.post("/auth/login")
def connexion(corps: Connexion, request: Request):
    if not UTILISATEURS:
        raise HTTPException(
            status_code=503,
            detail="Aucun utilisateur configuré (variable AUTH_USERS).",
        )

    cle = (corps.username, request.client.host if request.client else "")
    nb, premier = ECHECS_CONNEXION.get(cle, (0, time.time()))

    if nb >= 5 and time.time() - premier < 300:
        raise HTTPException(status_code=429,
                            detail="Trop d'échecs : réessayez dans 5 minutes.")

    u = UTILISATEURS.get(corps.username)

    if not u or not hmac.compare_digest(
        u["mot_de_passe"].encode(), corps.password.encode()
    ):
        ECHECS_CONNEXION[cle] = (
            (nb + 1, premier) if time.time() - premier < 300
            else (1, time.time())
        )
        raise HTTPException(status_code=401,
                            detail="Identifiants incorrects.")

    ECHECS_CONNEXION.pop(cle, None)

    return {
        "access_token": creer_jeton(corps.username, u["role"]),
        "token_type": "bearer",
        "role": u["role"],
        "expire_dans_minutes": JWT_TTL_MIN,
    }


@router.get("/auth/me")
def moi(request: Request):
    en_tete = request.headers.get("authorization", "")
    jeton = lire_jeton(en_tete[7:]) if en_tete.lower().startswith(
        "bearer "
    ) else None

    if jeton:
        return {"utilisateur": jeton["sub"], "role": jeton["role"],
                "auth_requise": AUTH_REQUIRED}

    return {"utilisateur": None,
            "role": None if AUTH_REQUIRED else AUTH_DEFAULT_ROLE,
            "auth_requise": AUTH_REQUIRED}


@router.get("/admin/utilisateurs")
def lister_utilisateurs():
    return {
        "auth_requise": AUTH_REQUIRED,
        "utilisateurs": [{"nom": n, "role": u["role"]}
                         for n, u in UTILISATEURS.items()],
        "roles": ROLES,
        "droits": {
            "lecteur": "assistant, graphe, sources, qualité, stats",
            "analyste": "lecteur + évaluation, supervision, debug",
            "admin": "analyste + rechargement, audit, configuration",
        },
    }


@router.get("/admin/audit")
def lire_audit(
    limite: int = Query(100, ge=1, le=1000),
    utilisateur: Optional[str] = None,
    chemin: Optional[str] = None,
    statut_min: Optional[int] = Query(None, ge=100, le=599),
):
    evenements = [
        e for e in reversed(AUDIT)
        if (not utilisateur or e.get("utilisateur") == utilisateur)
        and (not chemin or chemin in e.get("chemin", ""))
        and (statut_min is None or e.get("statut", 0) >= statut_min)
    ]

    return {
        "total_filtre": len(evenements),
        "en_memoire": len(AUDIT),
        "retention_jours": AUDIT_RETENTION_JOURS,
        "evenements": evenements[:limite],
    }


@router.post("/admin/audit/purge")
def purge():
    return {"supprimees": purger_audit(),
            "retention_jours": AUDIT_RETENTION_JOURS}


@router.get("/admin/config")
def configuration():
    return {
        "version": ctx.VERSION,
        "neo4j_database": ctx.NEO4J_DATABASE,
        "graph_label": ctx.GRAPH_LABEL,
        "relation_source": ctx.RELATION_SOURCE,
        "modeles_groq": ctx.GROQ_MODELS,
        "min_confidence": ctx.MIN_CONFIDENCE,
        "exclure_etrangers": ctx.EXCLURE_ETRANGERS,
        "auth_requise": AUTH_REQUIRED,
        "role_par_defaut_sans_auth": AUTH_DEFAULT_ROLE,
        "cors_origines": CORS_ORIGINS,
        "audit_retention_jours": AUDIT_RETENTION_JOURS,
        "secrets": "non exposés",
    }


# ============================================================
# 6. SUPERVISION
# ============================================================

def _etat_neo4j():
    debut = time.perf_counter()

    try:
        ctx.get_driver().verify_connectivity()
        return {"ok": True,
                "latence_ms": round((time.perf_counter() - debut) * 1000, 1)}
    except Exception as exc:
        return {"ok": False, "erreur": str(exc)[:200]}


def _graphe():
    with ctx.graphe_lock:
        return ctx.GRAPHE


@router.get("/supervision")
def supervision():
    g = _graphe()
    toutes = [v for d in LATENCES.values() for v in d]
    total = sum(STATUTS.values())

    par_route = [{
        "route": cle, "appels": len(vals),
        "p50_ms": percentile(vals, 50), "p95_ms": percentile(vals, 95),
        "max_ms": round(max(vals), 1),
    } for cle, vals in LATENCES.items() if vals]
    par_route.sort(key=lambda x: -x["appels"])

    return {
        "version": ctx.VERSION,
        "uptime_secondes": int(time.time() - DEMARRAGE),
        "requetes": {
            "total": total, "par_classe": dict(STATUTS),
            "taux_erreur_5xx": round(STATUTS.get("5xx", 0) / total, 4)
            if total else 0.0,
        },
        "latence_ms": {"p50": percentile(toutes, 50),
                       "p95": percentile(toutes, 95)},
        "routes": par_route,
        "neo4j": _etat_neo4j(),
        "groq": {
            "configure": ctx.groq_client is not None,
            "modeles": ctx.GROQ_MODELS,
            "modele_prioritaire": ctx.obtenir_modele_prioritaire(),
        },
        "graphe": {
            "charge": g is not None,
            "charge_le": g.charge_le if g else None,
            "entites": len(g.entites) if g else 0,
            "relations": len(g.relations) if g else 0,
        },
        "dernieres_erreurs": list(ERREURS)[-10:],
        "note": "Métriques en mémoire : remises à zéro au redémarrage.",
    }


# ============================================================
# 7. QUALITE DES DONNEES
# ============================================================

def _pct(x, n):
    return round(100 * x / n, 1) if n else 0.0


@router.get("/qualite")
def qualite():
    g = ctx.obtenir_graphe()
    rels = g.relations
    n = len(rels)

    niveaux = Counter(r["qualite"]["niveau_documentation"] for r in rels)
    confirmees = [r for r in rels if r["statut"] == "CONFIRME"]
    confirmees_sans_extrait = [r for r in confirmees if not r["extraits"]]

    personnes = [e for e in g.entites.values()
                 if e["categorie"] == "PERSONNE"]
    perimetre = Counter(ctx.perimetre_personne(e) for e in personnes)

    groupes = defaultdict(list)
    for e in g.entites.values():
        if not e["qid_brut"]:
            groupes[ctx.normaliser(e["label"])].append(e)

    doublons = sorted(
        ({"label": v[0]["label"], "ids": [x["id"] for x in v],
          "types": sorted({x["type"] for x in v})}
         for v in groupes.values() if len(v) > 1),
        key=lambda d: -len(d["ids"]),
    )

    non_qualifiees = Counter(
        r["type"] for r in rels if r["nature"] == "non_qualifie"
    )
    echantillon = [{
        "source": g.entites[r["source_id"]]["label"],
        "relation": r["type"],
        "cible": g.entites[r["cible_id"]]["label"],
        "type_cible": g.entites[r["cible_id"]]["type"],
    } for r in rels if r["nature"] == "non_qualifie"][:15]

    sans_relation = [e for e in g.entites.values() if g.degre(e["id"]) == 0]
    qid = [e for e in g.entites.values() if e["qid_brut"]]

    return {
        "relations": {
            "total": n,
            "avec_extrait": {"n": sum(1 for r in rels if r["extraits"]),
                             "pct": _pct(sum(1 for r in rels
                                             if r["extraits"]), n)},
            "wikidata_structure_seul": {
                "n": niveaux.get("source_structuree_a_examiner", 0),
                "pct": _pct(niveaux.get("source_structuree_a_examiner", 0),
                            n)},
            "avec_date": {"n": sum(1 for r in rels if r["debut"] or r["fin"]),
                          "pct": _pct(sum(1 for r in rels
                                          if r["debut"] or r["fin"]), n)},
            "confirmees": len(confirmees),
            "a_verifier": n - len(confirmees),
            "par_niveau_documentation": dict(niveaux),
            "par_nature": dict(Counter(r["nature"] for r in rels)),
        },
        "entites": {
            "total": len(g.entites),
            "par_categorie": dict(Counter(e["categorie"]
                                          for e in g.entites.values())),
            "sans_libelle_exploitable": len(qid),
            "sans_relation": len(sans_relation),
        },
        "personnes": {
            "total": len(personnes),
            "perimetre": dict(perimetre),
            "note": "« inconnu » = ni pays ni indice dans la description : "
                    "ne pas présenter comme marocain.",
        },
        "a_traiter": [
            {"constat": "Relations CONFIRME sans extrait textuel",
             "nombre": len(confirmees_sans_extrait),
             "action": "Ajouter un extrait ou repasser en A_VERIFIER."},
            {"constat": "Relations dont la nature n'est pas qualifiée",
             "nombre": sum(non_qualifiees.values()),
             "action": "Typer la cible ou étendre les règles de "
                       "classification."},
            {"constat": "Libellés en doublon (même nom, ids différents)",
             "nombre": len(doublons),
             "action": "Résolution d'entités à revoir."},
            {"constat": "Entités sans libellé (identifiant Wikidata brut)",
             "nombre": len(qid),
             "action": "Résoudre les libellés."},
            {"constat": "Personnes à périmètre inconnu",
             "nombre": perimetre.get("inconnu", 0),
             "action": "Renseigner pays / nationalité."},
        ],
        "doublons_potentiels": doublons[:25],
        "non_qualifiees_par_type": dict(non_qualifiees),
        "echantillon_non_qualifiees": echantillon,
        "relations_ignorees_au_chargement": dict(g.rejets),
        "note": "Indicateurs de couverture, pas de preuve : une confiance "
                "ou un extrait ne valident pas une relation.",
    }


# ============================================================
# 8. SOURCES
# ============================================================

@router.get("/sources")
def sources(
    niveau: Optional[str] = Query(None),
    statut: Optional[str] = Query(None),
    type_relation: Optional[str] = Query(None),
    limite: int = Query(50, ge=1, le=300),
):
    g = ctx.obtenir_graphe()

    domaines = defaultdict(lambda: {"relations": 0, "avec_extrait": 0})
    for r in g.relations:
        for d in {re.sub(r"^https?://(www\.)?", "", u).split("/")[0]
                  for u in r["urls"]}:
            domaines[d]["relations"] += 1
            domaines[d]["avec_extrait"] += 1 if r["extraits"] else 0

    filtrees = [
        r for r in g.relations
        if (not niveau or r["qualite"]["niveau_documentation"] == niveau)
        and (not statut or r["statut"] == statut)
        and (not type_relation or r["type"] == type_relation)
    ]
    filtrees.sort(key=lambda r: (-len(r["extraits"]), -(r["confiance"] or 0)))

    return {
        "niveaux": dict(Counter(r["qualite"]["niveau_documentation"]
                                for r in g.relations)),
        "domaines": sorted(
            [{"domaine": d, **v} for d, v in domaines.items()],
            key=lambda x: -x["relations"])[:50],
        "total_filtre": len(filtrees),
        "relations": [{"idx": r["idx"], **ctx.relation_api(g, r)}
                      for r in filtrees[:limite]],
    }


# ============================================================
# 9. SOUS-GRAPHE
# ============================================================

@router.get("/graphe/voisinage")
def voisinage(
    id: str,
    profondeur: int = Query(1, ge=1, le=2),
    max_noeuds: int = Query(150, ge=10, le=500),
    nature: Optional[str] = None,
    statut: Optional[str] = None,
):
    g = ctx.obtenir_graphe()

    if id not in g.entites:
        raise HTTPException(status_code=404, detail="Entité inconnue.")

    noeuds, aretes = {id: 0}, {}
    frontiere, tronque = [id], False

    for niveau in range(profondeur):
        suivante = []

        for u in frontiere:
            # Ne développe pas les hubs au-delà du centre.
            if niveau > 0 and g.degre(u) > ctx.HUB_DEGRE_MAX:
                continue

            for idx in g.adj.get(u, []):
                r = g.relations[idx]

                if nature and r["nature"] != nature:
                    continue

                if statut and r["statut"] != statut:
                    continue

                v = r["cible_id"] if r["source_id"] == u else r["source_id"]

                if v not in noeuds:
                    if len(noeuds) >= max_noeuds:
                        tronque = True
                        continue

                    noeuds[v] = niveau + 1
                    suivante.append(v)

                aretes[idx] = r

        frontiere = suivante

    return {
        "centre": id,
        "tronque": tronque,
        "noeuds": [{
            "id": i, "label": g.entites[i]["label"],
            "type": g.entites[i]["type"],
            "categorie": g.entites[i]["categorie"],
            "description": g.entites[i]["description"],
            "degre": g.degre(i), "niveau": nv,
        } for i, nv in noeuds.items()],
        "aretes": [{
            "idx": idx, "source": r["source_id"], "cible": r["cible_id"],
            "relation": r["type"], "nature": r["nature"],
            "statut": r["statut"], "confiance": r["confiance"],
            "debut": r["debut"], "fin": r["fin"],
            "nb_preuves": len(r["extraits"]),
            "niveau_documentation": r["qualite"]["niveau_documentation"],
        } for idx, r in aretes.items()
            if r["source_id"] in noeuds and r["cible_id"] in noeuds],
    }


@router.get("/graphe/relation/{idx}")
def detail_relation(idx: int):
    g = ctx.obtenir_graphe()

    if idx < 0 or idx >= len(g.relations):
        raise HTTPException(status_code=404, detail="Relation inconnue.")

    return ctx.relation_api(g, g.relations[idx])


# ============================================================
# 10. EVALUATION : GRAPHRAG vs CYPHER DIRECT
# ============================================================

_F = "($label = '' OR $label IN labels(p))"

CAS_DEFAUT = [
    {
        "id": "pol_eco",
        "question": "Quels responsables politiques marocains ont des "
                    "liens avec des entreprises ?",
        "colonne": "Personne",
        "cypher": f"""
MATCH (p:Entite)-[:MEMBRE_DE|OCCUPE_FONCTION|PARTICIPE_AU_GOUVERNEMENT|
                  COALITION_AVEC]->(:Entite)
WHERE {_F}
MATCH (p)-[:ACTIONNAIRE_DE|DIRIGE|PRESIDE|FONDATEUR_DE|
            ADMINISTRATEUR_DE]->(e:Entite)
WHERE toLower(coalesce(e.type,'')) CONTAINS 'entreprise'
   OR toLower(coalesce(e.type,'')) CONTAINS 'societe'
RETURN DISTINCT coalesce(p.name, p.label) AS nom""",
        "note": "Référence naïve : sans périmètre marocain ni "
                "qualification des cibles.",
    },
    {
        "id": "actionnaires",
        "question": "Quels responsables politiques sont actionnaires "
                    "d'entreprises ?",
        "colonne": "Personne",
        "cypher": f"""
MATCH (p:Entite)-[:ACTIONNAIRE_DE]->(e:Entite)
WHERE {_F}
  AND (toLower(coalesce(e.type,'')) CONTAINS 'entreprise'
       OR toLower(coalesce(e.type,'')) CONTAINS 'societe')
RETURN DISTINCT coalesce(p.name, p.label) AS nom""",
        "note": "Référence naïve : sans critère politique.",
    },
    {
        "id": "partis",
        "question": "Quels sont les partis politiques présents dans le "
                    "graphe ?",
        "colonne": "Parti",
        "cypher": """
MATCH (p:Entite)
WHERE ($label = '' OR $label IN labels(p))
  AND toLower(coalesce(p.type,'')) CONTAINS 'parti'
RETURN DISTINCT coalesce(p.name, p.label) AS nom""",
        "note": "Question simple : Cypher direct suffit.",
    },
    {
        "id": "dirigeants_mt",
        "question": "Qui dirige Maroc Telecom ?",
        "colonne": "Entité liée",
        "cypher": """
MATCH (p:Entite)-[:DIRIGE|PRESIDE]->(e:Entite)
WHERE ($label = '' OR $label IN labels(p))
  AND toLower(coalesce(e.name, e.label, '')) CONTAINS 'maroc telecom'
RETURN DISTINCT coalesce(p.name, p.label) AS nom""",
        "note": "Question à entité nommée.",
    },
]


class DemandeEvaluation(BaseModel):
    ids: Optional[list[str]] = None
    cas: Optional[list[dict]] = Field(
        None, description="Cas personnalisés : id, question, cypher, "
                          "colonne, attendu (liste de noms, optionnel).")
    avec_llm: bool = False


def _nettoyer(nom):
    return ctx.normaliser(re.sub(r"\s*\([^)]*\)\s*$", "", str(nom)))


def _ensemble_graphrag(resultat, colonne):
    for t in resultat["tableaux"]:
        if colonne in t["colonnes"]:
            return {_nettoyer(l.get(colonne)) for l in t["lignes"]
                    if l.get(colonne) not in (None, "", "—")}

    return set()


def _prf(prediction, reference):
    tp = len(prediction & reference)
    precision = tp / len(prediction) if prediction else (
        1.0 if not reference else 0.0)
    rappel = tp / len(reference) if reference else 1.0
    f1 = (2 * precision * rappel / (precision + rappel)
          if precision + rappel else 0.0)
    return {"precision": round(precision, 3), "rappel": round(rappel, 3),
            "f1": round(f1, 3)}


def _evaluer_cas(cas, avec_llm):
    sortie = {"id": cas.get("id"), "question": cas.get("question"),
              "note": cas.get("note")}

    debut = time.perf_counter()
    try:
        resultat = ctx.construire_resultat(cas["question"], limite=500)
        sortie["graphrag_ms"] = round((time.perf_counter() - debut) * 1000, 1)
        sortie["intention"] = resultat["intention"]
        gr = _ensemble_graphrag(resultat, cas.get("colonne", "Personne"))
    except Exception as exc:
        sortie["graphrag_erreur"] = str(exc)[:200]
        resultat, gr = None, set()

    if avec_llm and resultat is not None:
        debut = time.perf_counter()
        _, erreurs = ctx.composer_reponse(cas["question"], resultat, 20)
        sortie["graphrag_llm_ms"] = round(
            (time.perf_counter() - debut) * 1000, 1)
        sortie["erreurs_llm"] = erreurs

    cy = None
    if cas.get("cypher"):
        debut = time.perf_counter()
        try:
            lignes = ctx.executer(cas["cypher"],
                                  {"label": ctx.GRAPH_LABEL})
            sortie["cypher_ms"] = round(
                (time.perf_counter() - debut) * 1000, 1)
            cy = {_nettoyer(l.get("nom")) for l in lignes if l.get("nom")}
        except Exception as exc:
            sortie["cypher_erreur"] = str(getattr(exc, "detail", exc))[:200]

    sortie["tailles"] = {"graphrag": len(gr),
                         "cypher": None if cy is None else len(cy)}

    if cy is not None:
        sortie["graphrag_vs_cypher"] = _prf(gr, cy)
        sortie["seulement_graphrag"] = sorted(gr - cy)[:30]
        sortie["seulement_cypher"] = sorted(cy - gr)[:30]

    if cas.get("attendu"):
        attendu = {ctx.normaliser(x) for x in cas["attendu"]}
        sortie["graphrag_vs_attendu"] = _prf(gr, attendu)

        if cy is not None:
            sortie["cypher_vs_attendu"] = _prf(cy, attendu)

    return sortie


@router.get("/evaluation/cas")
def cas_evaluation():
    return {"cas": CAS_DEFAUT,
            "note": "Cypher est une référence de comparaison, pas une "
                    "vérité terrain. Ajoutez `attendu` pour mesurer contre "
                    "une vérité terrain vérifiée à la main."}


@router.post("/evaluation/run")
def lancer_evaluation(demande: DemandeEvaluation):
    cas = demande.cas or [
        c for c in CAS_DEFAUT if not demande.ids or c["id"] in demande.ids
    ]

    for c in cas:
        if "question" not in c:
            raise HTTPException(status_code=422,
                                detail="Chaque cas exige `question`.")

    resultats = [_evaluer_cas(c, demande.avec_llm) for c in cas]

    def moyenne(cle, sous):
        vals = [r[cle][sous] for r in resultats if cle in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    def moy_ms(cle):
        vals = [r[cle] for r in resultats if cle in r]
        return round(sum(vals) / len(vals), 1) if vals else None

    run = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avec_llm": demande.avec_llm,
        "synthese": {
            "cas": len(resultats),
            "f1_moyen_graphrag_vs_cypher": moyenne("graphrag_vs_cypher",
                                                   "f1"),
            "latence_graphrag_ms": moy_ms("graphrag_ms"),
            "latence_cypher_ms": moy_ms("cypher_ms"),
            "latence_graphrag_llm_ms": moy_ms("graphrag_llm_ms"),
        },
        "resultats": resultats,
    }

    try:
        with EVAL_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Historique d'évaluation non enregistré.")

    return run


@router.get("/evaluation/historique")
def historique_evaluation(limite: int = Query(20, ge=1, le=100)):
    if not EVAL_FILE.exists():
        return {"runs": []}

    with EVAL_FILE.open(encoding="utf-8") as f:
        lignes = list(deque(f, maxlen=limite))

    runs = []
    for l in lignes:
        try:
            r = json.loads(l)
            runs.append({"ts": r["ts"], "avec_llm": r["avec_llm"],
                         "synthese": r["synthese"]})
        except (json.JSONDecodeError, KeyError):
            continue

    return {"runs": list(reversed(runs))}


# ============================================================
# 11. ARCHITECTURE
# ============================================================

@router.get("/architecture")
def architecture():
    g = _graphe()

    return {
        "principe": "Un seul graphe de connaissances, deux approches : "
                    "analytique directe (Cypher) et GraphRAG (récupération "
                    "+ LLM).",
        "etapes": [
            {"n": 1, "nom": "Crawler", "role": "Collecte, parsing, "
             "métadonnées", "supervise_ici": False},
            {"n": 2, "nom": "HDFS", "role": "Documents bruts",
             "supervise_ici": False},
            {"n": 3, "nom": "ETL + LLM", "role": "Détection d'entités, "
             "extraction de relations, résolution", "supervise_ici": False},
            {"n": 4, "nom": "Spark ETL", "role": "Validation, "
             "déduplication, transformation", "supervise_ici": False},
            {"n": 5, "nom": "Neo4j", "role": "Graphe de connaissances",
             "supervise_ici": True, "etat": _etat_neo4j()},
            {"n": 6, "nom": "Analytique graphe", "role": "Cypher, "
             "classements, chemins", "supervise_ici": True},
            {"n": 7, "nom": "GraphRAG", "role": "Récupération déterministe "
             "+ synthèse LLM (Groq)", "supervise_ici": True,
             "etat": {"llm_configure": ctx.groq_client is not None,
                      "modeles": ctx.GROQ_MODELS}},
            {"n": 8, "nom": "Évaluation", "role": "Exactitude, rappel, "
             "latence, explicabilité", "supervise_ici": True},
            {"n": 9, "nom": "API + interface", "role": "FastAPI, rôles, "
             "audit, supervision", "supervise_ici": True},
        ],
        "graphe_en_memoire": {"entites": len(g.entites) if g else 0,
                              "relations": len(g.relations) if g else 0},
        "garde_fous": [
            "Sources et statut conservés par relation",
            "A_VERIFIER jamais présenté comme confirmé",
            "Périmètre géographique explicite",
            "Journal d'audit et rétention",
            "Accès par rôles",
        ],
        "note": "Les étapes 1 à 4 s'exécutent en amont de l'API : leur "
                "état n'est pas mesuré ici.",
    }


# ============================================================
# 12. BRANCHEMENT
# ============================================================

def register(app, module):
    """Appelé une fois, avec l'application FastAPI et le module principal."""

    global ctx
    ctx = module

    _charger_audit()
    supprimees = purger_audit()

    app.include_router(router)
    app.middleware("http")(garde)
    # Ajouté après : CORS devient la couche externe (pré-vols OPTIONS).
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS,
        allow_methods=["*"], allow_headers=["*"],
    )

    logger.info(
        "Extension chargée : auth=%s, %s utilisateur(s), audit purgé=%s",
        "ON" if AUTH_REQUIRED else "OFF", len(UTILISATEURS), supprimees,
    )
