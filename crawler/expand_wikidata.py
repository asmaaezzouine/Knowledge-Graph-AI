"""
Expansion Wikidata : collecte les fiches des entités citées dans les relations
déjà extraites (fondateurs, présidents, membres, sociétés liées...), pour
obtenir des personnes avec leurs libellés et leurs propres relations.

Principe : un « saut » par exécution. Relancer le script pour aller plus loin.

Minimisation des données : par défaut, on ne suit PAS les liens familiaux
(conjoint, parents, fratrie) ; option --include-family pour les inclure.

Pré-requis : avoir lancé extract_text.py (les fichiers entity_*.json doivent
exister dans HDFS, dossier processed).
Utilise l'API officielle Wikidata (50 identifiants par requête).
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import requests

from crawler_hdfs import (
    HDFS_JSON,
    HDFS_MANIFEST,
    HDFS_PROCESSED,
    USER_AGENT,
    HdfsClient,
    now_utc,
    sha256_bytes,
)

API = "https://www.wikidata.org/w/api.php"
BATCH = 50
FAMILY_RELATIONS = {"CONJOINT_DE", "PARENT_DE", "FRATRIE_DE"}


def read_entity_files(hdfs):
    """
    Lit TOUS les entity_*.json en une seule commande HDFS.
    (L'ancienne version lançait un « docker exec » par fichier : plusieurs
    minutes de silence pour 130 fiches.)
    """
    try:
        brut = hdfs.docker_exec(
            ["hdfs", "dfs", "-cat", f"{HDFS_PROCESSED}/entity_*.json"]
        ).decode("utf-8", errors="replace")
    except RuntimeError as exc:
        print(f"[ERREUR] Aucune fiche entity_*.json lisible : {exc}")
        print("         Lancez d'abord : python extract_text.py")
        return []

    decodeur = json.JSONDecoder()
    documents, position = [], 0

    while True:
        while position < len(brut) and brut[position].isspace():
            position += 1

        if position >= len(brut):
            break

        try:
            document, position = decodeur.raw_decode(brut, position)
        except json.JSONDecodeError as exc:
            print(f"[AVERTISSEMENT] Lecture interrompue à l'octet {position} : {exc}")
            break

        if isinstance(document, dict) and document.get("qid"):
            documents.append(document)

    return documents


def collect_candidates(hdfs, include_family):
    """Retourne (qid déjà connus, compteur des qid cités dans les relations)."""
    known = set()
    cited = Counter()

    documents = read_entity_files(hdfs)
    print(f"[INFO] Fiches entity_*.json lues : {len(documents)}", flush=True)

    for doc in documents:
        known.add(doc["qid"])

        for rel in doc.get("relations", []):
            if rel["relation"] in FAMILY_RELATIONS and not include_family:
                continue

            cited[rel["subject"]] += 1
            cited[rel["object"]] += 1

    return known, cited


def upload_batch(hdfs, contents):
    """
    Envoie {qid: octets} vers HDFS/json en 3 commandes au total
    (docker cp + hdfs put), au lieu de 3 commandes par fiche.
    """
    if not contents:
        return

    remote = "/tmp/expansion_wikidata"

    with tempfile.TemporaryDirectory() as tmp:
        for qid, data in contents.items():
            (Path(tmp) / f"{qid}.json").write_bytes(data)

        hdfs.docker_exec(["sh", "-c", f"rm -rf {remote}"])

        copie = subprocess.run(
            ["docker", "cp", tmp, f"{hdfs.container}:{remote}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if copie.returncode != 0:
            raise RuntimeError(copie.stderr.decode("utf-8", errors="replace"))

    hdfs.docker_exec([
        "sh", "-c",
        f"hdfs dfs -put -f {remote}/*.json {HDFS_JSON}/ && rm -rf {remote}",
    ])


def fetch_batch(session, ids):
    response = session.get(
        API,
        params={
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": "labels|descriptions|claims",
            "languages": "fr|ar|en",
            "format": "json",
        },
        timeout=60,
    )
    response.raise_for_status()

    return response.json().get("entities", {})


def main():
    parser = argparse.ArgumentParser(
        description="Collecte les fiches Wikidata citées dans les relations."
    )
    parser.add_argument("--max-entities", type=int, default=150,
                        help="Nombre maximal de nouvelles fiches à collecter.")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Délai entre deux requêtes API (secondes).")
    parser.add_argument("--include-family", action="store_true",
                        help="Suivre aussi les liens familiaux (désactivé par défaut).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les identifiants candidats sans rien collecter.")
    args = parser.parse_args()

    hdfs = HdfsClient()

    manifest = hdfs.read_json(HDFS_MANIFEST) or []
    if isinstance(manifest, dict):
        manifest = manifest.get("files", [])

    collected = {
        e["qid"] for e in manifest
        if e.get("file_type") == "json" and e.get("qid")
    }

    print("[INFO] Lecture des fiches HDFS...", flush=True)
    known, cited = collect_candidates(hdfs, args.include_family)
    known |= collected

    candidates = [q for q, _ in cited.most_common() if q not in known]
    selected = candidates[: args.max_entities]

    print(f"[INFO] Fiches déjà collectées : {len(known)}")
    print(f"[INFO] Identifiants cités mais non collectés : {len(candidates)}")
    print(f"[INFO] À collecter maintenant : {len(selected)}")

    if args.dry_run or not selected:
        print("[INFO] Rien à collecter." if not selected else
              f"[DRY-RUN] {', '.join(selected[:30])}...")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    saved, missing, failed = 0, 0, []
    a_envoyer = {}
    nb_lots = (len(selected) + BATCH - 1) // BATCH

    for start in range(0, len(selected), BATCH):
        ids = selected[start:start + BATCH]
        print(f"[API] Lot {start // BATCH + 1}/{nb_lots} : {len(ids)} identifiants", flush=True)

        try:
            entities = fetch_batch(session, ids)
        except requests.RequestException as exc:
            print(f"[ERREUR] {exc}")
            failed.append({"ids": ids, "error": str(exc)})
            continue

        for qid, entity in entities.items():
            if "missing" in entity:
                missing += 1
                continue

            content = json.dumps(
                {"entities": {qid: entity}},
                ensure_ascii=False,
            ).encode("utf-8")

            a_envoyer[qid] = content

            manifest.append({
                "source_url": (
                    f"{API}?action=wbgetentities&ids={qid}&format=json"
                ),
                "hdfs_path": f"{HDFS_JSON}/{qid}.json",
                "file_type": "json",
                "content_type": "application/json",
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
                "depth": "expansion_wikidata",
                "downloaded_at": now_utc(),
                "qid": qid,
            })

            saved += 1

        time.sleep(args.delay)

    print(f"[HDFS] Envoi de {len(a_envoyer)} fiches...", flush=True)

    try:
        upload_batch(hdfs, a_envoyer)
    except RuntimeError as exc:
        print(f"[AVERTISSEMENT] Envoi groupé impossible ({exc}) : envoi fiche par fiche.")

        for numero, (qid, content) in enumerate(a_envoyer.items(), start=1):
            hdfs.put_bytes(content, f"{HDFS_JSON}/{qid}.json")
            print(f"[HDFS] {numero}/{len(a_envoyer)} {qid}", flush=True)

    hdfs.upload_json(manifest, HDFS_MANIFEST)

    print("\n========== BILAN EXPANSION ==========")
    print(f"Nouvelles fiches enregistrées : {saved}")
    print(f"Identifiants introuvables : {missing}")
    print(f"Lots en échec : {len(failed)}")
    print("Étape suivante : python extract_text.py")
    print("=====================================")


if __name__ == "__main__":
    main()