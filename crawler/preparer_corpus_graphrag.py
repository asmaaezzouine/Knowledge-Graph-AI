"""
Prépare les deux entrées de la suite du pipeline :

1. corpus_graphrag.json  : documents texte (Wikipédia, presse, institutions)
   -> extraction d'entités/relations par le LLM.
2. wikidata_graph.json   : entités et relations DÉJÀ structurées issues de
   Wikidata (avec QID) -> directement exploitables par Spark / Neo4j.
"""

import json
from pathlib import Path

from config_corpus import (
    SOURCE,
    THEMES,
    charger_document,
    charger_exclusions,
    charger_qids_exclus,
    detecter_themes,
    lister_entites,
    lister_textes,
)

OUTPUT_TEXTES = Path("corpus_graphrag.json")
OUTPUT_WIKIDATA = Path("wikidata_graph.json")

CONFIANCE_WIKIDATA = 0.9


def preparer_textes():
    exclusions = charger_exclusions()
    documents, erreurs = [], []

    for path in lister_textes():
        if path.name in exclusions:
            continue

        try:
            doc = charger_document(path)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            print(f"Document ignoré : {path.name} — {exc}")
            continue

        if not doc["text"]:
            erreurs.append(path.name)
            continue

        doc["themes"] = list(detecter_themes(doc["title"], doc["text"]))
        documents.append(doc)

    OUTPUT_TEXTES.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Documents texte intégrés : {len(documents)}")
    print(f"Documents sans texte : {len(erreurs)}")
    print(f"Corpus enregistré : {OUTPUT_TEXTES.resolve()}")
    print("Thèmes détectés :")

    for theme in THEMES:
        n = sum(theme in d["themes"] for d in documents)
        print(f"- {theme} : {n} documents")


def preparer_wikidata():
    entites, relations = [], []
    exclus_qids = charger_qids_exclus()

    for path in lister_entites():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Entité ignorée : {path.name} — {exc}")
            continue

        if data["qid"] in exclus_qids:
            continue

        entites.append({
            "qid": data["qid"],
            "label": (
                data["labels"].get("fr")
                or data["labels"].get("en")
                or data["labels"].get("ar")
                or ""
            ),
            "labels": data["labels"],
            "descriptions": data["descriptions"],
            "types": data["types"],
            "professions": data["professions"],
            "nationalites": data["nationalites"],
            "annee_naissance": data["annee_naissance"],
            "comptes_officiels": data["comptes_officiels"],
        })

        for rel in data["relations"]:
            if rel["subject"] in exclus_qids or rel["object"] in exclus_qids:
                continue

            relations.append({
                **rel,
                "source": "wikidata",
                "source_url": data["source_url"],
                "confiance": CONFIANCE_WIKIDATA,
            })

    # Dédoublonnage : une même relation apparaît dans la fiche des deux
    # extrémités lorsque les deux sont collectées.
    uniques = {}
    for rel in relations:
        cle = (rel["subject"], rel["relation"], rel["object"],
               rel["debut"], rel["fin"])
        uniques.setdefault(cle, rel)

    relations = list(uniques.values())

    libelles = {e["qid"]: e["label"] for e in entites}
    for rel in relations:
        rel["subject_label"] = libelles.get(rel["subject"])
        rel["object_label"] = libelles.get(rel["object"])

    OUTPUT_WIKIDATA.write_text(
        json.dumps(
            {"entities": entites, "relations": relations},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    sans_libelle = sum(1 for r in relations if not r["object_label"])

    print(f"\nEntités Wikidata : {len(entites)}")
    print(f"Relations (dédoublonnées) : {len(relations)}")
    print(f"Relations dont l'objet n'a pas de libellé collecté : {sans_libelle}")
    print(f"Fichier enregistré : {OUTPUT_WIKIDATA.resolve()}")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dossier introuvable : {SOURCE}")

    preparer_textes()
    preparer_wikidata()


if __name__ == "__main__":
    main()