"""
01 - Bronze : ingestion des sorties brutes dans des tables Delta.

Entrées (HDFS, à déposer avec `hdfs dfs -put`) :
  /DataLake/bronze/llm/entites_maroc.json          extraction LLM (entités)
  /DataLake/bronze/llm/relations_maroc.json        extraction LLM (relations)
  /DataLake/bronze/wikidata/wikidata_graph.json    graphe Wikidata (QID, relations)
  /DataLake/bronze/corpus/corpus_graphrag.json     corpus (titre de chaque page)

Sorties (tables Delta, données telles quelles + traçabilité) :
  /DataLake/bronze/delta/entites
  /DataLake/bronze/delta/relations
  /DataLake/bronze/delta/wikidata_entites
  /DataLake/bronze/delta/wikidata_relations
  /DataLake/bronze/delta/documents

Aucun nettoyage ici : il se fait dans l'étape Silver.
Par défaut une table existante n'est pas écrasée (--overwrite pour la remplacer).

Variables d'environnement (facultatives) :
  LAKE_BASE    racine du lac            (défaut : hdfs://namenode:9000/DataLake)
  LAKE_FORMAT  delta | parquet          (défaut : delta ; parquet = test local)
"""

import argparse
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

INPUT_ENTITIES = f"{LAKE_BASE}/bronze/llm/entites_maroc.json"
INPUT_RELATIONS = f"{LAKE_BASE}/bronze/llm/relations_maroc.json"
INPUT_WIKIDATA = f"{LAKE_BASE}/bronze/wikidata/wikidata_graph.json"
INPUT_CORPUS = f"{LAKE_BASE}/bronze/corpus/corpus_graphrag.json"

OUTPUT_ENTITIES = f"{LAKE_BASE}/bronze/delta/entites"
OUTPUT_RELATIONS = f"{LAKE_BASE}/bronze/delta/relations"
OUTPUT_WD_ENTITIES = f"{LAKE_BASE}/bronze/delta/wikidata_entites"
OUTPUT_WD_RELATIONS = f"{LAKE_BASE}/bronze/delta/wikidata_relations"
OUTPUT_DOCUMENTS = f"{LAKE_BASE}/bronze/delta/documents"

S, B = T.StringType(), T.BooleanType()

COLONNES_ENTITES = {
    "name": S, "type": S, "description": S, "document_id": S,
    "source_url": S, "dans_texte": B,
}
COLONNES_RELATIONS = {
    "source": S, "target": S, "type": S, "debut": S, "fin": S,
    "evidence": S, "validation": S, "document_id": S, "source_url": S,
    "preuve_dans_texte": B,
}


def creer_spark(nom):
    builder = SparkSession.builder.appName(nom)
    if LAKE_FORMAT == "delta":
        builder = (
            builder
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def lire_json_python(spark, chemin, defaut):
    """Petit fichier JSON lu côté driver (local ou HDFS)."""
    contenus = spark.sparkContext.wholeTextFiles(chemin).values().collect()
    if not contenus:
        return defaut
    return json.loads(contenus[0].lstrip("﻿"))


def lire_liste_json(spark, chemin):
    """Liste JSON d'objets -> DataFrame, avec colonnes d'origine conservées."""
    return (
        spark.read.option("multiLine", "true").json(chemin)
        .withColumn("source_file", F.input_file_name())
        .withColumn("ingested_at", F.current_timestamp())
    )


def assurer_colonnes(df, colonnes):
    """Garantit la présence et le type des colonnes attendues."""
    for nom, type_spark in colonnes.items():
        if nom not in df.columns:
            df = df.withColumn(nom, F.lit(None).cast(type_spark))
        else:
            df = df.withColumn(nom, F.col(nom).cast(type_spark))
    return df


def ecrire(df, chemin, mode):
    (
        df.write.format(LAKE_FORMAT)
        .mode(mode)
        .option("overwriteSchema", "true")
        .save(chemin)
    )


def id_document(doc):
    for cle in ("id", "document_id"):
        if doc.get(cle):
            return str(doc[cle])
    if doc.get("file"):
        return Path(str(doc["file"])).stem
    return str(doc.get("source") or doc.get("url") or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="remplace les tables Bronze existantes")
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "errorifexists"

    spark = creer_spark("GraphRAG-Bronze-to-Delta")

    try:
        # ---------------- LLM : entités ----------------
        print("\n=== Lecture des entités LLM depuis HDFS ===")
        entites = assurer_colonnes(
            lire_liste_json(spark, INPUT_ENTITIES), COLONNES_ENTITES)
        entites.printSchema()
        print("Nombre d'entités :", entites.count())

        # ---------------- LLM : relations ----------------
        print("\n=== Lecture des relations LLM depuis HDFS ===")
        relations = lire_liste_json(spark, INPUT_RELATIONS)
        if "controles" in relations.columns:
            # indicateurs de contrôle calculés à l'extraction : conservés en JSON
            relations = relations.withColumn(
                "controles", F.to_json(F.col("controles")))
        else:
            relations = relations.withColumn(
                "controles", F.lit(None).cast("string"))
        relations = assurer_colonnes(relations, COLONNES_RELATIONS)
        relations.printSchema()
        print("Nombre de relations :", relations.count())

        # ---------------- Wikidata ----------------
        print("\n=== Lecture du graphe Wikidata ===")
        wd = lire_json_python(
            spark, INPUT_WIKIDATA, {"entities": [], "relations": []})
        horodatage = F.current_timestamp()

        lignes_wd = []
        for e in wd.get("entities", []):
            labels = e.get("labels") or {}
            descr = e.get("descriptions") or {}
            lignes_wd.append((
                e.get("qid"), e.get("label"),
                labels.get("fr"), labels.get("en"), labels.get("ar"),
                descr.get("fr"), descr.get("en"),
                None if e.get("annee_naissance") is None
                else str(e.get("annee_naissance")),
                json.dumps(e, ensure_ascii=False),
            ))
        wd_entites = (
            spark.createDataFrame(lignes_wd, T.StructType([
                T.StructField(c, S) for c in (
                    "qid", "label", "label_fr", "label_en", "label_ar",
                    "description_fr", "description_en",
                    "annee_naissance", "raw_json")]))
            .withColumn("source_file", F.lit(INPUT_WIKIDATA))
            .withColumn("ingested_at", horodatage)
        )

        lignes_rel = [
            (r.get("subject"), r.get("object"), r.get("relation"),
             None if r.get("debut") is None else str(r["debut"]),
             None if r.get("fin") is None else str(r["fin"]),
             r.get("source_url"),
             None if r.get("confiance") is None else float(r["confiance"]))
            for r in wd.get("relations", [])
        ]
        wd_relations = (
            spark.createDataFrame(lignes_rel, T.StructType(
                [T.StructField(c, S) for c in (
                    "subject", "object", "relation", "debut", "fin",
                    "source_url")]
                + [T.StructField("confiance", T.DoubleType())]))
            .withColumn("source_file", F.lit(INPUT_WIKIDATA))
            .withColumn("ingested_at", horodatage)
        )
        print("Entités Wikidata :", wd_entites.count())
        print("Relations Wikidata :", wd_relations.count())

        # ---------------- Documents (titre = sujet de la page) ----------------
        print("\n=== Lecture du corpus (titres des pages) ===")
        corpus = lire_json_python(spark, INPUT_CORPUS, [])
        if isinstance(corpus, dict):
            corpus = next((corpus[k] for k in
                           ("documents", "data", "results", "articles")
                           if isinstance(corpus.get(k), list)), [])
        lignes_doc = [
            (id_document(d), d.get("title") or d.get("titre"),
             d.get("url") or d.get("source_url"),
             str(len(str(d.get("text") or d.get("texte") or ""))))
            for d in corpus if isinstance(d, dict)
        ]
        documents = (
            spark.createDataFrame(lignes_doc, T.StructType([
                T.StructField(c, S) for c in (
                    "document_id", "titre", "url", "longueur_texte")]))
            .withColumn("longueur_texte", F.col("longueur_texte").cast("int"))
            .withColumn("source_file", F.lit(INPUT_CORPUS))
            .withColumn("ingested_at", horodatage)
        )
        print("Documents :", documents.count())

        # ---------------- Écriture ----------------
        print(f"\n=== Écriture des tables Bronze ({LAKE_FORMAT}, mode {mode}) ===")
        ecrire(entites, OUTPUT_ENTITIES, mode)
        ecrire(relations, OUTPUT_RELATIONS, mode)
        ecrire(wd_entites, OUTPUT_WD_ENTITIES, mode)
        ecrire(wd_relations, OUTPUT_WD_RELATIONS, mode)
        ecrire(documents, OUTPUT_DOCUMENTS, mode)

        print("\nBronze créé avec succès.")
        for nom, chemin in (
            ("Entités", OUTPUT_ENTITIES), ("Relations", OUTPUT_RELATIONS),
            ("Wikidata entités", OUTPUT_WD_ENTITIES),
            ("Wikidata relations", OUTPUT_WD_RELATIONS),
            ("Documents", OUTPUT_DOCUMENTS),
        ):
            print(f"{nom} : {chemin}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
