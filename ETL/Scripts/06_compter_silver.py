"""
06 - Contrôles d'intégrité de la couche Silver avant le chargement Neo4j.

Vérifie : volumétrie, libellés vides, types hors liste, relations pendantes
(source ou cible absente des noeuds), boucles, confiance hors [0, 1],
doublons de relation, noeuds isolés.
Code de sortie 1 si une anomalie bloquante est détectée.
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, trim

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

TYPES_NOEUDS = ["PERSONNE", "PARTI", "ENTREPRISE", "INSTITUTION", "FONCTION", "ENTITE"]
RELATIONS = [
    "MEMBRE_DE", "OCCUPE_FONCTION", "DIRIGE", "PRESIDE", "FONDATEUR_DE",
    "ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "FILIALE_DE",
    "CONJOINT_DE", "PARENT_DE", "FRATRIE_DE",
]

builder = SparkSession.builder.appName("Verifier-Silver-Integrite")
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

base = f"{LAKE_BASE}/silver/delta"
bloquants = []


def controle(nom, valeur, bloquant=True):
    etat = "OK" if valeur == 0 else ("BLOQUANT" if bloquant else "ATTENTION")
    print(f"  [{etat}] {nom} : {valeur}")
    if valeur and bloquant:
        bloquants.append(nom)


try:
    noeuds = spark.read.format(LAKE_FORMAT).load(f"{base}/noeuds")
    relations = spark.read.format(LAKE_FORMAT).load(f"{base}/relations")

    print("\n===== VERIFICATION SILVER =====")
    print(f"Nombre de noeuds : {noeuds.count()}")
    print(f"Identifiants distincts : {noeuds.select('id').distinct().count()}")
    print(f"Nombre de relations : {relations.count()}")

    print("\nControles :")
    controle("noeuds sans identifiant",
             noeuds.filter(col("id").isNull() | (trim(col("id")) == "")).count())
    controle("identifiants de noeud en double",
             noeuds.count() - noeuds.select("id").distinct().count())
    controle("noeuds sans libellé",
             noeuds.filter(col("label").isNull() | (trim(col("label")) == "")).count())
    controle("noeuds au type hors liste",
             noeuds.filter(~col("type").isin(*TYPES_NOEUDS)).count())
    controle("relations au type hors liste",
             relations.filter(~col("rel").isin(*RELATIONS)).count())

    ids = noeuds.select(col("id").alias("nid"))
    controle("relations dont la source n'est pas un noeud",
             relations.join(ids, relations.src == col("nid"), "left_anti").count())
    controle("relations dont la cible n'est pas un noeud",
             relations.join(ids, relations.tgt == col("nid"), "left_anti").count())
    controle("relations boucle (source = cible)",
             relations.filter(col("src") == col("tgt")).count())
    controle("confiance hors de [0, 1]",
             relations.filter((col("confidence") < 0) | (col("confidence") > 1)
                              | col("confidence").isNull()).count())
    controle("relations en double (source, type, cible)",
             relations.count()
             - relations.select("src", "rel", "tgt").distinct().count())

    # Indicateurs non bloquants
    extremites = (relations.select(col("src").alias("nid"))
                  .union(relations.select(col("tgt").alias("nid"))).distinct())
    controle("noeuds isolés (aucune relation)",
             noeuds.join(extremites, noeuds.id == col("nid"), "left_anti").count(),
             bloquant=False)
    controle("noeuds Wikidata sans libellé (à enrichir)",
             noeuds.filter(col("a_enrichir")).count(), bloquant=False)
    controle("relations sans aucune source",
             relations.filter(size("sources") == 0).count(), bloquant=False)

    print("\nNoeuds par type :")
    noeuds.groupBy("type").count().orderBy(col("count").desc()).show(truncate=False)
    print("Relations par type :")
    relations.groupBy("rel").count().orderBy(col("count").desc()).show(truncate=False)
    print("===== FIN VERIFICATION =====\n")

finally:
    spark.stop()

if bloquants:
    print(f"ANOMALIES BLOQUANTES : {', '.join(bloquants)}")
    sys.exit(1)
