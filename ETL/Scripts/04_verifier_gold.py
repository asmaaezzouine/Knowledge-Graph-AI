"""
04 - Vérification de la couche Gold : schéma, contenu et volumétrie de
chaque table d'analyse.
"""

import os

from pyspark.sql import SparkSession

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

builder = SparkSession.builder.appName("VerificationGold")
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

base = f"{LAKE_BASE}/gold/delta"

tables = {
    "Entités par type": f"{base}/entites_par_type",
    "Relations par type": f"{base}/relations_par_type",
    "Couverture des documents": f"{base}/couverture_documents",
    "Personnes : politique ET affaires": f"{base}/personnes_politique_affaires",
    "Liens indirects via une entreprise": f"{base}/liens_indirects_entreprise",
    "Degré des noeuds": f"{base}/degre_noeuds",
    "Distribution de la confiance": f"{base}/qualite_relations",
    "Rejets par motif": f"{base}/rejets_par_motif",
}

try:
    for nom, chemin in tables.items():
        print(f"\n===== {nom} =====")
        df = spark.read.format(LAKE_FORMAT).load(chemin)
        df.printSchema()
        df.show(50, truncate=False)
        print(f"Nombre de lignes : {df.count()}")

finally:
    spark.stop()
