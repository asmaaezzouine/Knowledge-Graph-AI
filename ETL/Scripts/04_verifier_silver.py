"""
04 - Vérification de la couche Silver (noeuds, relations, rejets, mapping).
Affiche schéma, volumétrie et quelques exemples pour chaque table.
"""

import os

from pyspark.sql import SparkSession

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

builder = SparkSession.builder.appName("VerifierSilver")
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

try:
    for table in ["noeuds", "relations", "mapping_noms", "rejets"]:
        print(f"\n{'=' * 20} {table.upper()} {'=' * 20}")

        df = spark.read.format(LAKE_FORMAT).load(f"{base}/{table}")

        print("Colonnes :")
        df.printSchema()

        print("Nombre de lignes :", df.count())

        print("Exemples :")
        df.show(5, truncate=False)

finally:
    spark.stop()
