
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = "hdfs://namenode:9000/DataLake/silver/delta"

spark = (
    SparkSession.builder
    .appName("Controle-Silver")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

relations_rejetees = spark.read.format("delta").load(
    f"{BASE}/relations_rejetees"
)

relations = spark.read.format("delta").load(
    f"{BASE}/relations"
)

entites = spark.read.format("delta").load(
    f"{BASE}/entites"
)

print("\n=== MOTIFS DES REJETS ===")
if "rejection_reason" in relations_rejetees.columns:
    relations_rejetees.groupBy("rejection_reason").count().show(
        truncate=False
    )

print("\n=== DETAILS DES 4 RELATIONS REJETEES ===")
relations_rejetees.select(
    "id",
    "source_entity_id",
    "source_name",
    "target_entity_id",
    "target_name",
    "type",
    "rejection_reason"
).show(20, truncate=False)

print("\n=== DATES INCOHERENTES ===")
if "date_coherent" in relations.columns:
    relations.filter(
        F.col("date_coherent") == False
    ).select(
        "id", "source_name", "type", "target_name",
        "start_date", "end_date"
    ).show(30, truncate=False)

print("\n=== CONTROLE DES NOMS D'ENTITES ===")
entites.select(
    "id", "name", "normalized_name"
).show(20, truncate=False)

print("\n=== BILAN ===")
print("Entités :", entites.count())
print("Relations valides :", relations.count())
print("Relations rejetées :", relations_rejetees.count())

spark.stop()
