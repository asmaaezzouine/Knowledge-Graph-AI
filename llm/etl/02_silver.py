
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StringType,
    ArrayType,
    StructType
)

# ============================================================
# CONFIGURATION
# ============================================================

HDFS_BASE = "hdfs://namenode:9000/DataLake"

BRONZE_ENTITES = f"{HDFS_BASE}/bronze/delta/entites"
BRONZE_RELATIONS = f"{HDFS_BASE}/bronze/delta/relations"

SILVER_BASE = f"{HDFS_BASE}/silver/delta"

SILVER_ENTITES = f"{SILVER_BASE}/entites"
SILVER_RELATIONS = f"{SILVER_BASE}/relations"
SILVER_RELATIONS_REJETEES = f"{SILVER_BASE}/relations_rejetees"

# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("GraphRAG-Maroc-Silver")
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n==============================================")
print("       PIPELINE SILVER - GRAPHRAG MAROC")
print("==============================================\n")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def normalize_text(column):
    """
    Nettoie les espaces et les chaînes vides.
    Conserve les accents et la casse originale.
    """
    return F.when(
        column.isNull(),
        F.lit(None).cast("string")
    ).otherwise(
        F.when(
            F.trim(column.cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(
                F.regexp_replace(
                    column.cast("string"),
                    r"\s+",
                    " "
                )
            )
        )
    )


def normalize_identifier(column):
    """
    Normalise les identifiants pour faciliter les jointures.
    """
    return F.lower(
        F.trim(column.cast("string"))
    )


def normalize_date(column):
    """
    Convertit les dates en DATE lorsque le format est reconnu.
    Les valeurs non reconnues deviennent NULL.
    """
    value = F.trim(column.cast("string"))

    return F.coalesce(
        F.to_date(value, "yyyy-MM-dd"),
        F.to_date(value, "dd/MM/yyyy"),
        F.to_date(value, "yyyy/MM/dd"),
        F.to_date(value, "dd-MM-yyyy")
    )


def has_column(df, name):
    return name in df.columns


# ============================================================
# 1. LECTURE BRONZE
# ============================================================

print("[1/7] Lecture des tables Bronze...")

entites_bronze = (
    spark.read.format("delta")
    .load(BRONZE_ENTITES)
)

relations_bronze = (
    spark.read.format("delta")
    .load(BRONZE_RELATIONS)
)

print("Entités Bronze :", entites_bronze.count())
print("Relations Bronze :", relations_bronze.count())

print("\nSchéma entités :")
entites_bronze.printSchema()

print("\nSchéma relations :")
relations_bronze.printSchema()


# ============================================================
# 2. NETTOYAGE DES ENTITÉS
# ============================================================

print("\n[2/7] Nettoyage des entités...")

entites = entites_bronze

# Normalisation des colonnes textuelles principales.
for col_name in [
    "id",
    "name",
    "type",
    "description",
    "source",
    "source_url",
    "document_id"
]:
    if has_column(entites, col_name):
        entites = entites.withColumn(
            col_name,
            normalize_text(F.col(col_name))
        )

# Normalisation de l'identifiant.
if has_column(entites, "id"):
    entites = entites.withColumn(
        "id",
        normalize_identifier(F.col("id"))
    )

# Normalisation du type.
if has_column(entites, "type"):
    entites = entites.withColumn(
        "type",
        F.lower(F.trim(F.col("type")))
    )

# Suppression des entités sans identifiant.
entites = entites.filter(
    F.col("id").isNotNull()
    & (F.col("id") != "")
)

# Suppression des doublons sur l'identifiant.
# On conserve la ligne la plus récente si ingested_at existe.
if has_column(entites, "ingested_at"):
    fenetre_entites = (
        Window.partitionBy("id")
        .orderBy(F.col("ingested_at").desc_nulls_last())
    )

    entites = (
        entites
        .withColumn("_rn", F.row_number().over(fenetre_entites))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
else:
    entites = entites.dropDuplicates(["id"])

# Ajout des métadonnées Silver.
entites = (
    entites
    .withColumn("silver_processed_at", F.current_timestamp())
    .withColumn("silver_status", F.lit("VALID"))
)

# Contrôle de qualité.
entites_invalides = entites.filter(
    F.col("id").isNull()
)

entites = entites.filter(
    F.col("id").isNotNull()
)

print("Entités Silver :", entites.count())


# ============================================================
# 3. NETTOYAGE DES RELATIONS
# ============================================================

print("\n[3/7] Nettoyage des relations...")

relations = relations_bronze

# Colonnes réelles de l'extraction LLM.
colonnes_texte_relations = [
    "id",
    "source_entity_id",
    "target_entity_id",
    "source_name",
    "target_name",
    "type",
    "description",
    "certainty",
    "status",
    "resolution_status"
]

for col_name in colonnes_texte_relations:
    if has_column(relations, col_name):
        relations = relations.withColumn(
            col_name,
            normalize_text(F.col(col_name))
        )

# Identifiants de référence.
for col_name in [
    "id",
    "source_entity_id",
    "target_entity_id"
]:
    if has_column(relations, col_name):
        relations = relations.withColumn(
            col_name,
            normalize_identifier(F.col(col_name))
        )

# Normalisation du type de relation.
if has_column(relations, "type"):
    relations = relations.withColumn(
        "type",
        F.lower(
            F.regexp_replace(
                F.trim(F.col("type")),
                r"\s+",
                "_"
            )
        )
    )

# Normalisation des dates.
if has_column(relations, "start_date"):
    relations = relations.withColumn(
        "start_date",
        normalize_date(F.col("start_date"))
    )

if has_column(relations, "end_date"):
    relations = relations.withColumn(
        "end_date",
        normalize_date(F.col("end_date"))
    )

# Suppression des relations sans identifiant.
relations = relations.filter(
    F.col("id").isNotNull()
    & (F.col("id") != "")
)

# Déduplication par identifiant de relation.
if has_column(relations, "ingested_at"):
    fenetre_relations = (
        Window.partitionBy("id")
        .orderBy(F.col("ingested_at").desc_nulls_last())
    )

    relations = (
        relations
        .withColumn("_rn", F.row_number().over(fenetre_relations))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
else:
    relations = relations.dropDuplicates(["id"])

print("Relations après nettoyage :", relations.count())


# ============================================================
# 4. VALIDATION DES RÉFÉRENCES DES RELATIONS
# ============================================================

print("\n[4/7] Validation des références...")

# Référentiel des identifiants d'entités.
ids_entites = (
    entites
    .select(F.col("id").alias("entity_id"))
    .distinct()
)

# Vérification de l'entité source.
relations = (
    relations
    .join(
        ids_entites.withColumnRenamed(
            "entity_id",
            "source_check"
        ),
        relations["source_entity_id"] ==
        F.col("source_check"),
        "left"
    )
)

# Vérification de l'entité cible.
relations = (
    relations
    .join(
        ids_entites.withColumnRenamed(
            "entity_id",
            "target_check"
        ),
        relations["target_entity_id"] ==
        F.col("target_check"),
        "left"
    )
)

# Contrôle des références.
relations = (
    relations
    .withColumn(
        "source_exists",
        F.col("source_check").isNotNull()
    )
    .withColumn(
        "target_exists",
        F.col("target_check").isNotNull()
    )
    .drop("source_check", "target_check")
)

# Contrôle des champs obligatoires.
relations = relations.withColumn(
    "relation_valid",
    F.col("source_exists")
    & F.col("target_exists")
    & F.col("type").isNotNull()
)

# Motif de rejet.
relations = relations.withColumn(
    "rejection_reason",
    F.when(
        ~F.col("source_exists") & ~F.col("target_exists"),
        F.lit("SOURCE_AND_TARGET_NOT_FOUND")
    )
    .when(
        ~F.col("source_exists"),
        F.lit("SOURCE_NOT_FOUND")
    )
    .when(
        ~F.col("target_exists"),
        F.lit("TARGET_NOT_FOUND")
    )
    .when(
        F.col("type").isNull(),
        F.lit("MISSING_RELATION_TYPE")
    )
    .otherwise(F.lit(None).cast("string"))
)

relations_valides = (
    relations
    .filter(F.col("relation_valid") == True)
    .drop("source_exists", "target_exists", "relation_valid", "rejection_reason")
    .withColumn("silver_status", F.lit("VALID"))
    .withColumn("silver_processed_at", F.current_timestamp())
)

relations_rejetees = (
    relations
    .filter(F.col("relation_valid") == False)
    .withColumn("silver_status", F.lit("REJECTED"))
    .withColumn("silver_processed_at", F.current_timestamp())
)

print("Relations valides :", relations_valides.count())
print("Relations rejetées :", relations_rejetees.count())


# ============================================================
# 5. CONTRÔLES SUPPLÉMENTAIRES
# ============================================================

print("\n[5/7] Contrôles supplémentaires...")

# Vérification des dates incohérentes.
if has_column(relations_valides, "start_date") and has_column(relations_valides, "end_date"):

    relations_valides = relations_valides.withColumn(
        "date_coherent",
        F.when(
            F.col("start_date").isNotNull()
            & F.col("end_date").isNotNull()
            & (F.col("start_date") > F.col("end_date")),
            F.lit(False)
        ).otherwise(F.lit(True))
    )

# Les preuves sont conservées dans la colonne evidences.
# On ne supprime pas les relations dont la preuve est vide :
# elles restent disponibles pour un contrôle ultérieur.
if has_column(relations_valides, "evidences"):
    relations_valides = relations_valides.withColumn(
        "evidence_count",
        F.when(
            F.col("evidences").isNull(),
            F.lit(0)
        ).otherwise(F.size(F.col("evidences")))
    )

# Nettoyage des colonnes génériques ajoutées en Bronze,
# uniquement si elles sont entièrement vides.
for col_name in [
    "source",
    "target",
    "debut",
    "fin",
    "evidence",
    "validation",
    "document_id",
    "source_url",
    "preuve_dans_texte"
]:
    if has_column(relations_valides, col_name):
        non_null_count = (
            relations_valides
            .filter(F.col(col_name).isNotNull())
            .limit(1)
            .count()
        )

        if non_null_count == 0:
            relations_valides = relations_valides.drop(col_name)


# ============================================================
# 6. ÉCRITURE DES TABLES SILVER
# ============================================================

print("\n[6/7] Écriture des tables Silver...")

(
    entites
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_ENTITES)
)

(
    relations_valides
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_RELATIONS)
)

(
    relations_rejetees
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_RELATIONS_REJETEES)
)


# ============================================================
# 7. BILAN FINAL
# ============================================================

print("\n[7/7] Vérification des résultats...")

silver_entites = (
    spark.read.format("delta")
    .load(SILVER_ENTITES)
)

silver_relations = (
    spark.read.format("delta")
    .load(SILVER_RELATIONS)
)

silver_rejetees = (
    spark.read.format("delta")
    .load(SILVER_RELATIONS_REJETEES)
)

print("\n==============================================")
print("             BILAN SILVER")
print("==============================================")

print("Entités Silver :", silver_entites.count())
print("Relations Silver valides :", silver_relations.count())
print("Relations rejetées :", silver_rejetees.count())

print("\nSchéma entités Silver :")
silver_entites.printSchema()

print("\nSchéma relations Silver :")
silver_relations.printSchema()

print("\nAperçu des entités :")
silver_entites.show(5, truncate=False)

print("\nAperçu des relations :")
silver_relations.show(5, truncate=False)

print("\n==============================================")
print("       PIPELINE SILVER TERMINÉ")
print("==============================================\n")

spark.stop()
