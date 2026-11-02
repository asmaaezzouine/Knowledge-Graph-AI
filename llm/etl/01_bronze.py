
"""
01 - BRONZE : ingestion des extractions LLM dans Delta Lake.

Sources :
  /DataLake/bronze/llm/entites_maroc.json
  /DataLake/bronze/llm/relations_maroc.json

Tables Delta :
  /DataLake/bronze/delta/entites
  /DataLake/bronze/delta/relations

Objectif :
  - Importer uniquement les nouvelles extractions LLM.
  - Préserver les données brutes et les colonnes supplémentaires.
  - Ajouter les données aux tables existantes (append).
  - Conserver la traçabilité.
  - Aucun nettoyage : réservé à Silver.

Spark 3.5.x + Delta Lake 3.1.x
"""

import argparse
import os
import sys

from pyspark.sql import SparkSession, functions as F, types as T


# ============================================================
# CONFIGURATION
# ============================================================

LAKE_BASE = os.getenv(
    "LAKE_BASE",
    "hdfs://namenode:9000/DataLake"
)

LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta").lower()

INPUT_ENTITIES = (
    f"{LAKE_BASE}/bronze/llm/entites_maroc.json"
)

INPUT_RELATIONS = (
    f"{LAKE_BASE}/bronze/llm/relations_maroc.json"
)

OUTPUT_ENTITIES = (
    f"{LAKE_BASE}/bronze/delta/entites"
)

OUTPUT_RELATIONS = (
    f"{LAKE_BASE}/bronze/delta/relations"
)


# Colonnes attendues des entités
COLONNES_ENTITES = {
    "name": T.StringType(),
    "type": T.StringType(),
    "description": T.StringType(),
    "document_id": T.StringType(),
    "source_url": T.StringType(),
    "dans_texte": T.BooleanType(),
}

# Colonnes attendues des relations
COLONNES_RELATIONS = {
    "source": T.StringType(),
    "target": T.StringType(),
    "type": T.StringType(),
    "debut": T.StringType(),
    "fin": T.StringType(),
    "evidence": T.StringType(),
    "validation": T.StringType(),
    "document_id": T.StringType(),
    "source_url": T.StringType(),
    "preuve_dans_texte": T.BooleanType(),
}


# ============================================================
# 1. CRÉATION DE LA SESSION SPARK
# ============================================================

def creer_spark():

    builder = (
        SparkSession.builder
        .appName("GraphRAG-Maroc-Bronze-LLM")
    )

    if LAKE_FORMAT == "delta":

        builder = (
            builder
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension"
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            )
        )

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# 2. VÉRIFICATION DES COLONNES
# ============================================================

def assurer_colonnes(df, colonnes):

    for nom, type_spark in colonnes.items():

        if nom not in df.columns:

            df = df.withColumn(
                nom,
                F.lit(None).cast(type_spark)
            )

        else:

            df = df.withColumn(
                nom,
                F.col(nom).cast(type_spark)
            )

    return df


# ============================================================
# 3. LECTURE DES FICHIERS JSON
# ============================================================

def lire_json(spark, chemin, nom_dataset):

    print(f"\nLecture : {nom_dataset}")
    print(f"Source : {chemin}")

    try:

        df = (
            spark.read
            .option("multiLine", "true")
            .option("mode", "PERMISSIVE")
            .json(chemin)
        )

        if not df.columns:
            raise ValueError("Le fichier JSON ne contient aucune colonne.")

        df = (
            df
            .withColumn(
                "source_file",
                F.input_file_name()
            )
            .withColumn(
                "ingested_at",
                F.current_timestamp()
            )
        )

        return df

    except Exception as e:

        print(f"ERREUR lecture {nom_dataset} : {e}")
        raise


# ============================================================
# 4. PRÉPARATION DES ENTITÉS
# ============================================================

def preparer_entites(spark):

    print("\n========== ENTITÉS LLM ==========")

    df = lire_json(
        spark,
        INPUT_ENTITIES,
        "entites"
    )

    df = assurer_colonnes(
        df,
        COLONNES_ENTITES
    )

    print("\nSchéma des entités :")
    df.printSchema()

    nombre = df.count()

    print(f"Nombre d'entités à ingérer : {nombre}")

    if nombre == 0:
        raise ValueError("Aucune entité trouvée.")

    return df


# ============================================================
# 5. PRÉPARATION DES RELATIONS
# ============================================================

def preparer_relations(spark):

    print("\n========== RELATIONS LLM ==========")

    df = lire_json(
        spark,
        INPUT_RELATIONS,
        "relations"
    )

    # Conserver les contrôles sous forme JSON
    if "controles" in df.columns:

        df = df.withColumn(
            "controles",
            F.to_json(F.col("controles"))
        )

    else:

        df = df.withColumn(
            "controles",
            F.lit(None).cast("string")
        )

    df = assurer_colonnes(
        df,
        COLONNES_RELATIONS
    )

    print("\nSchéma des relations :")
    df.printSchema()

    nombre = df.count()

    print(f"Nombre de relations à ingérer : {nombre}")

    if nombre == 0:
        raise ValueError("Aucune relation trouvée.")

    return df


# ============================================================
# 6. ÉCRITURE DANS DELTA LAKE
# ============================================================

def ecrire_delta(df, chemin, nom_table, mode):

    print(f"\nÉcriture de la table : {nom_table}")
    print(f"Destination : {chemin}")
    print(f"Mode : {mode}")

    writer = (
        df.write
        .format(LAKE_FORMAT)
        .mode(mode)
    )

    if LAKE_FORMAT == "delta":
        writer = writer.option("mergeSchema", "true")

    writer.save(chemin)

    print(f"Table {nom_table} enregistrée avec succès.")


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Ingestion Bronze des extractions LLM GraphRAG Maroc"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplacer les tables existantes (attention : destructif)"
    )

    args = parser.parse_args()

    # Par défaut, on ajoute les données.
    # --overwrite remplace les tables.
    mode = "overwrite" if args.overwrite else "append"

    if LAKE_FORMAT not in ("delta", "parquet"):
        raise ValueError("LAKE_FORMAT doit être delta ou parquet.")

    print("\n" + "=" * 60)
    print("       GRAPHRAG MAROC - BRONZE ETL")
    print("=" * 60)

    print(f"Stockage : {LAKE_BASE}")
    print(f"Format : {LAKE_FORMAT}")
    print(f"Mode d'écriture : {mode}")

    spark = creer_spark()

    try:

        # 1. Lecture et préparation des entités
        entites = preparer_entites(spark)

        # 2. Lecture et préparation des relations
        relations = preparer_relations(spark)

        # 3. Écriture des entités
        ecrire_delta(
            entites,
            OUTPUT_ENTITIES,
            "entites",
            mode
        )

        # 4. Écriture des relations
        ecrire_delta(
            relations,
            OUTPUT_RELATIONS,
            "relations",
            mode
        )

        # 5. Vérification des résultats
        print("\n========== VÉRIFICATION ==========")

        entites_finales = (
            spark.read
            .format(LAKE_FORMAT)
            .load(OUTPUT_ENTITIES)
        )

        relations_finales = (
            spark.read
            .format(LAKE_FORMAT)
            .load(OUTPUT_RELATIONS)
        )

        print(
            "Total entités dans Bronze :",
            entites_finales.count()
        )

        print(
            "Total relations dans Bronze :",
            relations_finales.count()
        )

        print("\nTables Bronze disponibles :")

        print(OUTPUT_ENTITIES)
        print(OUTPUT_RELATIONS)

        print("\nINGESTION BRONZE TERMINÉE AVEC SUCCÈS.")

    except Exception as e:

        print("\nÉCHEC DE L'INGESTION BRONZE.")
        print(str(e))

        raise

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
