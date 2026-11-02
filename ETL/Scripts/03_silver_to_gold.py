"""
03 - Gold : tables d'analyse prêtes à l'emploi pour le projet
         « liens entre personnalités publiques ».

Entrées : Silver (noeuds, relations, rejets) + Bronze (documents)
Sorties : /DataLake/gold/delta/
  entites_par_type             noeuds par type, rattachés à Wikidata, documents sources
  relations_par_type           relations par type, confiance moyenne, confirmations multiples
  couverture_documents         noeuds et relations issus de chaque page
  personnes_politique_affaires personnes à la fois liées à la politique ET aux affaires
  liens_indirects_entreprise   paires de personnes reliées via une même entreprise
  degre_noeuds                 degré de chaque noeud (hubs du graphe)
  qualite_relations            distribution de la confiance
  rejets_par_motif             ce que l'ETL a écarté, et pourquoi

Variables d'environnement : LAKE_BASE, LAKE_FORMAT (voir 01_bronze_to_delta.py).
"""

import os

from pyspark.sql import SparkSession, functions as F

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

SILVER_NOEUDS = f"{LAKE_BASE}/silver/delta/noeuds"
SILVER_RELATIONS = f"{LAKE_BASE}/silver/delta/relations"
SILVER_REJETS = f"{LAKE_BASE}/silver/delta/rejets"
BRONZE_DOCUMENTS = f"{LAKE_BASE}/bronze/delta/documents"

GOLD = f"{LAKE_BASE}/gold/delta"
GOLD_ENTITES = f"{GOLD}/entites_par_type"
GOLD_RELATIONS = f"{GOLD}/relations_par_type"
GOLD_DOCUMENTS = f"{GOLD}/couverture_documents"
GOLD_POLITIQUE_AFFAIRES = f"{GOLD}/personnes_politique_affaires"
GOLD_LIENS_INDIRECTS = f"{GOLD}/liens_indirects_entreprise"
GOLD_DEGRES = f"{GOLD}/degre_noeuds"
GOLD_QUALITE = f"{GOLD}/qualite_relations"
GOLD_REJETS = f"{GOLD}/rejets_par_motif"

# Relations qui lient une personne au monde des affaires / de la politique.
RELATIONS_AFFAIRES = ["DIRIGE", "PRESIDE", "FONDATEUR_DE", "ACTIONNAIRE_DE",
                      "ADMINISTRATEUR_DE", "EMPLOYE_PAR"]
RELATIONS_POLITIQUES = ["MEMBRE_DE", "OCCUPE_FONCTION", "DIRIGE", "PRESIDE", "FONDATEUR_DE"]


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


def ecrire(df, chemin):
    (
        df.write.format(LAKE_FORMAT)
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(chemin)
    )


def main():
    spark = creer_spark("GraphRAG-Silver-to-Gold")

    try:
        # ============================================================
        # Lecture Silver
        # ============================================================
        print("\n=== Lecture des tables Silver ===")
        noeuds = spark.read.format(LAKE_FORMAT).load(SILVER_NOEUDS)
        relations = spark.read.format(LAKE_FORMAT).load(SILVER_RELATIONS)
        rejets = spark.read.format(LAKE_FORMAT).load(SILVER_REJETS)
        documents = spark.read.format(LAKE_FORMAT).load(BRONZE_DOCUMENTS)

        print(f"Noeuds Silver : {noeuds.count()}")
        print(f"Relations Silver : {relations.count()}")

        horodatage = F.current_timestamp()

        # ============================================================
        # 1. Noeuds par type
        # ============================================================
        print("\n=== Noeuds par type ===")
        docs_par_type = (
            noeuds.select("type", F.explode("documents").alias("document_id"))
            .groupBy("type")
            .agg(F.countDistinct("document_id").alias("nombre_documents"))
        )
        entites_par_type = (
            noeuds.groupBy("type")
            .agg(
                F.count("*").alias("nombre_noeuds"),
                F.sum(F.when(F.col("qid").isNotNull(), 1).otherwise(0))
                .alias("rattaches_wikidata"),
            )
            .join(docs_par_type, "type", "left")
            .withColumn("nombre_documents", F.coalesce("nombre_documents", F.lit(0)))
            .withColumn("gold_created_at", horodatage)
        )
        entites_par_type.orderBy(F.col("nombre_noeuds").desc()).show(truncate=False)

        # ============================================================
        # 2. Relations par type
        # ============================================================
        print("\n=== Relations par type ===")
        relations_par_type = (
            relations.groupBy("rel")
            .agg(
                F.count("*").alias("nombre_relations"),
                F.round(F.avg("confidence"), 3).alias("confiance_moyenne"),
                F.min("confidence").alias("confiance_min"),
                F.sum(F.when(F.size("methodes") > 1, 1).otherwise(0))
                .alias("confirmees_plusieurs_methodes"),
                F.sum(F.when(F.array_contains("methodes", "wikidata"), 1).otherwise(0))
                .alias("avec_wikidata"),
                F.countDistinct("src").alias("nombre_sources"),
                F.countDistinct("tgt").alias("nombre_cibles"),
            )
            .withColumn("gold_created_at", horodatage)
        )
        relations_par_type.orderBy(F.col("nombre_relations").desc()).show(truncate=False)

        # ============================================================
        # 3. Couverture des documents
        # ============================================================
        print("\n=== Couverture des documents ===")
        noeuds_doc = (
            noeuds.select(F.explode("documents").alias("document_id"))
            .groupBy("document_id").agg(F.count("*").alias("nombre_noeuds"))
        )
        relations_doc = (
            relations.select(F.explode("documents").alias("document_id"))
            .groupBy("document_id").agg(F.count("*").alias("nombre_relations"))
        )
        couverture = (
            documents.select("document_id", "titre", "url", "longueur_texte")
            .join(noeuds_doc, "document_id", "left")
            .join(relations_doc, "document_id", "left")
            .withColumn("nombre_noeuds", F.coalesce("nombre_noeuds", F.lit(0)))
            .withColumn("nombre_relations", F.coalesce("nombre_relations", F.lit(0)))
            .withColumn("gold_created_at", horodatage)
        )
        couverture.orderBy("document_id").show(60, truncate=False)

        # ============================================================
        # 4. Personnes liées à la fois à la politique et aux affaires
        # ============================================================
        print("\n=== Personnes : politique ET affaires ===")
        personnes = noeuds.filter(F.col("type") == "PERSONNE").select(
            F.col("id").alias("pid"), F.col("label").alias("personne"), "qid")
        cibles = noeuds.select(
            F.col("id").alias("tid"), F.col("label").alias("cible"),
            F.col("type").alias("type_cible"))

        # Cible « monde des affaires » : une entreprise, ou un noeud encore sans type
        # (ex. fiche Wikidata non résolue) visé par une relation d'actionnariat.
        cible_affaires = (
            (F.col("type_cible") == "ENTREPRISE")
            | ((F.col("type_cible") == "ENTITE")
               & F.col("rel").isin("ACTIONNAIRE_DE", "ADMINISTRATEUR_DE"))
        )

        liens = (
            relations.join(personnes, relations.src == F.col("pid"))
            .join(cibles, relations.tgt == F.col("tid"))
            .withColumn("lien", F.concat_ws(" ", F.col("rel"), F.lit("->"), F.col("cible")))
        )
        politique = (
            liens.filter(F.col("rel").isin(*RELATIONS_POLITIQUES)
                         & F.col("type_cible").isin("PARTI", "INSTITUTION"))
            .groupBy("pid")
            .agg(F.collect_set("lien").alias("liens_politiques"),
                 F.min("confidence").alias("confiance_politique"))
        )
        affaires = (
            liens.filter(F.col("rel").isin(*RELATIONS_AFFAIRES) & cible_affaires)
            .groupBy("pid")
            .agg(F.collect_set("lien").alias("liens_affaires"),
                 F.min("confidence").alias("confiance_affaires"))
        )
        politique_affaires = (
            personnes.join(politique, "pid").join(affaires, "pid")
            .withColumn("nombre_liens_politiques", F.size("liens_politiques"))
            .withColumn("nombre_liens_affaires", F.size("liens_affaires"))
            .withColumn("confiance_min", F.least("confiance_politique", "confiance_affaires"))
            .drop("confiance_politique", "confiance_affaires")
            .withColumn("gold_created_at", horodatage)
        )
        politique_affaires.orderBy(F.col("nombre_liens_affaires").desc()).show(30, truncate=False)

        # ============================================================
        # 5. Liens indirects : deux personnes reliées via une même entreprise
        # ============================================================
        print("\n=== Liens indirects via une entreprise ===")
        aff = (
            relations.filter(F.col("rel").isin(*RELATIONS_AFFAIRES))
            .join(personnes, relations.src == F.col("pid"))
            .join(cibles, relations.tgt == F.col("tid"))
            .filter(cible_affaires)
            .select("pid", "personne", F.col("rel").alias("rel"), "tid", F.col("cible").alias("entreprise"),
                    "confidence")
        )
        a, b = aff.alias("a"), aff.alias("b")
        liens_indirects = (
            a.join(b, (F.col("a.tid") == F.col("b.tid")) & (F.col("a.pid") < F.col("b.pid")))
            .select(
                F.col("a.personne").alias("personne_a"), F.col("a.rel").alias("lien_a"),
                F.col("b.personne").alias("personne_b"), F.col("b.rel").alias("lien_b"),
                F.col("a.entreprise").alias("entreprise"),
                F.least("a.confidence", "b.confidence").alias("confiance_min"))
            .withColumn("gold_created_at", horodatage)
        )
        liens_indirects.show(30, truncate=False)

        # ============================================================
        # 6. Degré des noeuds (hubs)
        # ============================================================
        print("\n=== Degré des noeuds ===")
        sortants = relations.groupBy(F.col("src").alias("id")).agg(F.count("*").alias("degre_sortant"))
        entrants = relations.groupBy(F.col("tgt").alias("id")).agg(F.count("*").alias("degre_entrant"))
        degres = (
            noeuds.select("id", "label", "type")
            .join(sortants, "id", "left").join(entrants, "id", "left")
            .withColumn("degre_sortant", F.coalesce("degre_sortant", F.lit(0)))
            .withColumn("degre_entrant", F.coalesce("degre_entrant", F.lit(0)))
            .withColumn("degre", F.col("degre_sortant") + F.col("degre_entrant"))
            .withColumn("gold_created_at", horodatage)
        )
        degres.orderBy(F.col("degre").desc()).show(20, truncate=False)

        # ============================================================
        # 7. Qualité : distribution de la confiance
        # ============================================================
        print("\n=== Distribution de la confiance ===")
        qualite = (
            relations
            .withColumn(
                "niveau",
                F.when(F.col("confidence") >= 0.8, "forte (>= 0.8)")
                .when(F.col("confidence") >= 0.5, "moyenne (0.5 - 0.8)")
                .otherwise("faible (< 0.5)"))
            .groupBy("rel", "niveau").agg(F.count("*").alias("nombre_relations"))
            .withColumn("gold_created_at", horodatage)
        )
        qualite.orderBy("rel", "niveau").show(60, truncate=False)

        # ============================================================
        # 8. Rejets
        # ============================================================
        rejets_par_motif = (
            rejets.groupBy("categorie", "motif").agg(F.count("*").alias("nombre"))
            .withColumn("gold_created_at", horodatage)
        )
        rejets_par_motif.orderBy(F.col("nombre").desc()).show(truncate=False)

        # ============================================================
        # Écriture
        # ============================================================
        print(f"\n=== Écriture des tables Gold ({LAKE_FORMAT}) ===")
        for df, chemin in (
            (entites_par_type, GOLD_ENTITES),
            (relations_par_type, GOLD_RELATIONS),
            (couverture, GOLD_DOCUMENTS),
            (politique_affaires, GOLD_POLITIQUE_AFFAIRES),
            (liens_indirects, GOLD_LIENS_INDIRECTS),
            (degres, GOLD_DEGRES),
            (qualite, GOLD_QUALITE),
            (rejets_par_motif, GOLD_REJETS),
        ):
            ecrire(df, chemin)
            print(f"  {chemin}")

        print("\nGold créé avec succès.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
