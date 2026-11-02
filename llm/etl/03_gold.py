
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType
from functools import reduce


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "GraphRAG-Maroc-Gold"

HDFS_BASE = "hdfs://namenode:9000/DataLake"

SILVER_ENTITIES = f"{HDFS_BASE}/silver/delta/entites"
SILVER_RELATIONS = f"{HDFS_BASE}/silver/delta/relations"
SILVER_REJECTED = f"{HDFS_BASE}/silver/delta/relations_rejetees"

GOLD_BASE = f"{HDFS_BASE}/gold/delta"

OUTPUTS = {
    "noeuds": f"{GOLD_BASE}/noeuds",
    "liens": f"{GOLD_BASE}/liens",
    "personnes": f"{GOLD_BASE}/personnes",
    "personnes_politiques": f"{GOLD_BASE}/personnes_politiques",
    "parcours_politiques": f"{GOLD_BASE}/parcours_politiques",
    "roles_entreprises": f"{GOLD_BASE}/roles_entreprises",
    "evidences": f"{GOLD_BASE}/evidences",
    "relations_a_resoudre": f"{GOLD_BASE}/relations_a_resoudre",
}


# ============================================================
# 2. VOCABULAIRES DE RELATIONS
# ============================================================

# Relations pouvant décrire un parcours politique ou professionnel.
CAREER_TYPES = [
    "occupe_fonction",
    "occupe_le_poste_de",
    "est_ministre_de",
    "ministre_de",
    "president_de",
    "presidente_de",
    "directeur_de",
    "directrice_de",
    "directeur_general_de",
    "membre_de",
    "adhere_a",
    "fondateur_de",
    "fondatrice_de",
    "candidat_a",
    "candidate_a",
    "elu_a",
    "elue_a",
    "nomme_a",
    "nommee_a",
    "nomme",
    "travaille_pour",
    "responsable_de",
    "chef_de",
    "secretaire_general_de",
    "secretaire_d_etat",
    "depute_de",
    "senateur_de",
    "administrateur_de",
    "actionnaire_de",
    "associe_de",
    "membre_du_conseil_de",
]

# Relations pouvant correspondre à un rôle dans une entreprise.
BUSINESS_ROLE_TYPES = [
    "fondateur_de",
    "fondatrice_de",
    "president_de",
    "presidente_de",
    "directeur_de",
    "directrice_de",
    "directeur_general_de",
    "administrateur_de",
    "actionnaire_de",
    "associe_de",
    "membre_du_conseil_de",
    "travaille_pour",
    "responsable_de",
]

POLITICAL_TYPES = [
    "occupe_fonction",
    "est_ministre_de",
    "ministre_de",
    "president_de",
    "presidente_de",
    "candidat_a",
    "candidate_a",
    "elu_a",
    "elue_a",
    "nomme_a",
    "nommee_a",
    "depute_de",
    "senateur_de",
    "secretaire_general_de",
    "membre_de",
]


# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def has_col(df, name):
    return name in df.columns


def safe_col(df, name, default=None):
    """
    Retourne une colonne si elle existe.
    Sinon, retourne une colonne littérale par défaut.
    """
    if name in df.columns:
        return F.col(name)
    return F.lit(default)


def normalize_relation_type(column):
    """
    Normalisation technique du type de relation.
    Ne modifie pas le sens de la relation.
    """
    return F.regexp_replace(
        F.lower(F.trim(F.coalesce(column, F.lit("")))),
        r"[\s\-]+",
        "_"
    )


def write_delta(df, path):
    """
    Reconstruction idempotente de chaque table Gold.
    Silver reste inchangée.
    """
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(path)
    )


def print_count(name, df):
    print(f"{name}: {df.count()}")


# ============================================================
# 4. INITIALISATION SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName(APP_NAME)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n==========================================")
print("       GRAPH RAG MAROC - GOLD")
print("==========================================\n")


# ============================================================
# 5. LECTURE DES TABLES SILVER
# ============================================================

print("[1/9] Lecture des tables Silver...")

entities = (
    spark.read
    .format("delta")
    .load(SILVER_ENTITIES)
)

relations = (
    spark.read
    .format("delta")
    .load(SILVER_RELATIONS)
)

rejected = (
    spark.read
    .format("delta")
    .load(SILVER_REJECTED)
)

print_count("Entités Silver", entities)
print_count("Relations Silver", relations)
print_count("Relations rejetées", rejected)


# ============================================================
# 6. CONSTRUCTION DES NOEUDS
# ============================================================

print("\n[2/9] Construction des noeuds...")

# On conserve toutes les colonnes originales de Silver.
# Les colonnes supplémentaires facilitent l'exploitation Neo4j.

nodes = (
    entities
    .withColumn(
        "display_name",
        F.coalesce(
            F.col("name"),
            F.col("normalized_name"),
            F.col("id")
        )
    )
    .withColumn(
        "node_type",
        F.lower(
            F.trim(
                F.coalesce(F.col("type"), F.lit("inconnu"))
            )
        )
    )
    .withColumn(
        "is_person",
        F.col("node_type").rlike(
            r"personne|person|individu|individual"
        )
    )
    .withColumn(
        "is_company",
        F.col("node_type").rlike(
            r"entreprise|societe|société|company|business"
        )
    )
    .withColumn(
        "gold_processed_at",
        F.current_timestamp()
    )
)

# Écarter les lignes sans identifiant :
# un noeud Neo4j doit avoir un identifiant stable.
nodes = nodes.filter(
    F.col("id").isNotNull() &
    (F.trim(F.col("id")) != "")
)

# Une ligne par ID.
nodes = nodes.dropDuplicates(["id"])

# Création des vues spécialisées.
persons = nodes.filter(F.col("is_person"))

companies = nodes.filter(F.col("is_company"))

print_count("Noeuds Gold", nodes)
print_count("Personnes détectées par type", persons)
print_count("Entreprises détectées par type", companies)


# ============================================================
# 7. CONSTRUCTION DES LIENS DU GRAPHE
# ============================================================

print("\n[3/9] Construction des liens...")

# Enrichissement des relations avec les informations des noeuds.
source_lookup = nodes.select(
    F.col("id").alias("source_lookup_id"),
    F.col("display_name").alias("source_display_name"),
    F.col("node_type").alias("source_node_type"),
    F.col("is_person").alias("source_is_person")
)

target_lookup = nodes.select(
    F.col("id").alias("target_lookup_id"),
    F.col("display_name").alias("target_display_name"),
    F.col("node_type").alias("target_node_type"),
    F.col("is_person").alias("target_is_person"),
    F.col("is_company").alias("target_is_company")
)

links = (
    relations.alias("r")
    .join(
        source_lookup.alias("s"),
        F.col("r.source_entity_id") == F.col("s.source_lookup_id"),
        "inner"
    )
    .join(
        target_lookup.alias("t"),
        F.col("r.target_entity_id") == F.col("t.target_lookup_id"),
        "inner"
    )
    .select(
        F.col("r.id").alias("id"),
        F.col("r.source_entity_id").alias("source_id"),
        F.col("r.target_entity_id").alias("target_id"),

        F.coalesce(
            F.col("s.source_display_name"),
            F.col("r.source_name")
        ).alias("source_name"),

        F.coalesce(
            F.col("t.target_display_name"),
            F.col("r.target_name")
        ).alias("target_name"),

        F.col("s.source_node_type").alias("source_node_type"),
        F.col("t.target_node_type").alias("target_node_type"),

        normalize_relation_type(
            F.col("r.type")
        ).alias("relation_type"),

        F.col("r.description").alias("description"),
        F.col("r.certainty").alias("certainty"),
        F.col("r.start_date").alias("start_date"),
        F.col("r.end_date").alias("end_date"),
        F.col("r.status").alias("status"),
        F.col("r.resolution_status").alias("resolution_status"),

        F.col("r.evidences").alias("evidences"),
        F.col("r.evidence_count").alias("evidence_count"),
        F.col("r.date_coherent").alias("date_coherent"),

        F.col("r.source_file").alias("source_file"),
        F.col("r.ingested_at").alias("ingested_at"),

        F.col("s.source_is_person").alias("source_is_person"),
        F.col("t.target_is_person").alias("target_is_person"),
        F.col("t.target_is_company").alias("target_is_company"),

        F.current_timestamp().alias("gold_processed_at")
    )
)

# Sécurité : pas de lien sans source, cible ou type.
links = links.filter(
    F.col("source_id").isNotNull() &
    F.col("target_id").isNotNull() &
    F.col("relation_type").isNotNull() &
    (F.col("relation_type") != "")
)

links = links.dropDuplicates(["id"])

print_count("Liens Gold", links)


# ============================================================
# 8. PERSONNES POLITIQUES
# ============================================================

print("\n[4/9] Identification des personnes politiques...")

political_links = links.filter(
    F.col("relation_type").isin(POLITICAL_TYPES)
)

# Une personne est retenue ici si :
# 1. son type d'entité indique qu'il s'agit d'une personne ;
# 2. elle intervient dans une relation du vocabulaire politique.
#
# Cette règle est une sélection technique, pas une preuve
# indépendante de son statut politique.

political_person_ids = (
    political_links
    .filter(F.col("source_is_person") == True)
    .select(F.col("source_id").alias("person_id"))
    .union(
        political_links
        .filter(F.col("target_is_person") == True)
        .select(F.col("target_id").alias("person_id"))
    )
    .distinct()
)

political_persons = (
    persons
    .join(
        political_person_ids,
        persons.id == political_person_ids.person_id,
        "inner"
    )
    .drop("person_id")
    .withColumn("political_network_member", F.lit(True))
)

print_count("Personnes politiques détectées par les relations", political_persons)


# ============================================================
# 9. PARCOURS POLITIQUES ET PROFESSIONNELS
# ============================================================

print("\n[5/9] Construction des parcours politiques...")

career = (
    links
    .filter(F.col("relation_type").isin(CAREER_TYPES))
    .filter(F.col("source_is_person") == True)
    .withColumn(
        "career_category",
        F.when(
            F.col("relation_type").isin(POLITICAL_TYPES),
            F.lit("politique")
        ).otherwise(F.lit("professionnel"))
    )
    .select(
        "id",
        "source_id",
        "source_name",
        "source_node_type",
        "target_id",
        "target_name",
        "target_node_type",
        "relation_type",
        "career_category",
        "description",
        "start_date",
        "end_date",
        "status",
        "certainty",
        "evidence_count",
        "evidences",
        "source_file",
        "ingested_at",
        "gold_processed_at"
    )
)

print_count("Étapes de parcours", career)


# ============================================================
# 10. ROLES DES PERSONNES DANS LES ENTREPRISES
# ============================================================

print("\n[6/9] Construction des rôles dans les entreprises...")

business_roles = (
    links
    .filter(
        F.col("relation_type").isin(BUSINESS_ROLE_TYPES)
    )
    .filter(F.col("source_is_person") == True)
    .filter(F.col("target_is_company") == True)
    .select(
        "id",
        "source_id",
        "source_name",
        "target_id",
        "target_name",
        "relation_type",
        "description",
        "start_date",
        "end_date",
        "status",
        "certainty",
        "evidence_count",
        "evidences",
        "source_file",
        "ingested_at",
        "gold_processed_at"
    )
)

print_count("Rôles personne-entreprise détectés", business_roles)


# ============================================================
# 11. EXTRACTION DES PREUVES
# ============================================================

print("\n[7/9] Extraction des preuves documentaires...")

# Les preuves sont conservées séparément pour faciliter :
# - la traçabilité des relations ;
# - l'affichage des sources dans une interface ;
# - la vérification humaine ;
# - les citations dans les réponses GraphRAG.

evidence_schema = relations.schema["evidences"].dataType

if isinstance(evidence_schema, ArrayType):
    evidence_fields = (
        evidence_schema.elementType.fieldNames()
        if isinstance(evidence_schema.elementType, StructType)
        else []
    )
else:
    evidence_fields = []

if "evidences" in relations.columns and evidence_fields:
    evidence_base = (
        links
        .select(
            F.col("id").alias("relation_id"),
            "source_id",
            "target_id",
            "relation_type",
            F.explode_outer("evidences").alias("evidence")
        )
        .filter(F.col("evidence").isNotNull())
    )

    evidence_columns = [
        F.col("relation_id"),
        F.col("source_id"),
        F.col("target_id"),
        F.col("relation_type")
    ]

    for field in evidence_fields:
        evidence_columns.append(
            F.col(f"evidence.{field}").alias(field)
        )

    evidence_df = evidence_base.select(*evidence_columns)

else:
    # Table vide avec un schéma minimal si aucune preuve n'est disponible.
    evidence_df = (
        links
        .select(
            F.col("id").alias("relation_id"),
            "source_id",
            "target_id",
            "relation_type"
        )
        .filter(F.lit(False))
    )

print_count("Preuves documentaires", evidence_df)


# ============================================================
# 12. RELATIONS À RÉSOUDRE
# ============================================================

print("\n[8/9] Conservation des relations rejetées...")

# Les relations rejetées ne sont pas supprimées.
# Elles restent disponibles pour correction ou résolution manuelle.

unresolved = (
    rejected
    .withColumn(
        "gold_resolution_status",
        F.lit("a_verifier")
    )
    .withColumn(
        "gold_processed_at",
        F.current_timestamp()
    )
)

print_count("Relations à résoudre", unresolved)


# ============================================================
# 13. ÉCRITURE DES TABLES GOLD
# ============================================================

print("\n[9/9] Écriture des tables Gold Delta...")

gold_tables = {
    "noeuds": nodes,
    "liens": links,
    "personnes": persons,
    "personnes_politiques": political_persons,
    "parcours_politiques": career,
    "roles_entreprises": business_roles,
    "evidences": evidence_df,
    "relations_a_resoudre": unresolved
}

for table_name, dataframe in gold_tables.items():
    print(f"\nÉcriture de {table_name}...")
    write_delta(dataframe, OUTPUTS[table_name])
    print(f"OK : {OUTPUTS[table_name]}")


# ============================================================
# 14. CONTRÔLES FINAUX
# ============================================================

print("\n==========================================")
print("             BILAN GOLD")
print("==========================================")

for table_name, dataframe in gold_tables.items():
    print_count(table_name, dataframe)

print("\n==========================================")
print("       EXEMPLES DE RELATIONS GOLD")
print("==========================================")

(
    links
    .select(
        "source_name",
        "relation_type",
        "target_name",
        "start_date",
        "end_date"
    )
    .show(20, truncate=False)
)

print("\n==========================================")
print("       EXEMPLES DE PARCOURS")
print("==========================================")

(
    career
    .select(
        "source_name",
        "relation_type",
        "target_name",
        "start_date",
        "end_date"
    )
    .show(20, truncate=False)
)

print("\n==========================================")
print("          GOLD TERMINE AVEC SUCCES")
print("==========================================")

spark.stop()
