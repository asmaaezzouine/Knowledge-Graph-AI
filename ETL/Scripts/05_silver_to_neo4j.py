"""
05 - Chargement du graphe Silver dans Neo4j.

Modèle :
  (:Entite:Personne|Parti|Entreprise|Institution|Fonction|Autre {id, label, type, qid,
       description, alias, documents, annee_naissance, comptes_officiels, a_enrichir})
  (a)-[:MEMBRE_DE|OCCUPE_FONCTION|DIRIGE|PRESIDE|FONDATEUR_DE|ACTIONNAIRE_DE|
        ADMINISTRATEUR_DE|EMPLOYE_PAR|FILIALE_DE {confidence, n_sources, debut, fin,
        evidences, sources, methodes, roles, documents}]->(b)

Le chargement est rejouable : MERGE sur `id`, contraintes d'unicité créées si absentes.

Variables d'environnement :
  NEO4J_PASSWORD   (obligatoire)
  NEO4J_URI        défaut bolt://neo4j-gds:7687
  NEO4J_USER       défaut neo4j
  NEO4J_DATABASE   défaut neo4j
  NEO4J_RESET=1    supprime d'abord tous les noeuds :Entite
  LAKE_BASE, LAKE_FORMAT (voir 01_bronze_to_delta.py)
"""

import os

from pyspark.sql import SparkSession

HDFS_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake") + "/silver/delta"
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-gds:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_RESET = os.getenv("NEO4J_RESET") == "1"

# Listes blanches : le type de noeud et de relation est injecté dans la requête Cypher.
LABELS = {
    "PERSONNE": "Personne", "PARTI": "Parti", "ENTREPRISE": "Entreprise",
    "INSTITUTION": "Institution", "FONCTION": "Fonction", "ENTITE": "Autre",
}
RELATIONS_AUTORISEES = {
    "MEMBRE_DE", "OCCUPE_FONCTION", "DIRIGE", "PRESIDE", "FONDATEUR_DE",
    "ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "FILIALE_DE",
    "CONJOINT_DE", "PARENT_DE", "FRATRIE_DE",
}

TAILLE_LOT = 500


def lire_silver():
    builder = SparkSession.builder.appName("GraphRAG-Silver-To-Neo4j")
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

    try:
        noeuds = [
            row.asDict(recursive=True) for row in
            spark.read.format(LAKE_FORMAT).load(f"{HDFS_BASE}/noeuds").select(
                "id", "label", "type", "qid", "description", "alias",
                "documents", "annee_naissance", "comptes_officiels",
                "a_enrichir").collect()
        ]
        relations = [
            row.asDict(recursive=True) for row in
            spark.read.format(LAKE_FORMAT).load(f"{HDFS_BASE}/relations").select(
                "src", "rel", "tgt", "confidence", "n_sources", "debut", "fin",
                "evidences", "sources", "methodes", "roles", "documents").collect()
        ]
    finally:
        spark.stop()

    return noeuds, relations


def par_lots(liste, taille=TAILLE_LOT):
    for debut in range(0, len(liste), taille):
        yield liste[debut:debut + taille]


def charger(session, noeuds, relations, reset=False):
    """Charge noeuds et relations ; retourne (noeuds, relations) envoyés."""
    session.run(
        "CREATE CONSTRAINT entite_id IF NOT EXISTS "
        "FOR (e:Entite) REQUIRE e.id IS UNIQUE"
    )

    if reset:
        session.run("MATCH (n:Entite) DETACH DELETE n")

    # ---------------- Noeuds, par label ----------------
    par_label = {}
    for n in noeuds:
        if not n.get("id") or not n.get("label"):
            continue
        par_label.setdefault(LABELS.get(n.get("type"), "Autre"), []).append(n)

    nb_noeuds = 0
    for label, lot_complet in par_label.items():
        for lot in par_lots(lot_complet):
            session.run(
                f"""
                UNWIND $rows AS r
                MERGE (e:Entite {{id: r.id}})
                SET e:{label},
                    e.label = r.label,
                    e.type = r.type,
                    e.qid = r.qid,
                    e.description = r.description,
                    e.alias = r.alias,
                    e.documents = r.documents,
                    e.annee_naissance = r.annee_naissance,
                    e.comptes_officiels = r.comptes_officiels,
                    e.a_enrichir = r.a_enrichir
                """,
                rows=lot,
            )
            nb_noeuds += len(lot)

    # ---------------- Relations, par type ----------------
    par_type = {}
    for r in relations:
        if r.get("rel") in RELATIONS_AUTORISEES and r.get("src") and r.get("tgt"):
            par_type.setdefault(r["rel"], []).append(r)

    nb_relations = 0
    for rel, lot_complet in par_type.items():
        for lot in par_lots(lot_complet):
            session.run(
                f"""
                UNWIND $rows AS r
                MATCH (a:Entite {{id: r.src}})
                MATCH (b:Entite {{id: r.tgt}})
                MERGE (a)-[x:{rel}]->(b)
                SET x.confidence = r.confidence,
                    x.n_sources = r.n_sources,
                    x.debut = r.debut,
                    x.fin = r.fin,
                    x.evidences = r.evidences,
                    x.sources = r.sources,
                    x.methodes = r.methodes,
                    x.roles = r.roles,
                    x.documents = r.documents
                """,
                rows=lot,
            )
            nb_relations += len(lot)

    return nb_noeuds, nb_relations


def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "Variable NEO4J_PASSWORD absente. "
            "Configure-la avant de lancer le script."
        )

    from neo4j import GraphDatabase

    noeuds, relations = lire_silver()
    print(f"Noeuds lus depuis Silver : {len(noeuds)}")
    print(f"Relations lues depuis Silver : {len(relations)}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            nb_n, nb_r = charger(session, noeuds, relations, reset=NEO4J_RESET)

            # Contrôle : ce qui est réellement dans la base
            compte_n = session.run("MATCH (n:Entite) RETURN count(n) AS c").single()["c"]
            compte_r = session.run(
                "MATCH (:Entite)-[x]->(:Entite) RETURN count(x) AS c").single()["c"]

        print(f"Noeuds envoyés : {nb_n} | présents dans Neo4j : {compte_n}")
        print(f"Relations envoyées : {nb_r} | présentes dans Neo4j : {compte_r}")
        print("Import GraphRAG terminé.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
