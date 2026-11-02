#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMPORT GOLD DELTA LAKE -> NEO4J
Projet : GraphRAG Maroc

Sources :
  /DataLake/gold/delta/noeuds
  /DataLake/gold/delta/liens

Principes :
  - Aucun DELETE
  - Aucun DETACH DELETE
  - Fusion des noeuds par Entite.id
  - Fusion des relations par leur id
  - Conservation des proprietes existantes
  - Identification des donnees importees par GraphRAGMaroc
  - Conservation des preuves dans des noeuds Evidence
"""

import os
import sys
import json
import hashlib
import logging

from pathlib import Path
from datetime import date, datetime
from decimal import Decimal

from pyspark.sql import SparkSession
from neo4j import GraphDatabase


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

HDFS_BASE = os.getenv(
    "GOLD_BASE",
    "hdfs://namenode:9000/DataLake/gold/delta"
)

NODES_PATH = HDFS_BASE + "/noeuds"
LINKS_PATH = HDFS_BASE + "/liens"

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://neo4j-gds:7687"
)

NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

SPARK_MASTER = os.getenv(
    "SPARK_MASTER",
    "spark://spark-master:7077"
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))

APP_NAME = "GraphRAG-Maroc-Gold-to-Neo4j"

LOG_DIR = Path("/tmp/graphrag-maroc-import")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "import_neo4j.log",
            encoding="utf-8"
        )
    ]
)

log = logging.getLogger("gold_to_neo4j")


# ==========================================================
# 2. OUTILS DE CONVERSION
# ==========================================================

def json_default(value):

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "asDict"):
        return value.asDict(recursive=True)

    return str(value)


def clean_value(value):

    if value is None:
        return None

    if hasattr(value, "asDict"):
        value = value.asDict(recursive=True)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=json_default
        )

    if isinstance(value, (list, tuple)):

        values = [clean_value(v) for v in value]

        if all(
            v is None or isinstance(v, (str, int, float, bool))
            for v in values
        ):
            return [v for v in values if v is not None]

        return json.dumps(
            value,
            ensure_ascii=False,
            default=json_default
        )

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def row_to_dict(row):
    return row.asDict(recursive=True)


def stable_id(*parts):

    raw = "|".join(
        "" if p is None else str(p)
        for p in parts
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def normalize_rel_type(value):

    text = str(
        value or "RELATION_NON_TYPEE"
    ).strip().upper()

    text = "".join(
        ch if ch.isalnum() else "_"
        for ch in text
    )

    while "__" in text:
        text = text.replace("__", "_")

    text = text.strip("_")

    if not text:
        text = "RELATION_NON_TYPEE"

    if not text[0].isalpha():
        text = "REL_" + text

    return text


def non_null_properties(data):

    result = {}

    for key, value in data.items():

        if value is None:
            continue

        cleaned = clean_value(value)

        if cleaned is not None:
            result[str(key)] = cleaned

    return result


def get_value(row, *names, default=None):

    for name in names:

        if name in row and row[name] is not None:
            return row[name]

    return default


def parse_evidences(value):

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except Exception:
            pass

    return []


# ==========================================================
# 3. INITIALISATION SPARK
# ==========================================================

def create_spark():

    return (
        SparkSession.builder
        .appName(APP_NAME)
        .master(SPARK_MASTER)

        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )

        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )

        .config(
            "spark.sql.shuffle.partitions",
            "8"
        )

        .getOrCreate()
    )


# ==========================================================
# 4. PREPARATION DES NOEUDS
# ==========================================================

NODE_BATCH_QUERY = """

UNWIND $rows AS row

MERGE (n:Entite {id: row.id})

ON CREATE SET n += row.props

SET n:GraphRAGMaroc

SET n.gold_source = 'GraphRAGMaroc'

SET n.gold_imported_at = datetime()

SET n.gold_display_name =
    coalesce(n.gold_display_name, row.display_name)

SET n.gold_node_type =
    coalesce(n.gold_node_type, row.node_type)

SET n.description =
    coalesce(n.description, row.description)

RETURN count(n) AS traite

"""


def prepare_node(row):

    entity_id = get_value(row, "id")

    if entity_id is None:
        return None

    entity_id = str(entity_id).strip()

    if not entity_id:
        return None

    props = non_null_properties(row)

    props["id"] = entity_id

    display_name = get_value(
        row,
        "display_name",
        "name",
        "label"
    )

    node_type = get_value(
        row,
        "node_type",
        "type"
    )

    description = get_value(
        row,
        "description"
    )

    return {
        "id": entity_id,
        "props": props,
        "display_name": clean_value(display_name),
        "node_type": clean_value(node_type),
        "description": clean_value(description)
    }


# ==========================================================
# 5. IMPORT DES NOEUDS
# ==========================================================

def import_nodes(session, dataframe):

    log.info("Debut import des noeuds Gold.")

    batch = []

    count = 0
    skipped = 0

    for row in dataframe.toLocalIterator():

        prepared = prepare_node(
            row_to_dict(row)
        )

        if prepared is None:
            skipped += 1
            continue

        batch.append(prepared)

        if len(batch) >= BATCH_SIZE:

            session.run(
                NODE_BATCH_QUERY,
                rows=batch
            ).consume()

            count += len(batch)

            log.info(
                "Noeuds traites : %s",
                count
            )

            batch = []

    if batch:

        session.run(
            NODE_BATCH_QUERY,
            rows=batch
        ).consume()

        count += len(batch)

    log.info(
        "Import noeuds termine : %s traites, %s ignores.",
        count,
        skipped
    )

    return count, skipped


# ==========================================================
# 6. PREPARATION DES RELATIONS
# ==========================================================

def prepare_link(row):

    source_id = get_value(
        row,
        "source_entity_id",
        "source_id"
    )

    target_id = get_value(
        row,
        "target_entity_id",
        "target_id"
    )

    rel_type = normalize_rel_type(
        get_value(
            row,
            "relation_type",
            "type"
        )
    )

    if source_id is None or target_id is None:
        return None

    source_id = str(source_id)
    target_id = str(target_id)

    rel_id = get_value(row, "id")

    if rel_id is None or not str(rel_id).strip():

        rel_id = "GRM_" + stable_id(
            source_id,
            rel_type,
            target_id,
            get_value(row, "start_date"),
            get_value(row, "end_date")
        )

    props = non_null_properties(row)

    props.update({
        "id": str(rel_id),
        "source": "GraphRAGMaroc",
        "relation_type": rel_type
    })

    return {
        "id": str(rel_id),
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
        "props": props,
        "evidences": parse_evidences(
            get_value(row, "evidences")
        )
    }


# ==========================================================
# 7. CREATION DES RELATIONS
# ==========================================================

def relation_query(rel_type):

    return f"""

    UNWIND $rows AS row

    MATCH (s:Entite {{id: row.source_id}})

    MATCH (t:Entite {{id: row.target_id}})

    MERGE (s)-[r:{rel_type} {{id: row.id}}]->(t)

    ON CREATE SET r += row.props

    SET r.source = 'GraphRAGMaroc'

    SET r.relation_type = '{rel_type}'

    SET r.gold_imported_at = datetime()

    RETURN count(r) AS traite

    """


# ==========================================================
# 8. IMPORT DES PREUVES — VERSION CORRIGEE
# ==========================================================

def import_evidence(
    tx,
    relation_id,
    source_id,
    target_id,
    evidence
):
    """
    Neo4j ne permet pas de créer une relation sortante depuis une relation.
    Le noeud Evidence est donc rattaché aux deux entités de la relation.
    relation_id est conservé sur les liens pour retrouver la relation précise.
    """

    if hasattr(evidence, "asDict"):
        evidence = evidence.asDict(recursive=True)

    evidence_dict = (
        evidence
        if isinstance(evidence, dict)
        else {"text": str(evidence)}
    )

    evidence_props = non_null_properties(evidence_dict)

    evidence_props.update({
        "relation_id": str(relation_id),
        "source_entity_id": str(source_id),
        "target_entity_id": str(target_id),
        "source": "GraphRAGMaroc"
    })

    evidence_key = "GRM_EVID_" + stable_id(
        relation_id,
        json.dumps(
            evidence_dict,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default
        )
    )

    tx.run(
        """
        MATCH (s:Entite {id: $source_id})

        MATCH (s)-[r {id: $relation_id, source: 'GraphRAGMaroc'}]->(t:Entite {id: $target_id})

        MERGE (e:Evidence:GraphRAGMarocEvidence {id: $evidence_id})

        ON CREATE SET e += $props

        SET e.source = 'GraphRAGMaroc'

        MERGE (e)-[:EVIDENCE_FOR {relation_id: $relation_id}]->(s)

        MERGE (e)-[:EVIDENCE_TARGET {relation_id: $relation_id}]->(t)

        RETURN count(e) AS traite
        """,
        source_id=str(source_id),
        target_id=str(target_id),
        relation_id=str(relation_id),
        evidence_id=evidence_key,
        props=evidence_props
    ).consume()


# ==========================================================
# 9. IMPORT DES LIENS ET PREUVES
# ==========================================================

def import_links(session, dataframe):

    log.info("Debut import des liens Gold.")

    grouped = {}

    skipped = 0

    for row in dataframe.toLocalIterator():

        prepared = prepare_link(
            row_to_dict(row)
        )

        if prepared is None:
            skipped += 1
            continue

        grouped.setdefault(
            prepared["rel_type"],
            []
        ).append(prepared)

    imported = 0
    evidence_count = 0

    for rel_type, rows in grouped.items():

        query = relation_query(rel_type)

        for start in range(
            0,
            len(rows),
            BATCH_SIZE
        ):

            batch = rows[
                start:start + BATCH_SIZE
            ]

            session.run(
                query,
                rows=batch
            ).consume()

            imported += len(batch)

            log.info(
                "Relations %s : %s lignes soumises.",
                rel_type,
                len(batch)
            )

            # Import des preuves associees aux relations.
            for item in batch:

                for evidence in item["evidences"]:

                    with session.begin_transaction() as tx:

                        import_evidence(
                            tx,
                            item["id"],
                            item["source_id"],
                            item["target_id"],
                            evidence
                        )

                        tx.commit()

                    evidence_count += 1

    # Verification des extremites manquantes.
    all_rows = [
        item
        for group in grouped.values()
        for item in group
    ]

    missing_result = session.run(
        """
        UNWIND $rows AS row

        OPTIONAL MATCH (s:Entite {id: row.source_id})

        OPTIONAL MATCH (t:Entite {id: row.target_id})

        WITH row, s, t

        WHERE s IS NULL OR t IS NULL

        RETURN count(*) AS manquantes
        """,
        rows=all_rows
    ).single()

    missing_endpoints = (
        missing_result["manquantes"]
        if missing_result
        else 0
    )

    log.info(
        "Import liens termine : %s soumis, "
        "%s extremites manquantes, "
        "%s ignores, %s preuves.",
        imported,
        missing_endpoints,
        skipped,
        evidence_count
    )

    return (
        imported,
        missing_endpoints,
        skipped,
        evidence_count
    )


# ==========================================================
# 10. FONCTION PRINCIPALE
# ==========================================================

def main():

    if not NEO4J_PASSWORD:

        raise RuntimeError(
            "Variable NEO4J_PASSWORD absente. "
            "Definis-la dans l'environnement."
        )

    spark = create_spark()

    driver = None

    try:

        # Lecture des tables Delta Gold.
        log.info(
            "Lecture des noeuds : %s",
            NODES_PATH
        )

        nodes_df = (
            spark.read
            .format("delta")
            .load(NODES_PATH)
        )

        log.info(
            "Lecture des liens : %s",
            LINKS_PATH
        )

        links_df = (
            spark.read
            .format("delta")
            .load(LINKS_PATH)
        )

        # Affichage des schemas.
        log.info("Schema des noeuds :")
        nodes_df.printSchema()

        log.info("Schema des liens :")
        links_df.printSchema()

        nodes_total = nodes_df.count()
        links_total = links_df.count()

        log.info(
            "Gold : %s noeuds, %s liens.",
            nodes_total,
            links_total
        )

        # Connexion Neo4j.
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(
                NEO4J_USER,
                NEO4J_PASSWORD
            ),
            max_connection_pool_size=20
        )

        driver.verify_connectivity()

        log.info(
            "Connexion Neo4j reussie : %s",
            NEO4J_URI
        )

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            # La contrainte existante est conservee.
            session.run(
                """
                CREATE CONSTRAINT entite_id IF NOT EXISTS
                FOR (n:Entite)
                REQUIRE n.id IS UNIQUE
                """
            ).consume()

            # Import des noeuds.
            nodes_imported, nodes_skipped = (
                import_nodes(
                    session,
                    nodes_df
                )
            )

            # Import des liens.
            (
                links_imported,
                missing,
                links_skipped,
                evidences_imported
            ) = import_links(
                session,
                links_df
            )

            # Bilan final.
            summary = session.run(
                """
                MATCH (n:Entite:GraphRAGMaroc)

                WITH count(n) AS nodes

                OPTIONAL MATCH ()-[r]->()

                WHERE r.source = 'GraphRAGMaroc'

                RETURN
                    nodes,
                    count(r) AS relations
                """
            ).single()

            log.info(
                "===================================="
            )

            log.info(
                "IMPORT GRAPH RAG MAROC TERMINE"
            )

            log.info(
                "Noeuds soumis : %s",
                nodes_imported
            )

            log.info(
                "Noeuds ignores : %s",
                nodes_skipped
            )

            log.info(
                "Relations soumises : %s",
                links_imported
            )

            log.info(
                "Extremites manquantes : %s",
                missing
            )

            log.info(
                "Relations ignorees : %s",
                links_skipped
            )

            log.info(
                "Preuves traitees : %s",
                evidences_imported
            )

            if summary:

                log.info(
                    "Noeuds GraphRAGMaroc : %s",
                    summary["nodes"]
                )

                log.info(
                    "Relations GraphRAGMaroc : %s",
                    summary["relations"]
                )

            log.info(
                "===================================="
            )

    finally:

        if driver is not None:
            driver.close()

        spark.stop()


if __name__ == "__main__":
    main()