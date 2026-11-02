"""
02 - Silver : nettoyage, résolution d'entités et score de confiance.

Entrées  : tables Bronze Delta (entites, relations, wikidata_entites,
           wikidata_relations, documents)
Sorties  : /DataLake/silver/delta/
             noeuds         un noeud par entité réelle (QID Wikidata ou identifiant local)
             relations      une relation par (source, type, cible) + confiance + provenance
             mapping_noms   nom écrit dans les textes -> noeud (traçabilité)
             rejets         tout ce qui a été écarté, avec le motif

Nettoyage appliqué ICI (l'extraction LLM ne rejette plus rien sur le fond) :
  - entités / relations marquées hors du texte source (hallucinations probables)
  - résolution d'entités : accents, casse, parenthèses, sigles (PAM, RNI...),
    libellés et alias Wikidata
  - types : type FINAL du noeud (Wikidata en priorité) puis contrôle de compatibilité
  - fonctions génériques (« Secrétaire général ») rattachées à l'organisation
    de la page, avec la fonction en propriété `roles`
  - boucles, doublons, agrégation de la confiance (noisy-OR) et de la provenance

Variables d'environnement : LAKE_BASE, LAKE_FORMAT (voir 01_bronze_to_delta.py).
"""

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter

from pyspark.sql import SparkSession, Window, functions as F, types as T

LAKE_BASE = os.getenv("LAKE_BASE", "hdfs://namenode:9000/DataLake")
LAKE_FORMAT = os.getenv("LAKE_FORMAT", "delta")

BRONZE = f"{LAKE_BASE}/bronze/delta"
SILVER = f"{LAKE_BASE}/silver/delta"

BRONZE_ENTITES = f"{BRONZE}/entites"
BRONZE_RELATIONS = f"{BRONZE}/relations"
BRONZE_WD_ENTITES = f"{BRONZE}/wikidata_entites"
BRONZE_WD_RELATIONS = f"{BRONZE}/wikidata_relations"
BRONZE_DOCUMENTS = f"{BRONZE}/documents"

SILVER_NOEUDS = f"{SILVER}/noeuds"
SILVER_RELATIONS = f"{SILVER}/relations"
SILVER_MAPPING = f"{SILVER}/mapping_noms"
SILVER_REJETS = f"{SILVER}/rejets"

CONFIANCE_LLM = {
    "lexicale": 0.80,
    "structure": 0.70,
    "sujet_page": 0.60,
    "structure_page": 0.50,
    "non_verifiee": 0.30,   # relation extraite sans preuve exploitable : faible confiance
}
CONFIANCE_LLM_DEFAUT = 0.60
CONFIANCE_MAX = 0.99
TYPES_NOEUDS = {"PERSONNE", "PARTI", "ENTREPRISE", "INSTITUTION", "FONCTION", "ENTITE"}
RELATIONS_AUTORISEES = {
    "MEMBRE_DE", "OCCUPE_FONCTION", "DIRIGE", "PRESIDE", "FONDATEUR_DE",
    "ACTIONNAIRE_DE", "ADMINISTRATEUR_DE", "EMPLOYE_PAR", "FILIALE_DE",
    "CONJOINT_DE", "PARENT_DE", "FRATRIE_DE",
}
# Types (source, cible) acceptés par relation : contrôle fait ICI, sur le type
# FINAL du noeud (Wikidata en priorité), pas sur le type donné par le LLM.
COMPATIBILITE = {
    "MEMBRE_DE": ({"PERSONNE"}, {"PARTI", "INSTITUTION"}),
    "OCCUPE_FONCTION": ({"PERSONNE"}, {"FONCTION"}),
    "DIRIGE": ({"PERSONNE"}, {"ENTREPRISE", "INSTITUTION", "PARTI"}),
    "PRESIDE": ({"PERSONNE"}, {"ENTREPRISE", "INSTITUTION", "PARTI"}),
    "FONDATEUR_DE": ({"PERSONNE"}, {"ENTREPRISE", "INSTITUTION", "PARTI"}),
    "ACTIONNAIRE_DE": ({"PERSONNE"}, {"ENTREPRISE"}),
    "ADMINISTRATEUR_DE": ({"PERSONNE"}, {"ENTREPRISE"}),
    "EMPLOYE_PAR": ({"PERSONNE"}, {"ENTREPRISE", "INSTITUTION"}),
    "FILIALE_DE": ({"ENTREPRISE"}, {"ENTREPRISE"}),
    "CONJOINT_DE": ({"PERSONNE"}, {"PERSONNE"}),
    "PARENT_DE": ({"PERSONNE"}, {"PERSONNE"}),
    "FRATRIE_DE": ({"PERSONNE"}, {"PERSONNE"}),
}


def types_compatibles(relation, type_source, type_cible):
    """Un type inconnu (ENTITE / absent) n'est pas bloquant."""
    if relation not in COMPATIBILITE:
        return False
    ok_source, ok_cible = COMPATIBILITE[relation]
    return (type_source in (None, "ENTITE") or type_source in ok_source) and \
           (type_cible in (None, "ENTITE") or type_cible in ok_cible)


MOTS_VIDES = {"de", "du", "des", "la", "le", "les", "et", "d", "l", "en", "au", "aux", "a"}


# ------------------------------------------------------------------
# Normalisation et clés de résolution
# ------------------------------------------------------------------
def _sans_accents(valeur):
    valeur = unicodedata.normalize("NFKD", (valeur or "").replace("’", "'"))
    return "".join(c for c in valeur if not unicodedata.combining(c)).casefold()


def _nettoyer(valeur):
    valeur = re.sub(r"[^\w\s]", " ", valeur)
    return " ".join(valeur.split())


def sigle(nom):
    """Initiales d'un nom d'organisation : « Parti authenticité et modernité » -> pam."""
    brut = _sans_accents(re.sub(r"\([^)]*\)", " ", nom or "")).replace("'", " ")
    mots = [m for m in re.findall(r"\w+", brut) if m not in MOTS_VIDES]
    return "".join(m[0] for m in mots) if len(mots) >= 2 else ""


def cles_resolution(nom):
    """Liste de (clé, rang). Rang 0 = clé principale, 1 = variantes moins sûres."""
    if not nom:
        return []
    brut = _sans_accents(nom)
    entre_parentheses = re.findall(r"\(([^)]{2,15})\)", brut)
    principal = re.sub(r"\([^)]*\)", " ", brut)
    principal = re.sub(r"^groupe (?:parlementaire )?(?:(?:du|de la|de l'|des|de) ?)?", "", principal.strip())
    cles = [(_nettoyer(principal), 0)]
    sans_parti = re.sub(r"^parti (?:(?:du|de la|de l'|des|de) ?)?", "", principal.strip())
    if sans_parti != principal.strip():
        cles.append((_nettoyer(sans_parti), 1))
    for contenu in entre_parentheses:
        cles.append((_nettoyer(contenu), 1))
    vus, resultat = set(), []
    for cle, rang in cles:
        if cle and cle not in vus:
            vus.add(cle)
            resultat.append((cle, rang))
    return resultat


def fonction_generique(label):
    """« Secrétaire général », « Président »... : fonction sans organisation."""
    n = _sans_accents(label or "")
    return not re.search(r"\b(de|du|des|d'|de la|de l')\b|'", n) and len(n.split()) <= 3


PRIORITE_TYPES = ["PERSONNE", "ENTREPRISE", "PARTI", "INSTITUTION", "FONCTION"]
ORGANISATIONS = {"PARTI", "INSTITUTION", "ENTREPRISE"}


def identifiant_local(cle):
    return "L_" + hashlib.sha1(cle.encode("utf-8")).hexdigest()[:12]


def type_wikidata(entite):
    """Heuristique sur P31 / professions : PERSONNE, PARTI, ENTREPRISE, INSTITUTION."""
    texte = _sans_accents(json.dumps(
        [entite.get("types"), entite.get("descriptions")], ensure_ascii=False))
    if entite.get("annee_naissance") or entite.get("professions") \
            or re.search(r"\b(human|humain|personne)\b", texte):
        return "PERSONNE"
    if re.search(r"parti|party|political", texte):
        return "PARTI"
    if re.search(r"entreprise|societe|company|business|banque|holding|groupe", texte):
        return "ENTREPRISE"
    return "INSTITUTION"


# ------------------------------------------------------------------
# Spark
# ------------------------------------------------------------------
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


def lire(spark, chemin):
    return spark.read.format(LAKE_FORMAT).load(chemin)


def ecrire(df, chemin):
    (
        df.write.format(LAKE_FORMAT)
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(chemin)
    )


SCHEMA_REJETS = T.StructType([T.StructField(c, T.StringType()) for c in (
    "categorie", "motif", "document_id", "nom_source", "nom_cible", "type", "evidence")])


def nettoyer_bronze(df, colonnes_texte, colonne_type):
    """Espaces superflus, types en majuscules, lignes vides retirées."""
    for c in colonnes_texte:
        df = df.withColumn(c, F.trim(F.col(c)))
    if colonne_type:
        df = df.withColumn(colonne_type, F.upper(F.trim(F.col(colonne_type))))
    return df


# ------------------------------------------------------------------
# Construction des noeuds et des relations
# ------------------------------------------------------------------
def construire(spark):
    # ---------- Lecture Bronze + nettoyage de surface ----------
    ent_b = nettoyer_bronze(lire(spark, BRONZE_ENTITES), ["name", "document_id"], "type")
    rel_b = nettoyer_bronze(lire(spark, BRONZE_RELATIONS),
                            ["source", "target", "document_id"], "type")
    ent_b = ent_b.filter(F.col("name").isNotNull() & (F.col("name") != ""))
    rel_b = rel_b.filter(F.col("source").isNotNull() & (F.col("source") != "")
                         & F.col("target").isNotNull() & (F.col("target") != "")
                         & F.col("type").isNotNull() & (F.col("type") != ""))

    ent_llm = [r.asDict() for r in ent_b.collect()]
    rel_llm = [r.asDict() for r in rel_b.collect()]
    wd_entites = [json.loads(r["raw_json"]) for r in lire(spark, BRONZE_WD_ENTITES).collect()]
    wd_relations = [r.asDict() for r in lire(spark, BRONZE_WD_RELATIONS).collect()]
    wd = {"entities": wd_entites,
          "relations": [{"subject": r["subject"], "object": r["object"],
                         "relation": r["relation"], "debut": r["debut"], "fin": r["fin"],
                         "source_url": r["source_url"], "confiance": r["confiance"]}
                        for r in wd_relations]}

    # Titre de la page de chaque document = sujet de la page (organisation).
    titres = {r["document_id"]: r["titre"]
              for r in lire(spark, BRONZE_DOCUMENTS).collect() if r["titre"]}

    print(f"Wikidata : {len(wd['entities'])} entités, {len(wd['relations'])} relations")
    print(f"LLM      : {len(ent_llm)} entités, {len(rel_llm)} relations")

    rejets = []   # DataFrames de rejets, réunis à la fin

    # Documents sortis du corpus (pages hors périmètre exclues à l'audit) : leurs
    # entités et relations restent dans les fichiers d'extraction, on les écarte ici.
    ids_corpus = {r["document_id"] for r in lire(spark, BRONZE_DOCUMENTS).collect()}
    ent_hors = [e for e in ent_llm if e.get("document_id") not in ids_corpus]
    rel_hors = [r for r in rel_llm if r.get("document_id") not in ids_corpus]
    ent_llm = [e for e in ent_llm if e.get("document_id") in ids_corpus]
    rel_llm = [r for r in rel_llm if r.get("document_id") in ids_corpus]
    print(f"[SILVER] Hors corpus : {len(ent_hors)} entités, {len(rel_hors)} relations écartées.")
    rejets.append(spark.createDataFrame(
        [("entite", "document_hors_corpus", e.get("document_id"), e.get("name"), None, e.get("type"), None)
         for e in ent_hors]
        + [("relation", "document_hors_corpus", r.get("document_id"), r.get("source"),
            r.get("target"), r.get("type"), r.get("evidence")) for r in rel_hors],
        SCHEMA_REJETS))

    # ---------- Table des alias Wikidata (clé -> QID, clés uniques seulement) ----------
    lignes_alias, noeuds_wd = [], []
    for e in wd["entities"]:
        labels = e.get("labels") or {}
        noms = [e.get("label")] + list(labels.values())
        alias = e.get("aliases") or []
        if isinstance(alias, dict):
            alias = [a for liste in alias.values() for a in liste]
        noms += alias
        for nom in {n for n in noms if n}:
            for cle, _ in cles_resolution(nom):
                lignes_alias.append((cle, e["qid"]))
        for nom in (e.get("label"), labels.get("fr"), labels.get("en")):
            s = sigle(nom)
            if s:
                lignes_alias.append((s, e["qid"]))
        noeuds_wd.append((
            e["qid"], e.get("label") or labels.get("fr") or labels.get("en") or labels.get("ar") or e["qid"],
            type_wikidata(e),
            (e.get("descriptions") or {}).get("fr") or (e.get("descriptions") or {}).get("en"),
            json.dumps(e.get("comptes_officiels"), ensure_ascii=False) if e.get("comptes_officiels") else None,
            None if e.get("annee_naissance") is None else str(e.get("annee_naissance")),
        ))

    schema_alias = T.StructType([T.StructField("cle", T.StringType()), T.StructField("qid", T.StringType())])
    alias_df = spark.createDataFrame(lignes_alias, schema_alias).distinct()
    # une clé qui désigne plusieurs QID est ambiguë : on ne l'utilise pas
    alias_uniques = (alias_df.groupBy("cle")
                     .agg(F.countDistinct("qid").alias("n"), F.first("qid").alias("qid"))
                     .filter("n = 1").select("cle", "qid"))

    wd_noeuds = spark.createDataFrame(noeuds_wd, T.StructType([
        T.StructField("qid", T.StringType()), T.StructField("label_wd", T.StringType()),
        T.StructField("type_wd", T.StringType()), T.StructField("description_wd", T.StringType()),
        T.StructField("comptes_officiels", T.StringType()), T.StructField("annee_naissance", T.StringType()),
    ])) if noeuds_wd else None

    # ---------- Données LLM : écart des éléments hors du texte source ----------
    ent_ko = [e for e in ent_llm if e.get("dans_texte") is False]
    rel_ko = [r for r in rel_llm if r.get("preuve_dans_texte") is False]
    ent_llm = [e for e in ent_llm if e.get("dans_texte") is not False]
    rel_llm = [r for r in rel_llm if r.get("preuve_dans_texte") is not False]
    print(f"[SILVER] Entités écartées (hors texte) : {len(ent_ko)} ; "
          f"relations écartées (preuve hors texte) : {len(rel_ko)}")
    rejets.append(spark.createDataFrame(
        [("entite", "hors_du_texte_source", e.get("document_id"), e.get("name"), None, e.get("type"), None)
         for e in ent_ko]
        + [("relation", "preuve_hors_du_texte_source", r.get("document_id"), r.get("source"),
            r.get("target"), r.get("type"), r.get("evidence")) for r in rel_ko],
        SCHEMA_REJETS))

    schema_ent = T.StructType([
        T.StructField(c, T.StringType()) for c in ("name", "type", "description", "document_id", "source_url")])
    ent_df = spark.createDataFrame(
        [(e.get("name"), e.get("type"), e.get("description"), e.get("document_id"), e.get("source_url"))
         for e in ent_llm], schema_ent)

    schema_rel = T.StructType([T.StructField(c, T.StringType()) for c in (
        "source", "target", "type", "debut", "fin", "evidence", "validation", "document_id", "source_url", "titre")])
    rel_df = spark.createDataFrame(
        [tuple(titres.get(str(r.get("document_id"))) if c == "titre"
               else (None if r.get(c) is None else str(r.get(c))) for c in schema_rel.fieldNames())
         for r in rel_llm],
        schema_rel)

    # Les extrémités des relations deviennent des entités même si le LLM ne les
    # a pas listées (le nettoyage est fait ici, pas à l'extraction).
    for colonne in ("source", "target"):
        ent_df = ent_df.unionByName(rel_df.select(
            F.col(colonne).alias("name"), F.lit(None).cast("string").alias("type"),
            F.lit(None).cast("string").alias("description"), "document_id", "source_url"))

    # ---------- Résolution : nom -> identifiant de noeud ----------
    udf_cles = F.udf(cles_resolution, T.ArrayType(T.StructType([
        T.StructField("cle", T.StringType()), T.StructField("rang", T.IntegerType())])))
    noms = (ent_df.select(F.col("name").alias("nom"))
            .union(rel_df.select(F.col("source").alias("nom")))
            .union(rel_df.select(F.col("target").alias("nom")))
            .union(rel_df.select(F.col("titre").alias("nom")))
            .filter(F.col("nom").isNotNull()).distinct())
    noms_cles = (noms.withColumn("c", F.explode(udf_cles("nom")))
                 .select("nom", F.col("c.cle").alias("cle"), F.col("c.rang").alias("rang")))
    meilleur = (noms_cles.join(alias_uniques, "cle")
                .withColumn("r", F.row_number().over(Window.partitionBy("nom").orderBy("rang", "qid")))
                .filter("r = 1").select("nom", "qid"))
    cle_principale = (noms_cles.filter("rang = 0")
                      .withColumn("r", F.row_number().over(Window.partitionBy("nom").orderBy("cle")))
                      .filter("r = 1").select("nom", F.col("cle").alias("cle_locale")))
    # Sigles : « PAM » est rattaché à « Parti authenticité et modernité » quand le
    # rattachement est sans ambiguïté (nœuds locaux, hors Wikidata).
    table_sigles = {}
    cles_noms = {}
    for (nom,) in noms.collect():
        principal = (cles_resolution(nom) or [("", 0)])[0][0]
        cles_noms[nom] = principal
        if principal and len(principal.split()) >= 2:
            candidats = {sigle(nom)}
            candidats |= {c for c, rang in cles_resolution(nom) if rang == 1 and " " not in c and 2 <= len(c) <= 6}
            for c in candidats:
                if c:
                    table_sigles.setdefault(c, set()).add(principal)
    surcharges = [(nom, next(iter(table_sigles[cle])))
                  for nom, cle in cles_noms.items()
                  if nom.isupper() and " " not in cle and 2 <= len(cle) <= 6
                  and len(table_sigles.get(cle, ())) == 1]
    print(f"[SILVER] Sigles rattachés à leur nom complet : {len(surcharges)}")
    surcharge_df = spark.createDataFrame(surcharges, T.StructType([
        T.StructField("nom", T.StringType()), T.StructField("cle_sur", T.StringType())]))
    cle_principale = (cle_principale.join(surcharge_df, "nom", "left")
                      .withColumn("cle_locale", F.coalesce("cle_sur", "cle_locale")).drop("cle_sur"))
    udf_local = F.udf(identifiant_local, T.StringType())
    mapping = (noms.join(meilleur, "nom", "left").join(cle_principale, "nom", "left")
               .withColumn("node_id", F.coalesce("qid", udf_local("cle_locale")))
               .select("nom", "node_id", "qid"))
    mapping.cache()

    # ---------- Noeuds ----------
    llm_noeuds = (ent_df.join(mapping, ent_df.name == mapping.nom)
                  .groupBy("node_id")
                  .agg(F.collect_set("name").alias("noms"),
                       F.collect_list("type").alias("types"),
                       F.collect_set("description").alias("descriptions"),
                       F.collect_set("document_id").alias("documents")))
    wd_ids = wd_noeuds.select(F.col("qid").alias("node_id"), "label_wd", "type_wd",
                              "description_wd", "comptes_officiels", "annee_naissance") if wd_noeuds is not None else None

    ids_relations_wd = spark.createDataFrame(
        [(r["subject"],) for r in wd["relations"]] + [(r["object"],) for r in wd["relations"]],
        T.StructType([T.StructField("node_id", T.StringType())])).distinct()

    tous_ids = (llm_noeuds.select("node_id").union(ids_relations_wd)
                .union(wd_ids.select("node_id") if wd_ids is not None else ids_relations_wd.limit(0))
                .distinct())
    noeuds = tous_ids.join(llm_noeuds, "node_id", "left")
    if wd_ids is not None:
        noeuds = noeuds.join(wd_ids, "node_id", "left")
    else:
        for c in ("label_wd", "type_wd", "description_wd", "comptes_officiels", "annee_naissance"):
            noeuds = noeuds.withColumn(c, F.lit(None).cast("string"))

    def choisir(noms, types, label_wd, type_wd, node_id):
        noms = sorted(noms or [], key=lambda n: (-len(n), n))
        compte = Counter(t for t in (types or []) if t in TYPES_NOEUDS)
        if "ENTREPRISE" in compte and "INSTITUTION" in compte:
            type_llm = "ENTREPRISE"          # une société typée « institution » par erreur est fréquente
        elif compte:
            maxi = max(compte.values())
            type_llm = next(t for t in PRIORITE_TYPES + sorted(compte) if compte.get(t) == maxi)
        else:
            type_llm = None
        label = label_wd or (noms[0] if noms else node_id)
        return (label, type_wd or type_llm or "ENTITE")

    udf_choisir = F.udf(choisir, T.StructType([
        T.StructField("label", T.StringType()), T.StructField("type", T.StringType())]))
    noeuds = (noeuds.withColumn("c", udf_choisir("noms", "types", "label_wd", "type_wd", "node_id"))
              .select(
                  F.col("node_id").alias("id"),
                  F.col("c.label").alias("label"),
                  F.col("c.type").alias("type"),
                  F.when(F.col("node_id").rlike(r"^Q\d+$"), F.col("node_id")).alias("qid"),
                  F.coalesce(F.col("noms"), F.array()).alias("alias"),
                  F.coalesce(F.col("documents"), F.array()).alias("documents"),
                  "comptes_officiels", "annee_naissance", "descriptions", "description_wd")
              .withColumn("description",
                          F.coalesce("description_wd", F.when(F.size("descriptions") > 0, F.col("descriptions")[0])))
              .drop("descriptions", "description_wd")
              .withColumn("a_enrichir", F.col("qid").isNotNull() & (F.col("label") == F.col("qid"))))

    # ---------- Relations ----------
    conf_llm = F.create_map(*[x for k, v in CONFIANCE_LLM.items() for x in (F.lit(k), F.lit(v))])
    llm_rel = (rel_df.join(mapping.select(F.col("nom").alias("s_nom"), F.col("node_id").alias("src")),
                           rel_df.source == F.col("s_nom"))
               .join(mapping.select(F.col("nom").alias("t_nom"), F.col("node_id").alias("tgt")),
                     F.col("target") == F.col("t_nom"))
               .select("src", "tgt", F.col("type").alias("rel"), "debut", "fin", "evidence",
                       F.col("source_url").alias("url"), "document_id", "titre",
                       F.col("source").alias("nom_src"), F.col("target").alias("nom_tgt"),
                       F.coalesce(conf_llm[F.col("validation")], F.lit(CONFIANCE_LLM_DEFAUT)).alias("conf"),
                       F.concat(F.lit("llm:"), F.coalesce("validation", F.lit("lexicale"))).alias("methode")))

    # Contrôle des types FINAUX des noeuds (ex. une filiale typée PERSONNE est écartée).
    types_n = noeuds.select("id", "type", "label")
    llm_rel = (llm_rel
               .join(types_n.select(F.col("id").alias("src"), F.col("type").alias("type_src")), "src", "left")
               .join(types_n.select(F.col("id").alias("tgt"), F.col("type").alias("type_tgt"),
                                    F.col("label").alias("label_tgt")), "tgt", "left"))
    udf_compat = F.udf(types_compatibles, T.BooleanType())
    llm_rel = llm_rel.withColumn("compat", udf_compat("rel", "type_src", "type_tgt")).cache()
    n_avant = llm_rel.count()
    rejets.append(llm_rel.filter(~F.col("compat")).select(
        F.lit("relation").alias("categorie"),
        F.concat(F.lit("types_incompatibles:"), F.coalesce("type_src", F.lit("?")), F.lit(">"),
                 F.coalesce("type_tgt", F.lit("?"))).alias("motif"),
        "document_id", F.col("nom_src").alias("nom_source"), F.col("nom_tgt").alias("nom_cible"),
        F.col("rel").alias("type"), "evidence"))
    llm_rel = llm_rel.filter("compat")
    print(f"[SILVER] Relations LLM : {n_avant} -> {llm_rel.count()} après contrôle des types.")

    # Fonctions génériques (« Secrétaire général » seul) : un seul noeud pour tout le
    # monde ferait un faux hub. On relie la personne à l'ORGANISATION de la page
    # (sujet du document) avec la fonction en propriété `roles`. Sans organisation
    # identifiable, la relation est écartée.
    org_map = mapping.select(F.col("nom").alias("titre"), F.col("node_id").alias("org"))
    udf_generique = F.udf(fonction_generique, T.BooleanType())
    llm_rel = (llm_rel.join(org_map, "titre", "left")
               .join(types_n.select(F.col("id").alias("org"), F.col("type").alias("type_org")), "org", "left")
               .withColumn("generique", (F.col("rel") == "OCCUPE_FONCTION") & udf_generique("label_tgt"))
               .withColumn("org_ok", F.coalesce(
                   F.col("type_org").isin(*ORGANISATIONS) & (F.col("org") != F.col("src")), F.lit(False))))
    n_gen = llm_rel.filter("generique").count()
    n_gen_ok = llm_rel.filter("generique AND org_ok").count()
    print(f"[SILVER] Fonctions génériques : {n_gen} ; rattachées à l'organisation de la page : {n_gen_ok}.")
    rejets.append(llm_rel.filter(F.col("generique") & ~F.col("org_ok")).select(
        F.lit("relation").alias("categorie"), F.lit("fonction_generique_sans_organisation").alias("motif"),
        "document_id", F.col("nom_src").alias("nom_source"), F.col("nom_tgt").alias("nom_cible"),
        F.col("rel").alias("type"), "evidence"))
    llm_rel = (llm_rel.filter(~F.col("generique") | F.col("org_ok"))
               .withColumn("role", F.when(F.col("generique"), F.col("label_tgt")))
               .withColumn("tgt", F.when(F.col("generique"), F.col("org")).otherwise(F.col("tgt"))))

    wd_rel_df = spark.createDataFrame(
        [(r["subject"], r["object"], r["relation"],
          None if r.get("debut") is None else str(r["debut"]),
          None if r.get("fin") is None else str(r["fin"]),
          r.get("source_url"), float(r.get("confiance") or 0.9)) for r in wd["relations"]],
        T.StructType([T.StructField(c, T.StringType()) for c in ("src", "tgt", "rel", "debut", "fin", "url")]
                     + [T.StructField("conf", T.DoubleType())])
    ).withColumn("evidence", F.lit(None).cast("string")).withColumn("document_id", F.lit(None).cast("string")) \
     .withColumn("role", F.lit(None).cast("string")) \
     .withColumn("methode", F.lit("wikidata"))

    colonnes = ["src", "tgt", "rel", "debut", "fin", "evidence", "url", "document_id", "conf", "methode", "role"]
    toutes = (llm_rel.select(*colonnes).union(wd_rel_df.select(*colonnes))
              .filter(F.col("rel").isin(*RELATIONS_AUTORISEES))
              .filter(F.col("src").isNotNull() & F.col("tgt").isNotNull() & (F.col("src") != F.col("tgt")))
              .dropDuplicates(["src", "tgt", "rel", "methode", "url", "evidence"]))

    relations = (toutes.groupBy("src", "rel", "tgt")
                 .agg(
                     (F.lit(1.0) - F.exp(F.sum(F.log(F.lit(1.0) - F.col("conf"))))).alias("conf_brute"),
                     F.countDistinct("url").alias("n_sources"),
                     F.min("debut").alias("debut"), F.max("fin").alias("fin"),
                     F.slice(F.sort_array(F.collect_set("evidence")), 1, 5).alias("evidences"),
                     F.collect_set("url").alias("sources"),
                     F.collect_set("methode").alias("methodes"),
                     F.collect_set("role").alias("roles"),
                     F.collect_set("document_id").alias("documents"))
                 .withColumn("confidence", F.round(F.least(F.col("conf_brute"), F.lit(CONFIANCE_MAX)), 3))
                 .drop("conf_brute"))

    # Les noeuds FONCTION devenus inutiles (fonctions génériques rattachées à une
    # organisation) sont retirés : ils ne relient plus personne.
    refs = (relations.select(F.col("src").alias("id")).union(relations.select(F.col("tgt").alias("id"))).distinct())
    inutiles = noeuds.filter(F.col("type") == "FONCTION").join(refs, "id", "left_anti").select("id")
    noeuds = noeuds.join(inutiles, "id", "left_anti")

    rejets_df = rejets[0]
    for autre in rejets[1:]:
        rejets_df = rejets_df.unionByName(autre)
    return noeuds, relations, mapping, rejets_df


def rapport(noeuds, relations, mapping, rejets):
    print("\n========== RAPPORT QUALITÉ SILVER ==========")
    total = mapping.count()
    resolus = mapping.filter(F.col("qid").isNotNull()).count()
    print(f"Noms distincts : {total} | rattachés à un QID Wikidata : {resolus}")
    print("Noeuds par type :")
    noeuds.groupBy("type").count().orderBy(F.desc("count")).show(truncate=False)
    print("Relations par type :")
    relations.groupBy("rel").count().orderBy(F.desc("count")).show(truncate=False)
    multi = relations.filter(F.size("methodes") > 1).count()
    print(f"Relations confirmées par plusieurs méthodes (ex. Wikidata + LLM) : {multi}")
    print(f"Noeuds Wikidata sans libellé (à enrichir) : {noeuds.filter('a_enrichir').count()}")
    print("Éléments écartés, par motif :")
    rejets.groupBy("categorie", "motif").count().orderBy(F.desc("count")).show(truncate=False)


def main():
    spark = creer_spark("GraphRAG-Bronze-to-Silver")
    try:
        print("\n=== Construction de Silver ===")
        noeuds, relations, mapping, rejets = construire(spark)
        noeuds.cache()
        relations.cache()
        rejets.cache()
        rapport(noeuds, relations, mapping, rejets)

        print(f"\n=== Écriture des tables Silver ({LAKE_FORMAT}) ===")
        ecrire(noeuds.withColumn("silver_processed_at", F.current_timestamp()), SILVER_NOEUDS)
        ecrire(relations.withColumn("silver_processed_at", F.current_timestamp()), SILVER_RELATIONS)
        ecrire(mapping, SILVER_MAPPING)
        ecrire(rejets, SILVER_REJETS)

        print("\nSilver créé avec succès.")
        print(f"Noeuds : {SILVER_NOEUDS}")
        print(f"Relations : {SILVER_RELATIONS}")
        print(f"Mapping noms : {SILVER_MAPPING}")
        print(f"Rejets : {SILVER_REJETS}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
