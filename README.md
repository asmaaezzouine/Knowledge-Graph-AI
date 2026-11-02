# 🇲🇦 GraphRAG Maroc — Knowledge Graph & Retrieval-Augmented Generation

## 1. Présentation du projet

**GraphRAG Maroc** est un projet de construction et d'exploitation d'un graphe de connaissances consacré aux informations et ressources documentaires relatives au Maroc.

Le projet combine des techniques de collecte de données, de traitement distribué, de transformation ETL, d'extraction d'entités et de relations, de graphes de connaissances et de modèles de langage (LLM).

L'objectif est de structurer des données documentaires sous forme de connaissances interconnectées, de les stocker dans une base de données orientée graphe et de permettre leur exploitation à travers une API et une interface web.

Le projet s'appuie sur une architecture modulaire organisée autour de plusieurs composants :

- Collecte et extraction des données.
- Préparation et nettoyage des corpus.
- Traitement et transformation ETL.
- Stockage et traitement distribué avec Hadoop et Apache Spark.
- Construction et exploitation d'un graphe de connaissances avec Neo4j.
- Extraction d'entités et de relations avec des modèles de langage.
- Exposition des fonctionnalités via une API REST.
- Interface web pour l'interaction avec la plateforme.

---

## 2. Objectifs du projet

Les principaux objectifs sont :

1. Collecter des données documentaires relatives au Maroc à partir de différentes sources.
2. Extraire et préparer le contenu textuel des documents.
3. Organiser les données selon une architecture de traitement structurée.
4. Transformer les données à travers plusieurs étapes ETL.
5. Identifier les entités et les relations présentes dans les documents.
6. Construire un graphe de connaissances dans Neo4j.
7. Exploiter les modèles de langage pour faciliter l'extraction et l'analyse des connaissances.
8. Mettre à disposition une API permettant d'accéder aux fonctionnalités du système.
9. Fournir une interface web pour l'utilisation de la plateforme.

---

## 3. Architecture générale

L'architecture du projet est organisée en plusieurs couches complémentaires.

### 3.1. Couche de collecte (Crawler)

Le module `crawler/` contient les scripts responsables de la collecte et de la préparation des données.

Principaux scripts :

- `config_corpus.py` : configuration du corpus.
- `crawler_hdfs.py` : collecte et intégration des données dans l'environnement HDFS.
- `expand_wikidata.py` : enrichissement des données à partir de Wikidata.
- `extract_text.py` : extraction du contenu textuel des documents.
- `preparer_corpus_graphrag.py` : préparation du corpus destiné au traitement GraphRAG.

### 3.2. Couche ETL

Le dossier `ETL/Scripts/` contient les scripts de transformation et de contrôle des données.

Les principales étapes sont :

| Script | Fonction |
|---|---|
| `01_bronze_to_delta.py` | Traitement de la couche Bronze vers Delta |
| `02_bronze_to_silver (1).py` | Transformation des données vers la couche Silver |
| `03_silver_to_gold.py` | Transformation de la couche Silver vers Gold |
| `04_verifier_silver.py` | Vérification de la couche Silver |
| `04_verifier_gold.py` | Vérification de la couche Gold |
| `05_silver_to_neo4j.py` | Intégration des données Silver dans Neo4j |
| `06_compter_silver.py` | Comptage des données de la couche Silver |

Le projet utilise une organisation de type Medallion Architecture :

- **Bronze** : données brutes collectées.
- **Silver** : données nettoyées et structurées.
- **Gold** : données transformées et préparées pour leur exploitation.

### 3.3. Couche de stockage et de traitement distribué

L'infrastructure Hadoop et Spark permet de prendre en charge le stockage et le traitement des données.

Technologies utilisées :

- Hadoop HDFS.
- Apache Spark.
- Docker.
- PostgreSQL.
- Apache Superset.

La configuration Docker correspondante est disponible dans `docker/hdfs-spark/`.

### 3.4. Couche graphe de connaissances

Neo4j est utilisé comme système de gestion de base de données orientée graphe.

Cette couche permet de représenter les connaissances sous forme :

- D'entités représentées par des nœuds.
- De relations entre les entités.
- De propriétés associées aux nœuds et aux relations.

La configuration Docker de Neo4j est disponible dans `docker/neo4j/`.

### 3.5. Couche LLM et API

Le dossier `llm/` regroupe les fonctionnalités liées aux modèles de langage et à l'exploitation du graphe.

Principaux fichiers :

- `main.py` : point d'entrée de l'application API.
- `graphrag_api.py` : définition de l'API.
- `extras.py` : fonctionnalités complémentaires de l'application.
- `extraire_entites_relations.py` : extraction d'entités et de relations.

Le projet utilise notamment FastAPI, Groq et Google Gemini.

### 3.6. Interface utilisateur

Le dossier `frontend/` contient l'interface web du projet.

Fichier principal :

- `plateforme.html`

---

## 4. Structure du dépôt

```text
Knowledge-Graph-AI/
│
├── crawler/
│   ├── config_corpus.py
│   ├── crawler_hdfs.py
│   ├── expand_wikidata.py
│   ├── extract_text.py
│   └── preparer_corpus_graphrag.py
│
├── ETL/
│   └── Scripts/
│       ├── 01_bronze_to_delta.py
│       ├── 02_bronze_to_silver (1).py
│       ├── 03_silver_to_gold.py
│       ├── 04_verifier_gold.py
│       ├── 04_verifier_silver.py
│       ├── 05_silver_to_neo4j.py
│       └── 06_compter_silver.py
│
├── frontend/
│   └── plateforme.html
│
├── llm/
│   ├── main.py
│   ├── graphrag_api.py
│   ├── extras.py
│   ├── extraire_entites_relations.py
│   ├── .env.example
│   └── etl/
│       ├── 01_bronze.py
│       ├── 02_silver.py
│       ├── 03_gold.py
│       ├── 04_import_neo4j.py
│       └── controle_silver.py
│
├── docker/
│   ├── hdfs-spark/
│   │   ├── docker-compose.yml
│   │   └── Dockerfile.jupyter
│   │
│   └── neo4j/
│       ├── docker-compose.yml
│       └── Dockerfile
│
├── docs/
│   ├── architecture.md
│   └── installation.md
│
├── scripts/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement des scripts et de l'API |
| FastAPI | Développement de l'API REST |
| Apache Spark | Traitement distribué des données |
| Hadoop HDFS | Stockage distribué |
| Neo4j | Stockage du graphe de connaissances |
| Groq | Accès aux modèles de langage |
| Google Gemini | Services de modèles de langage |
| Docker | Conteneurisation de l'infrastructure |
| PostgreSQL | Service de base de données associé à l'infrastructure |
| Apache Superset | Visualisation et exploration des données |
| BeautifulSoup | Extraction de contenu web |
| PyMuPDF | Extraction de contenu PDF |
| Requests | Requêtes HTTP |
| Git / GitHub | Gestion du code source |

---

## 6. Prérequis

Avant de commencer, installer les outils suivants :

- Python 3.10 ou version compatible avec les bibliothèques du projet.
- Git.
- Docker Desktop.
- Docker Compose.
- Une instance Neo4j.
- Les services Hadoop et Spark nécessaires aux traitements ETL.
- Une clé API Groq.
- Une clé API Google Gemini si les fonctionnalités correspondantes sont utilisées.

Les ressources nécessaires dépendent de la taille du corpus et de l'environnement de traitement.

---

## 7. Installation

### 7.1. Cloner le dépôt

```bash
git clone https://github.com/asmaaezzouine/Knowledge-Graph-AI.git
cd Knowledge-Graph-AI
```

### 7.2. Créer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
```

Activer l'environnement :

```powershell
.venv\Scripts\Activate.ps1
```

Sous Linux ou macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.3. Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Remarque :** les traitements Spark nécessitent un environnement disposant de PySpark et des composants compatibles avec le cluster Spark. L'installation locale de toutes les dépendances ne remplace pas la configuration du cluster.

---

## 8. Configuration des variables d'environnement

Le fichier modèle est disponible dans :

`llm/.env.example`

Créer une copie nommée `.env` dans le dossier `llm/`.

Sous PowerShell :

```powershell
Copy-Item .\llm\.env.example .\llm\.env
```

Renseigner ensuite les paramètres nécessaires :

### Clés API

- `GEMINI_API_KEY`
- `GROQ_API_KEY`

### Connexion Neo4j

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

### Paramètres de traitement

- `GROQ_MODELS`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MAX_OUTPUT_TOKENS`
- `REASONING_EXTRA_TOKENS`
- `TPM_LIMIT`
- `CHUNK_PAUSE`

### Paramètres du graphe

- `GRAPH_LABEL`
- `RELATION_SOURCE`

### Sécurité et authentification

- `AUTH_REQUIRED`
- `AUTH_USERS`
- `JWT_SECRET`
- `CORS_ORIGINS`

### Audit

- `AUDIT_RETENTION_JOURS`

Les valeurs doivent être adaptées à l'environnement d'exécution.

**Important :** ne jamais publier le fichier `.env`, les clés API, les mots de passe ou les secrets JWT sur GitHub.

---

## 9. Configuration de l'infrastructure Docker

Les fichiers Docker Compose sont organisés en deux parties.

### 9.1. Infrastructure Hadoop / Spark

Emplacement :

`docker/hdfs-spark/`

Démarrer les services depuis ce dossier :

```powershell
cd docker/hdfs-spark
docker compose up -d
```

Vérifier les conteneurs :

```powershell
docker compose ps
```

Afficher les logs :

```powershell
docker compose logs
```

### 9.2. Infrastructure Neo4j

Emplacement :

`docker/neo4j/`

Démarrer les services :

```powershell
cd docker/neo4j
docker compose up -d
```

Vérifier les conteneurs :

```powershell
docker compose ps
```

Accès habituel à Neo4j Browser :

http://localhost:7474

Connexion Bolt :

`bolt://localhost:7687`

Les identifiants doivent correspondre aux variables configurées dans l'environnement.

**Attention :** les paramètres Docker doivent être configurés avant le démarrage. Les valeurs d'exemple ne doivent jamais être utilisées en production.

---

## 10. Exécution de l'API

Revenir à la racine du projet :

```powershell
cd ..\..
```

Depuis `Knowledge-Graph-AI/`, lancer :

```powershell
python -m uvicorn llm.main:app --host 127.0.0.1 --port 8000
```

L'API est accessible à :

http://127.0.0.1:8000

Documentation interactive FastAPI :

http://127.0.0.1:8000/docs

---

## 11. Pipeline de traitement des données

Le traitement général repose sur les étapes suivantes :

1. Collecte des données avec les scripts du crawler.
2. Extraction du contenu textuel des documents.
3. Préparation et organisation du corpus.
4. Stockage des données dans l'environnement HDFS.
5. Traitement et transformation ETL.
6. Contrôle et validation des données transformées.
7. Extraction des entités et des relations.
8. Intégration des connaissances dans Neo4j.
9. Exploitation du graphe à travers l'API et les modèles de langage.

Les scripts et leur ordre d'exécution doivent être adaptés à la configuration des services et aux chemins de données utilisés dans l'environnement.

---

## 12. Données du projet

Les données nécessaires au fonctionnement complet du projet comprennent notamment :

- Les corpus documentaires.
- Les fichiers JSON générés pendant la collecte.
- Les données intermédiaires des couches Bronze, Silver et Gold.
- Les fichiers de préparation et d'enrichissement.
- Les données du graphe Neo4j.
- Les checkpoints et les résultats intermédiaires.

Pour des raisons de taille, de sécurité et de gestion des versions, ces données ne sont pas stockées dans le dépôt GitHub.

Elles doivent être transmises séparément sous forme d'archives ou par un espace de stockage adapté.

Pour reproduire complètement le projet, il est nécessaire de disposer :

1. Du code source présent dans ce dépôt.
2. Des fichiers de configuration adaptés à l'environnement.
3. Des données nécessaires au traitement.
4. Des services Docker et des bases de données configurés.
5. Des clés API requises.

Les procédures de restauration des données doivent accompagner leur livraison.

---

## 13. Sécurité

Les règles suivantes doivent être respectées :

- Ne jamais publier de fichiers `.env`.
- Ne jamais publier de clés API.
- Ne jamais publier de mots de passe.
- Ne jamais publier de secrets JWT.
- Ne pas versionner les environnements virtuels Python.
- Ne pas versionner les caches et fichiers temporaires.
- Ne pas versionner les corpus JSON privés ou volumineux.
- Utiliser des variables d'environnement pour les informations sensibles.
- Utiliser des mots de passe forts pour Neo4j et les services Docker.

---

## 14. Documentation complémentaire

Des informations complémentaires sont disponibles dans :

- `docs/architecture.md` : présentation de l'architecture.
- `docs/installation.md` : procédure d'installation.
- `docker/` : configurations de l'infrastructure.
- `requirements.txt` : dépendances Python.
- `llm/.env.example` : modèle de configuration des variables d'environnement.

---

## 15. État du projet

Le dépôt contient actuellement :

- Les scripts de collecte.
- Les scripts ETL.
- Les scripts d'extraction d'entités et de relations.
- L'API FastAPI.
- L'interface web.
- Les configurations Docker.
- La documentation initiale.

La reproduction complète nécessite également la préparation des données, la configuration des services et le renseignement des variables d'environnement.

---

