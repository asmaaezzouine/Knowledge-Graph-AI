"""
Point d'entrée de la plateforme : API existante + extension.
Évite de modifier graphrag_api.py.

Lancement (depuis C:\\graphrag-project) :
    python -m uvicorn llm.main:app --host 127.0.0.1 --port 8000
"""

try:
    from . import graphrag_api as api
    from . import extras
except ImportError:  # lancé depuis le dossier llm/
    import graphrag_api as api
    import extras

app = api.app
extras.register(app, api)
