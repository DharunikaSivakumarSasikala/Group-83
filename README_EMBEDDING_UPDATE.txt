SkillScope AI - Embedding Retriever Update
==========================================

What this update does
---------------------
- Keeps the existing deterministic competency scoring unchanged.
- Keeps role filtering unchanged.
- Adds a local dense embedding retriever using Ollama + nomic-embed-text.
- Caches the 120 knowledge-base vectors locally under results/embedding_cache/.
- Embeds only the incoming query at runtime after the cache is created.
- Keeps TF-IDF as a selectable baseline/fallback.
- Adds the active retriever name to RAG output for traceability.

Files to replace/add
--------------------
Replace:
  skillscope_core.py
  rag_pipeline.py
  web_app.py
  demo.py
  requirements.txt

Add:
  test_embedding_retriever.py
  smoke_test_embedding_retriever.py

Configuration
-------------
Embedding mode (now the default):
  export SKILLSCOPE_RETRIEVER=embedding

TF-IDF baseline:
  export SKILLSCOPE_RETRIEVER=tfidf

Embedding model:
  export SKILLSCOPE_EMBED_MODEL=nomic-embed-text

By default, if the embedding service is unavailable, SkillScope prints a warning
and falls back to TF-IDF. To disable that fallback:
  export SKILLSCOPE_RETRIEVER_FALLBACK=none

Install / verify
----------------
1. Activate your existing .venv.
2. Run:
     pip install -r requirements.txt
3. Verify model:
     ollama list
   If nomic-embed-text is missing:
     ollama pull nomic-embed-text
4. Run all tests:
     python -m unittest -v test_prototype.py test_rag_pipeline.py test_v3_features.py test_embedding_retriever.py
5. Run retriever smoke test:
     SKILLSCOPE_RETRIEVER=embedding python smoke_test_embedding_retriever.py
6. Start web app:
     SKILLSCOPE_RETRIEVER=embedding python web_app.py

Expected first startup
----------------------
Retriever: embedding
Embedding model: nomic-embed-text (built and cached)

Expected later startups
-----------------------
Retriever: embedding
Embedding model: nomic-embed-text (loaded from cache)

Baseline comparison
-------------------
To demonstrate the old baseline without changing code:
  SKILLSCOPE_RETRIEVER=tfidf python web_app.py
