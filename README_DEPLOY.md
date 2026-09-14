# Streamlit deployment package

This package was prepared from `sol(3).ipynb`.

## Required repository structure

```text
repo/
├─ streamlit_app.py
├─ backend.py
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│  └─ secrets.example.toml
└─ ALLURE Docs/
   ├─ *.md
   └─ Knowledge Base/
      ├─ *.md
      ├─ Products Prices.xlsx
      ├─ Products Stock.xlsx
      └─ Infinity_Products_DB_Ready.xlsx
```

The complete `ALLURE Docs` folder was not included in the uploaded conversation, so it must be copied into the repository before the app can run.

## Local run

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.streamlit/secrets.toml` from the example and put a **new** OpenRouter key in it.
4. Run:

```bash
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder plus `ALLURE Docs` to a GitHub repository.
2. Do **not** commit `.streamlit/secrets.toml`.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Entry point: `streamlit_app.py`.
5. In **Advanced settings**, select Python 3.12.
6. In **Secrets**, add:

```toml
OPENROUTER_API_KEY = "your-new-key"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-2507"
```

7. Deploy.

## Changes made for deployment

- Removed the hard-coded OpenRouter API key and moved it to Streamlit Secrets.
- Generated a Streamlit chat UI around the LangGraph graph.
- Added LangGraph `Command(resume=...)` handling for interrupts.
- Replaced the fixed `thread_id="user-1"` with a unique browser-session thread ID.
- Isolated the local long-term-memory JSONL file per session instead of sharing one file across all visitors.
- Changed the SQLite product database to an in-memory database initialized from the Excel source.
- Replaced order-flow references to an undefined global `bm25` with local BM25 retrievers.
- Replaced fragile positional metadata assignment (`docs[0]`, `docs[1]`, etc.) with filename-based mapping.
- Added a unified LangGraph state schema so current LangGraph versions can accept the keys returned by routine/safety/product nodes.
- Replaced the hard-coded scientific RAG concern `['acne']` with the concern extracted into graph state.

## Important limitations before public production use

- Streamlit Community Cloud's local filesystem is ephemeral. The per-session JSONL memory is fine for a demo but is **not durable storage**. Use a database for real persistent user memory.
- The Hugging Face embedding and reranker models are relatively heavy. Cold start and RAM use can be significant on Community Cloud.
- The app still contains the notebook's business logic and prompts. This package focuses on making it deployable; you should run end-to-end tests for every route before exposing it to customers.
