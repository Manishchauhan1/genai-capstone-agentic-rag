# Agentic Document Q&A

Upload PDF, TXT, CSV or Excel files and ask questions in plain language. A small
team of LangGraph agents plans the task, retrieves the relevant passages from a
vector database, reasons over them and checks the answer before showing it, with
the sources it used.

Live demo: _add the Streamlit URL here after deploying (see `docs/DOCUMENTATION.md`, part 4)._

Full documentation (setup, architecture, agent roles, deployment, limitations and
challenges) is in [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Quick start

You need Python 3.11 or newer and a free Groq API key from <https://console.groq.com>.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env                 # Windows: copy .env.example .env
# open .env and fill in GROQ_API_KEY and GROQ_MODEL (a general chat model from your Groq console list)

streamlit run app.py
```

Open the address Streamlit prints, upload a file from `samples/`, and ask a
question such as "How many days do customers have to request a refund?".

The first upload downloads a small embedding model (internet needed, one time).
Your `.env` file is private: it is git-ignored and left out of the submission zip.

## Tests

```bash
python -m pytest                      # 112 offline tests, no internet or key needed
python -m pytest -m network -v        # real Groq call + real embedding model (needs .env and internet)
python eval/run_eval.py --scores-only # retrieval scores for the 15 evaluation questions
python eval/run_eval.py               # full evaluation, writes eval/results.md
```

## Project layout

| Path | What it does |
| --- | --- |
| `app.py` | Streamlit page: upload, ask, answers with sources and an agent trace |
| `docqa/guardrails.py` | Checks uploads and questions |
| `docqa/ingestion.py`, `chunking.py` | Read files and split them into chunks |
| `docqa/vectorstore.py`, `retrieval.py`, `workspace.py` | Chroma store, similarity search, one user's session |
| `docqa/llm.py`, `rag.py` | Groq client with retries; grounded answer prompt |
| `docqa/tables.py` | Safe table-query tool used by the Analyst agent |
| `docqa/agents/` | The five agent roles and the LangGraph workflow |
| `samples/`, `eval/` | Demo files and the 15-question evaluation |
| `make_submission_zip.py` | Builds the submission zip without secrets |
