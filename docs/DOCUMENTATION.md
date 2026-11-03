# Agentic Document Q&A: Documentation

Generative AI and ML Capstone Project (edureka and Illinois Tech)

This application lets a user upload documents (PDF, TXT, CSV, Excel) and ask questions in
natural language. LangGraph agents plan the task, retrieve relevant passages from a vector
database, reason over them and validate the answer before showing it. Every answer lists the
sources it came from.

Contents: 1. System setup · 2. Architecture · 3. Agent roles · 4. Deployment steps ·
5. Limitations · 6. Challenges faced during development

---

## 1. System setup

**Prerequisites**

- Python 3.11 or newer
- A free Groq account and API key (<https://console.groq.com>), and the name of a general chat
  model from the model list in your Groq console (not a "reasoning" model: the app strips
  `<think>` blocks from replies, but plain chat models are faster and use fewer free-tier tokens)
- Internet access the first time you run the app (it downloads a small embedding model once)

**Steps**

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env                      # Windows: copy .env.example .env
# edit .env: set GROQ_API_KEY and GROQ_MODEL

streamlit run app.py
```

The `.env` file is private. It is git-ignored and never included in the submission zip.

**Using the app.** Upload one or more files in the sidebar (PDF, TXT, CSV, or Excel `.xlsx`, up
to 10 MB each). Then type a question in the chat box. The answer appears with a Sources list and
an expandable "How the agents answered" trace. Demo files are in `samples/`.

**Running the tests**

```bash
python -m pytest                         # offline tests: no internet or API key needed
python -m pytest -m network -v           # real Groq call and real embedding model
python eval/run_eval.py --scores-only    # retrieval scores for the evaluation questions
python eval/run_eval.py                  # full 15-question evaluation, writes eval/results.md
```

## 2. Architecture

**One question, end to end**

```
 Upload                                   Question
   |                                         |
   v                                         v
 guardrails.validate_upload          guardrails.validate_question
   |                                         |
   v                                         v
 ingestion.parse_file             +--> PLANNER --------------+-------------------+
   |  (PDF/TXT/CSV/XLSX)          |    (lookup | table | unrelated)             |
   v                              |         |                |                  v
 chunking.chunk_document          |         v                v             "doesn't seem to
   |  (800 chars, table row       |    RETRIEVER         ANALYST             be about your
   |   groups with headers)       |    (Chroma search;   (LLM fills a         documents"
   v                              |     weak? rewrite    TableQuery; pandas
 Chroma vector store -------------+     up to 2x)        runs it)
   (local embeddings)                       |                |
                                            +-------+--------+
                                                    v
                                                ANSWERER  (answers only from the passages, cites [n])
                                                    |
                                                    v
                                                VALIDATOR (every claim supported? cited?)
                                                    |
                                                    v
                                         answer + sources + trace
```

**Modules**

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface |
| `docqa/guardrails.py` | Upload and question validation |
| `docqa/ingestion.py` | Parse PDF, TXT, CSV, Excel into a `ParsedDocument` |
| `docqa/chunking.py` | Split text into 800-character chunks (120 overlap) and tables into 20-row groups, each keeping file name and page or sheet |
| `docqa/vectorstore.py` | Chroma collection with cosine distance and local embeddings |
| `docqa/retrieval.py` | Top-5 similarity search with a weak-match flag |
| `docqa/workspace.py` | One user's files, tables and vector store |
| `docqa/llm.py` | LLM interface; Groq client with retry on rate limits |
| `docqa/rag.py` | Grounded answer prompt; document text is quoted as data |
| `docqa/tables.py` | Safe table-query tool (no code execution) |
| `docqa/agents/` | The five agent roles and the LangGraph workflow |

**Data flow.** A file is validated (type, size, content signature), parsed, split into chunks and
stored in Chroma with its file name and page or sheet. For CSV and Excel, the tables are also kept
in memory for the Analyst. A question is validated, then flows through the agent graph. The
Retriever searches the store; the Analyst computes from the tables. The Answerer writes a cited
answer from only those passages. The Validator checks it.

**Technology.** Python, Streamlit, LangGraph, Chroma (in-memory, local embeddings), Groq (free
tier) for the LLM, pandas, pypdf, openpyxl, pydantic.

## 3. Agent roles

| Role | Job | Tool | LLM calls |
| --- | --- | --- | --- |
| Planner | Decides whether the question needs a text lookup, a table calculation, or is unrelated to the files | none | 1 |
| Retriever | Searches the vector database. Judges relevance by similarity score (no LLM call). If the match is weak it rewrites the query and tries again, at most twice | vector search | 0 to 2 |
| Analyst | For table questions, the LLM writes a small structured query (table, operation, column, filters, group by). Our code runs it with plain pandas. If the query is invalid, the run falls back to text search | table-query tool | 1 |
| Answerer | Writes the answer using only the retrieved or computed passages, with citations such as [1] | none | 1 |
| Validator | Rejects an answer with no citation without calling the LLM. Otherwise checks that every claim is supported by the cited passages. A rejected answer becomes "I couldn't find this in your documents." | none | 0 or 1 |

Worst case is 5 LLM calls on the lookup route (Planner, two rewrites, Answerer, Validator) and 4
on the table route (Planner, Analyst, Answerer, Validator), or 6 if the Analyst's query is invalid
and the run falls back to text search (Planner, Analyst, two rewrites, Answerer, Validator). A hard
cap of 8 calls per question applies regardless.

## 4. Deployment steps

**Local run.** See part 1. This always works and is the fallback if the hosted app is asleep.

**Streamlit Community Cloud (free).** The app is deployed from a GitHub repository.

1. Push the project to a GitHub repository. Confirm `.env` is not tracked:
   `git ls-files | grep -E "^\.env$"` must print nothing.
2. Sign in at Streamlit Community Cloud with GitHub and create a new app from the repository,
   choosing `app.py` as the main file.
3. Open the app's Secrets settings and paste:

   ```toml
   GROQ_API_KEY = "your-key"
   GROQ_MODEL = "your-model-name"
   ```

4. Deploy. If the first deploy fails with "unsupported version of sqlite3", add
   `pysqlite3-binary` to `requirements.txt` and put the standard `pysqlite3` swap at the top of
   `app.py` before any Chroma import, then redeploy.
5. Open the live URL, upload `samples/company_policy.txt` and `samples/sales.csv`, and ask one
   text question and one table question.

Live URL: **TODO: add after deploying.**

Free-tier behaviour: the app sleeps after a period without visitors and takes a moment to wake,
and its disk is not persistent. This is fine here because each visitor's index is rebuilt in
memory for their session.

## 5. Limitations

- **Scanned PDFs are not supported.** There is no OCR. A PDF with no extractable text is rejected
  with a clear message.
- **Excel `.xls` is not supported**, only `.xlsx`.
- **Files are limited to 10 MB each**, and questions to 1,000 characters.
- **Free-tier rate limits.** Groq's free tier limits requests and tokens per minute. The app keeps
  prompts short, caps LLM calls per question, and retries with a pause, but heavy use can still
  show a "busy, try again in a minute" message.
- **Per-session memory.** Each visitor's documents and index live only for their session. Closing
  the tab, or the hosted app sleeping, clears them.
- **One answer from limited context.** Only the top 5 passages are used, so a question that needs
  information spread across many places may get an incomplete answer.
- **Table questions depend on the model producing a valid query.** If it cannot, the system falls
  back to text search, which is weaker for arithmetic.
- **No conversation memory.** Each question is answered on its own, so a follow-up such as "what
  about West?" does not know what "what" refers to. Ask the full question each time.
- **Excel sheets need the header in the first row.** A sheet with a title row above the column
  names is read with the wrong headers, so table questions on it fall back to text search.
- **Large table answers are shortened.** A grouped result with more than 50 groups shows the 50
  largest and says how many were left out.
- **Same file name, different content.** The second file is kept and shown as `name (2).ext`.
  Removing a file in the uploader rebuilds the index from the files that remain.
- **Evaluation.** The 15-question evaluation (12 fact questions and 3 that should be refused) is
  small. **TODO: add the real pass rate and any failures from `eval/results.md`.**

## 6. Challenges faced during development

- **Free-tier rate limits shaped the design.** One agent run makes several LLM calls, and the free
  tier allows few tokens per minute. The Retriever therefore judges relevance by similarity score
  instead of an LLM call, the Validator only sees cited passages, calls are capped at 8 per
  question, and the Groq client retries with a pause.
- **Testing without internet.** Chroma's embedding model downloads on first use. The vector store
  accepts an embedding function, so unit tests use a deterministic offline one, while separate
  `network` tests check the real model and the real Groq call.
- **Documents can contain instructions.** Document text is quoted as data, context tags inside it
  are neutralised so a file cannot break out of the quoted block, and tests cover this.
- **Messy real-world files.** Scanned PDFs have no text and Excel-exported CSVs are often
  Windows-1252. Scanned PDFs get a clear error, and decoding falls back from UTF-8 to Windows-1252
  to Latin-1.
- **Upload limit mismatch.** A screenshot of the running app showed Streamlit's default 200 MB
  upload limit against our 10 MB rule. The Streamlit config now matches, and a test keeps them
  equal.
- **Arithmetic without running model-written code.** The model fills in a structured query and our
  own pandas code runs it. A test proves a hostile filter value is never executed.
