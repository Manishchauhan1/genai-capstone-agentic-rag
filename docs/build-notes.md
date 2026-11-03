# Build notes

Real problems met while building, with what was done. The "Challenges" part of
`DOCUMENTATION.md` is written from this file.

- **Free-tier rate limits shaped the agent design.** Groq's free tier allows a small number of
  tokens per minute, and one agent run makes several LLM calls. Fix: the Retriever judges relevance
  with the similarity score instead of an LLM call, the Validator only sees the passages the answer
  cites, there is a hard cap of 8 LLM calls per question, and the Groq client retries with a pause.
- **The embedding model needs internet.** Chroma downloads its small local model on first use, so
  tests that depend on it could not run offline. Fix: the vector store accepts an embedding function;
  unit tests use a deterministic offline one, and separate `network`-marked tests check the real
  model and real Groq call on a machine with internet.
- **Documents can contain instructions.** A hostile file could try to close the `<context>` block
  and talk to the model. Fix: document text is quoted as data, any context tags inside it are
  neutralised, and tests cover both the injection sentence and the tag break-out.
- **Messy real-world files.** Scanned PDFs have no text, and CSV files saved by Excel are often
  Windows-1252, not UTF-8. Fix: scanned PDFs get a clear "No readable text found" message, and text
  decoding falls back from UTF-8 to Windows-1252 to Latin-1.
- **Streamlit's upload limit did not match ours.** A screenshot of the running app showed the
  default "200MB per file" while the guardrail allows 10 MB. Fix: `.streamlit/config.toml` sets
  `maxUploadSize = 10`, and a test keeps the two numbers equal.
- **The model must never run code.** Table questions ("total sales for the West region") need
  arithmetic. Fix: the model only fills in a small structured query, and our own code runs it with
  plain pandas; a test proves a hostile filter value is never executed.
