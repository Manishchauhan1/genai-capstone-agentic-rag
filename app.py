"""Streamlit page: upload documents, ask questions, see answers with sources."""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from docqa import llm as llm_module
from docqa.agents.graph import run_agent
from docqa.config import load_settings
from docqa.display import plain
from docqa.errors import DocQAError
from docqa.uploads import UploadSync

load_dotenv()
logger = logging.getLogger("docqa.app")

st.set_page_config(page_title="Ask your documents", page_icon=":page_facing_up:")
st.title("Ask your documents")
st.caption("Upload PDF, TXT, CSV or Excel files, then ask questions. Answers cite their sources.")


def _environment() -> dict:
    """Environment variables, with Streamlit secrets filling any gaps (for deployment)."""
    env = dict(os.environ)
    try:
        for name in ("GROQ_API_KEY", "GROQ_MODEL"):
            if not env.get(name) and name in st.secrets:
                env[name] = st.secrets[name]
    except Exception:  # no secrets file is normal when running locally
        pass
    return env


try:
    settings = load_settings(_environment())
except DocQAError as error:
    st.error(
        f"{error}\n\nLocally: copy `.env.example` to `.env` and fill it in. "
        "On Streamlit Community Cloud: add the values under the app's Secrets."
    )
    st.stop()

if "uploads" not in st.session_state:
    st.session_state.uploads = UploadSync()
    st.session_state.llm = llm_module.build_llm(settings)
    st.session_state.messages = []

upload_sync: UploadSync = st.session_state.uploads


def _render_answer(answer: str, sources: list, trace: list) -> None:
    st.markdown(plain(answer))
    if sources:
        st.markdown("**Sources**")
        for source in sources:
            st.markdown(plain(f"- {source['filename']}, {source['location']}"))
    with st.expander("How the agents answered"):
        for line in trace:
            st.markdown(f"- {line}")


with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader(
        "Upload files", type=["pdf", "txt", "csv", "xlsx"], accept_multiple_files=True
    )
    with st.spinner("Reading your files..."):
        upload_sync.sync(uploads)
    workspace = upload_sync.workspace
    for level, text in upload_sync.notes.values():
        getattr(st, level)(plain(text))
    if workspace.filenames():
        st.markdown(plain("**Loaded:** " + ", ".join(workspace.filenames())))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(plain(message["content"]))
        else:
            _render_answer(message["content"], message["sources"], message["trace"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(plain(question))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = run_agent(question, upload_sync.workspace, st.session_state.llm)
            sources = [{"filename": s.filename, "location": s.location} for s in result.sources]
            _render_answer(result.answer, sources, result.trace)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": sources,
                 "trace": result.trace}
            )
        except DocQAError as error:
            st.error(str(error))
        except Exception:
            logger.exception("Unexpected error while answering")
            st.error("Something went wrong while answering. Please try again.")
