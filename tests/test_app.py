from pathlib import Path

from conftest import FakeLLM
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def test_shows_setup_message_when_key_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_MODEL", "")
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert any("GROQ_API_KEY" in error.value for error in at.error)
    assert len(at.chat_input) == 0


def test_renders_chat_input_when_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("GROQ_MODEL", "dummy-model")
    monkeypatch.setattr("docqa.llm.build_llm", lambda settings: FakeLLM([]))
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert len(at.chat_input) == 1
    assert len(at.error) == 0


def test_asking_before_uploading_shows_friendly_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("GROQ_MODEL", "dummy-model")
    fake = FakeLLM([])
    monkeypatch.setattr("docqa.llm.build_llm", lambda settings: fake)
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("What is the refund window?").run()
    assert not at.exception
    assert any("upload" in error.value.lower() for error in at.error)
    assert fake.calls == []


def test_streamlit_upload_limit_matches_guardrail():
    import tomllib

    from docqa.guardrails import MAX_UPLOAD_BYTES

    config = tomllib.loads((Path(APP).parent / ".streamlit" / "config.toml").read_text())
    assert config["server"]["maxUploadSize"] * 1024 * 1024 == MAX_UPLOAD_BYTES


def test_answers_show_dollar_amounts_as_text_not_math(monkeypatch):
    from docqa.models import AgentResult

    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("GROQ_MODEL", "dummy-model")
    monkeypatch.setattr("docqa.llm.build_llm", lambda settings: FakeLLM([]))
    monkeypatch.setattr(
        "docqa.agents.graph.run_agent",
        lambda question, workspace, llm: AgentResult("East $1,200 and West $2,400", [], ["trace"], 1),
    )
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("What did each region sell?").run()
    assert not at.exception
    assert any(r"\$1,200" in m.value and r"\$2,400" in m.value for m in at.markdown)
