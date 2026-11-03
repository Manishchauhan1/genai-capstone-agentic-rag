import pytest

from docqa.config import Settings, load_settings
from docqa.errors import DocQAError


def test_load_settings_reads_env():
    settings = load_settings({"GROQ_API_KEY": "k", "GROQ_MODEL": "m"})
    assert settings == Settings("k", "m")


def test_missing_key_raises():
    with pytest.raises(DocQAError) as exc:
        load_settings({"GROQ_MODEL": "m"})
    assert "GROQ_API_KEY" in str(exc.value)


def test_missing_model_raises():
    with pytest.raises(DocQAError) as exc:
        load_settings({"GROQ_API_KEY": "k"})
    assert "GROQ_MODEL" in str(exc.value)


def test_empty_key_raises():
    with pytest.raises(DocQAError) as exc:
        load_settings({"GROQ_API_KEY": "", "GROQ_MODEL": "m"})
    assert "GROQ_API_KEY" in str(exc.value)
