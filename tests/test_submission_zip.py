import zipfile

import pytest

from docqa.errors import DocQAError
from make_submission_zip import build_zip


def _write(root, relative, content="x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_zip_excludes_secrets_and_junk(tmp_path):
    project = tmp_path / "project"
    for name in [".env", ".venv/x", "__pycache__/y.pyc", "app.py", "docs/DOCUMENTATION.md"]:
        _write(project, name, "GROQ_API_KEY=abc123secret" if name == ".env" else "ok")
    names = build_zip(project, tmp_path / "out.zip")
    assert names == ["app.py", "docs/DOCUMENTATION.md"]
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        assert sorted(archive.namelist()) == names


def test_zip_refuses_when_key_appears_in_a_file(tmp_path):
    project = tmp_path / "project"
    _write(project, ".env", "GROQ_API_KEY=abc123secret\n")
    _write(project, "app.py", "key = 'abc123secret'")
    out = tmp_path / "out.zip"
    with pytest.raises(DocQAError) as exc:
        build_zip(project, out)
    assert "app.py" in str(exc.value)
    assert not out.exists()


def test_zip_leaves_out_nested_caches_zips_and_real_secrets_only(tmp_path):
    project = tmp_path / "project"
    for name in [
        "docqa/__pycache__/x.pyc",
        "docqa/a.py",
        "old.zip",
        ".streamlit/secrets.toml",
        ".streamlit/secrets.toml.example",
        ".streamlit/config.toml",
        ".git/config",
        ".pytest_cache/v",
        ".superpowers/sdd/ledger.md",
        "tests/fixtures/sample.pdf",
        "tests/test_a.py",
    ]:
        _write(project, name)
    names = build_zip(project, tmp_path / "out.zip")
    assert names == [
        ".streamlit/config.toml",
        ".streamlit/secrets.toml.example",
        "docqa/a.py",
        "tests/test_a.py",
    ]


def test_zip_works_without_an_env_file(tmp_path):
    project = tmp_path / "project"
    _write(project, "app.py")
    assert build_zip(project, tmp_path / "out.zip") == ["app.py"]
