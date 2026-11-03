from types import SimpleNamespace

from docqa.display import plain
from docqa.uploads import UploadSync
from docqa.workspace import Workspace


def _upload(file_id, name, data):
    return SimpleNamespace(file_id=file_id, name=name, size=len(data), getvalue=lambda: data)


def _sync(offline_embedding, factory=None):
    return UploadSync(factory or (lambda: Workspace(embedding_function=offline_embedding)))


def test_added_files_are_loaded_once(offline_embedding):
    sync = _sync(offline_embedding)
    a = _upload("1", "a.txt", b"Refund window is thirty days.")
    sync.sync([a])
    sync.sync([a])
    assert sync.workspace.filenames() == ["a.txt"]
    assert sync.notes["1"][0] == "success"


def test_removing_a_file_removes_it_from_the_workspace(offline_embedding):
    sync = _sync(offline_embedding)
    a = _upload("1", "a.txt", b"Alpha document text.")
    b = _upload("2", "b.txt", b"Beta document text.")
    sync.sync([a, b])
    sync.sync([a])
    assert sync.workspace.filenames() == ["a.txt"]
    assert "2" not in sync.notes


def test_unexpected_failure_is_retried_on_the_next_run(offline_embedding):
    class Flaky(Workspace):
        failures = 1

        def add_file(self, filename, data):
            if Flaky.failures:
                Flaky.failures -= 1
                raise RuntimeError("model download failed")
            return super().add_file(filename, data)

    sync = _sync(offline_embedding, lambda: Flaky(embedding_function=offline_embedding))
    a = _upload("1", "a.txt", b"Alpha document text.")
    sync.sync([a])
    assert sync.notes["1"][0] == "error"
    sync.sync([a])
    assert sync.workspace.filenames() == ["a.txt"]
    assert sync.notes["1"][0] == "success"


def test_unreadable_file_reports_error_and_is_not_retried(offline_embedding):
    calls = []

    class Counting(Workspace):
        def add_file(self, filename, data):
            calls.append(filename)
            return super().add_file(filename, data)

    sync = _sync(offline_embedding, lambda: Counting(embedding_function=offline_embedding))
    bad = _upload("1", "bad.pdf", b"not a pdf at all")
    sync.sync([bad])
    sync.sync([bad])
    assert sync.notes["1"][0] == "error"
    assert calls == ["bad.pdf"]


def test_plain_escapes_dollar_signs_so_amounts_are_not_math():
    assert plain("East sold $1,200 and West $2,400") == r"East sold \$1,200 and West \$2,400"
