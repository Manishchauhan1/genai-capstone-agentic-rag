"""Keeps the workspace in step with the files shown in the uploader."""

import logging
from typing import Callable

from docqa.errors import DocQAError
from docqa.workspace import Workspace

logger = logging.getLogger("docqa.uploads")


class UploadSync:
    """Adds newly uploaded files, rebuilds the workspace when a file is removed, and
    keeps one note per file for the page to show.

    A file is marked done after success or a clear DocQAError. An unexpected failure
    (for example a one-time model download problem) is not marked done, so the next
    run tries it again.
    """

    def __init__(self, workspace_factory: Callable[[], Workspace] = Workspace):
        self._factory = workspace_factory
        self.workspace = workspace_factory()
        self._done: set[str] = set()
        self.notes: dict[str, tuple[str, str]] = {}

    def sync(self, uploads) -> None:
        uploads = list(uploads or [])
        current = {upload.file_id for upload in uploads}
        if self._done - current:  # a file was removed: start clean from what is left
            self.workspace = self._factory()
            self._done = set()
        self.notes = {file_id: note for file_id, note in self.notes.items() if file_id in current}
        for upload in uploads:
            if upload.file_id in self._done:
                continue
            try:
                result = self.workspace.add_file(upload.name, upload.getvalue())
                if result.duplicate:
                    self.notes[upload.file_id] = ("info", f"{upload.name} was already added.")
                else:
                    self.notes[upload.file_id] = (
                        "success", f"Added {upload.name} ({result.chunks_added} chunks).")
                self._done.add(upload.file_id)
            except DocQAError as error:
                self.notes[upload.file_id] = ("error", str(error))
                self._done.add(upload.file_id)
            except Exception:
                logger.exception("Unexpected error while adding %s", upload.name)
                self.notes[upload.file_id] = (
                    "error", f"Something went wrong while reading {upload.name}. "
                             "It will be tried again.")
