import dataclasses

import pytest

from docqa.models import Chunk


def test_chunk_is_frozen():
    chunk = Chunk(id="h:0", text="hello", filename="a.txt", location="full text", file_hash="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "x"
