"""Chroma-backed vector store with local embeddings (FR-5)."""

import uuid

import chromadb

from docqa.models import Chunk, Hit


class VectorStore:
    """An in-memory Chroma collection. Each store has its own collection."""

    def __init__(self, collection_name: str | None = None, embedding_function=None):
        """embedding_function=None uses Chroma's built-in local model (production).

        Tests pass a deterministic offline function instead.
        """
        self._client = chromadb.EphemeralClient()
        name = collection_name or f"docqa-{uuid.uuid4().hex}"
        options = {}
        if embedding_function is not None:
            options["embedding_function"] = embedding_function
        # Cosine distance, so score = 1 - distance is a similarity.
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}, **options
        )

    def add(self, chunks: list[Chunk]) -> int:
        """Store chunks whose id is not already present. Returns how many were added."""
        unique = {chunk.id: chunk for chunk in chunks}
        if not unique:
            return 0
        present = set(self._collection.get(ids=list(unique))["ids"])
        new = [chunk for chunk_id, chunk in unique.items() if chunk_id not in present]
        if not new:
            return 0
        self._collection.add(
            ids=[chunk.id for chunk in new],
            documents=[chunk.text for chunk in new],
            metadatas=[
                {"filename": c.filename, "location": c.location, "file_hash": c.file_hash}
                for c in new
            ],
        )
        return len(new)

    def count(self) -> int:
        return self._collection.count()

    def query(self, text: str, k: int = 5) -> list[Hit]:
        """Return up to k hits, best first. An empty store returns []."""
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(query_texts=[text], n_results=min(k, total))
        hits = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = Chunk(
                id=chunk_id,
                text=document,
                filename=metadata["filename"],
                location=metadata["location"],
                file_hash=metadata["file_hash"],
            )
            hits.append(Hit(chunk=chunk, score=1.0 - distance))
        return hits
