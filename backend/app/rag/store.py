import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from backend.app.db.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors using NumPy."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStoreAdapter:
    """
    Adapter for MySQL 8.0 vector storage and application-side similarity search.
    Embeddings are serialized as JSON arrays and queried into memory for fast dot-product
    and cosine similarity computation over curated document chunks.
    """

    def __init__(self, db: Session):
        self.db = db

    def add_document(
        self,
        title: str,
        source_uri: str,
        category: str,
        chunks: List[Dict[str, Any]]
    ) -> Document:
        """
        Idempotently inserts or updates a curated document and its embedded chunks.
        Existing records matching source_uri are removed before re-ingestion.
        """
        # Remove any existing document with the same source URI for idempotency
        existing = self.db.query(Document).filter(Document.source_uri == source_uri).first()
        if existing:
            self.db.delete(existing)
            self.db.commit()

        doc = Document(
            title=title,
            source_uri=source_uri,
            category=category
        )
        self.db.add(doc)
        self.db.flush()  # Populates doc.id

        for idx, chunk_data in enumerate(chunks):
            embedding = chunk_data["embedding"]
            # Ensure embedding is stored as Python list or JSON-serializable list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                content=chunk_data["content"],
                embedding=embedding,
                token_count=chunk_data.get("token_count", len(chunk_data["content"].split())),
                metadata_json=chunk_data.get("metadata", {})
            )
            self.db.add(chunk)

        self.db.commit()
        self.db.refresh(doc)
        return doc

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.60
    ) -> List[Tuple[DocumentChunk, float, Document]]:
        """
        Performs application-side cosine similarity search against all stored chunks.
        Returns sorted list of (chunk, similarity_score, document) tuples exceeding threshold.
        """
        chunks = self.db.query(DocumentChunk).join(Document).all()
        if not chunks:
            return []

        results: List[Tuple[DocumentChunk, float, Document]] = []

        for chunk in chunks:
            emb = chunk.embedding
            if isinstance(emb, str):
                emb = json.loads(emb)

            score = cosine_similarity(query_embedding, emb)
            if score >= threshold:
                results.append((chunk, score, chunk.document))

        # Sort descending by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count_documents(self) -> int:
        """Returns total number of ingested curated documents."""
        return self.db.query(Document).count()

    def count_chunks(self) -> int:
        """Returns total number of text chunks."""
        return self.db.query(DocumentChunk).count()

    def clear_all(self) -> None:
        """Deletes all documents and chunks (used in testing)."""
        self.db.query(DocumentChunk).delete()
        self.db.query(Document).delete()
        self.db.commit()
