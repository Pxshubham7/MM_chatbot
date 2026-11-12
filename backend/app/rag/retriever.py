import logging
from typing import List, Tuple
from sqlalchemy.orm import Session
from backend.app.config import get_settings
from backend.app.db.models import Document, DocumentChunk
from backend.app.providers.base import EmbeddingProvider
from backend.app.rag.store import VectorStoreAdapter
from backend.app.schemas import CitationItem

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    Retrieval-Augmented Generation query orchestrator.
    Embeds query, computes similarity application-side against MySQL chunks,
    and returns citations and formatted context.
    """

    def __init__(self, db: Session, embedding_provider: EmbeddingProvider):
        self.db = db
        self.embedding_provider = embedding_provider
        self.store = VectorStoreAdapter(db)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> Tuple[List[Tuple[DocumentChunk, float, Document]], List[CitationItem]]:
        """
        Embeds the search query and retrieves the highest-similarity chunks above threshold.
        Returns matching chunks tuple list and structured CitationItem list.
        """
        k = top_k or settings.rag_top_k
        min_sim = threshold if threshold is not None else settings.rag_similarity_threshold

        query_embedding = self.embedding_provider.embed_text(query)
        matches = self.store.similarity_search(query_embedding, top_k=k, threshold=min_sim)

        citations: List[CitationItem] = []
        for chunk, score, doc in matches:
            # Create a concise excerpt (first 250 characters)
            excerpt = chunk.content.strip().replace("\n", " ")
            if len(excerpt) > 250:
                excerpt = excerpt[:247] + "..."

            citations.append(CitationItem(
                document_title=doc.title,
                source_uri=doc.source_uri,
                category=doc.category,
                chunk_index=chunk.chunk_index,
                content_excerpt=excerpt,
                similarity_score=round(score, 4)
            ))

        return matches, citations
