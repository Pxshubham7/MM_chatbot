import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    """
    Metadata representation of a curated knowledge document.
    No private user conversations or user uploads are stored here.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    source_uri = Column(String(512), nullable=False)
    category = Column(String(100), nullable=False, default="general", index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_uri": self.source_uri,
            "category": self.category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentChunk(Base):
    """
    Text chunks and corresponding vector embeddings for RAG retrieval.
    Embeddings are serialized as float JSON arrays.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)  # Stores [float, float, ...]
    token_count = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    document = relationship("Document", back_populates="chunks")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "metadata_json": self.metadata_json,
        }
