import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.models import Base
from backend.app.providers.mock import MockEmbeddingProvider
from backend.app.rag.store import VectorStoreAdapter, cosine_similarity


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_cosine_similarity_math():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(vec_a, vec_b), 0.001) == 1.0

    vec_c = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(vec_a, vec_c), 0.001) == 0.0


def test_vector_store_add_and_search(in_memory_db):
    store = VectorStoreAdapter(in_memory_db)
    embedder = MockEmbeddingProvider()

    # Ingest test document
    chunks = [
        {
            "content": "Box breathing involves inhaling for 4 seconds, holding for 4 seconds, and exhaling for 4 seconds.",
            "embedding": embedder.embed_text("Box breathing involves inhaling for 4 seconds and exhaling"),
            "token_count": 18,
            "metadata": {"section": "Breathing"}
        },
        {
            "content": "5-4-3-2-1 sensory grounding technique helps you notice 5 things you see and 4 things you touch.",
            "embedding": embedder.embed_text("5-4-3-2-1 sensory grounding technique notice things you see"),
            "token_count": 17,
            "metadata": {"section": "Grounding"}
        }
    ]

    doc = store.add_document(
        title="Test Coping Guide",
        source_uri="test/coping.md",
        category="wellness",
        chunks=chunks
    )

    assert doc.id is not None
    assert store.count_documents() == 1
    assert store.count_chunks() == 2

    # Query for breathing
    query_emb = embedder.embed_text("How do I do box breathing exercises?")
    results = store.similarity_search(query_emb, top_k=1, threshold=0.3)

    assert len(results) == 1
    chunk, score, found_doc = results[0]
    assert "Box breathing" in chunk.content
    assert found_doc.title == "Test Coping Guide"
    assert score > 0.3
