import os
import pytest

# Ensure tests use local database and mock providers
os.environ["DATABASE_URL"] = "sqlite:///./mm_chatbot_rag.db"
os.environ["ALLOW_SQLITE_FALLBACK"] = "true"
os.environ["PROVIDER_MODE"] = "mock"
os.environ["APP_ENV"] = "test"


@pytest.fixture(autouse=True)
def setup_test_db():
    from backend.app.db.session import init_db
    init_db()
    yield
