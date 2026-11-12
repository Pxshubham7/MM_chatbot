"""
Database initialization CLI script.
Creates MySQL 8.0 schema tables for curated RAG documents idempotently.
"""
import sys
from backend.app.db.session import init_db, get_db_status


def main():
    print("Checking database connection...")
    status = get_db_status()
    print(f"Database Target: {status.get('type')}")
    print(f"Connected: {status.get('connected')}")
    if not status.get("connected"):
        print(f"Error: {status.get('error')}")
        sys.exit(1)

    print("Initializing database tables (documents, document_chunks)...")
    init_db()
    print("Database tables created successfully!")


if __name__ == "__main__":
    main()
