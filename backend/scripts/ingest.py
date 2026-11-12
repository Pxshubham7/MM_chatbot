"""
Local Curated Document Ingestion CLI Script.
Loads local curated markdown/text wellness documents into MySQL 8.0 RAG tables.
No administrative web portal required.
"""
import argparse
import os
import sys
from pathlib import Path
from backend.app.db.session import get_db, init_db, get_db_status
from backend.app.providers.factory import ProviderFactory
from backend.app.rag.chunker import TextChunker
from backend.app.rag.store import VectorStoreAdapter


def ingest_directory(docs_dir: str):
    path = Path(docs_dir)
    if not path.exists() or not path.is_dir():
        print(f"Error: Directory '{docs_dir}' does not exist.")
        sys.exit(1)

    print(f"Initializing database and verifying tables...")
    init_db()

    embedding_provider = ProviderFactory.get_embedding_provider()
    chunker = TextChunker(chunk_size=300, chunk_overlap=40)

    db_gen = get_db()
    db = next(db_gen)

    try:
        store = VectorStoreAdapter(db)
        files = list(path.glob("*.md")) + list(path.glob("*.txt"))

        if not files:
            print(f"No .md or .txt files found in '{docs_dir}'.")
            return

        print(f"Found {len(files)} document(s) to ingest in '{docs_dir}':")
        total_chunks = 0

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Determine title from first markdown header or filename
            lines = content.strip().splitlines()
            title = file_path.stem.replace("_", " ").title()
            for line in lines:
                if line.startswith("# "):
                    title = line.replace("# ", "").strip()
                    break

            category = "general"
            if "grounding" in file_path.stem.lower():
                category = "grounding"
            elif "breathing" in file_path.stem.lower():
                category = "breathing"
            elif "sleep" in file_path.stem.lower():
                category = "sleep_hygiene"

            # Chunk document
            chunks = chunker.chunk_markdown(content, source_uri=str(file_path))
            if not chunks:
                continue

            # Embed chunks
            chunk_texts = [c["content"] for c in chunks]
            embeddings = embedding_provider.embed_batch(chunk_texts)

            for chunk, emb in zip(chunks, embeddings):
                chunk["embedding"] = emb

            doc = store.add_document(
                title=title,
                source_uri=str(file_path.as_posix()),
                category=category,
                chunks=chunks
            )

            total_chunks += len(chunks)
            print(f"  Ingested '{title}' ({category}): {len(chunks)} chunks -> doc_id={doc.id}")

        print(f"\nIngestion Complete!")
        print(f"Total documents in database: {store.count_documents()}")
        print(f"Total chunks in database: {store.count_chunks()}")

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest curated documents into RAG database.")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/sample_docs",
        help="Path to directory containing curated markdown/text documents (default: data/sample_docs)"
    )
    args = parser.parse_args()
    ingest_directory(args.dir)


if __name__ == "__main__":
    main()
