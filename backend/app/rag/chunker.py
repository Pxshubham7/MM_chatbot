import re
from typing import Any, Dict, List


class TextChunker:
    """
    Splits text documents into semantically coherent chunks based on
    markdown headers and paragraph boundaries with configurable chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # In approximate words
        self.chunk_overlap = chunk_overlap

    def chunk_markdown(self, text: str, source_uri: str) -> List[Dict[str, Any]]:
        """
        Splits markdown content, preserving section headers in metadata.
        """
        # Split by markdown headers (#, ##, ###)
        sections = re.split(r"(?m)^(?=#{1,3}\s+)", text)
        chunks: List[Dict[str, Any]] = []

        for section in sections:
            section_str = section.strip()
            if not section_str:
                continue

            # Extract header if present
            header_match = re.match(r"^(#{1,3})\s+(.+)$", section_str, re.MULTILINE)
            section_title = header_match.group(2).strip() if header_match else "General"

            # Split words
            words = section_str.split()
            if len(words) <= self.chunk_size:
                chunks.append({
                    "content": section_str,
                    "token_count": len(words),
                    "metadata": {
                        "section_title": section_title,
                        "source_uri": source_uri
                    }
                })
            else:
                # Sliding window
                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < len(words):
                    chunk_words = words[start: start + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    chunks.append({
                        "content": chunk_text,
                        "token_count": len(chunk_words),
                        "metadata": {
                            "section_title": section_title,
                            "source_uri": source_uri,
                            "offset": start
                        }
                    })
                    start += step

        return chunks
