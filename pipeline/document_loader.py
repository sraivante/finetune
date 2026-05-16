"""Parse uploaded documents and split into chunks for Q&A generation."""
from __future__ import annotations

import re
from pathlib import Path


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
    ".markdown": _read_text,
    ".rst": _read_text,
    ".py": _read_text,
    ".json": _read_text,
    ".csv": _read_text,
}


def load_document(path: Path) -> str:
    reader = _READERS.get(path.suffix.lower())
    if not reader:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return _normalise(reader(path))


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple character-window splitter with paragraph awareness."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(overlap, chunk_size - 1))

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= chunk_size:
                buf = para
            else:
                # Hard-split overlong paragraph with overlap.
                step = chunk_size - overlap
                for i in range(0, len(para), step):
                    chunks.append(para[i:i + chunk_size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def load_and_chunk(paths: list[Path], chunk_size: int, overlap: int) -> list[dict]:
    """Return [{source, chunk_index, text}] across all docs."""
    rows: list[dict] = []
    for p in paths:
        try:
            full = load_document(p)
        except Exception as exc:
            rows.append({"source": p.name, "chunk_index": -1,
                         "text": f"[ERROR loading {p.name}: {exc}]"})
            continue
        for idx, chunk in enumerate(chunk_text(full, chunk_size, overlap)):
            rows.append({"source": p.name, "chunk_index": idx, "text": chunk})
    return rows
