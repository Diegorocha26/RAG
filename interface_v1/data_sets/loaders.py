"""
loaders.py
----------
Reads files from the knowledge base into plain text.
Supports: .md, .txt, .pdf, .csv, .docx
Add new loaders by registering a function in LOADERS dict at the bottom.
"""

# TODO: edge-case: divide extremely big files (ex. multi-page pdfs, excels, etc.) into parts so that they don't take out the entire context window of the model

import os
import csv
from pathlib import Path


# ── Individual loaders ──────────────────────────────────────────────────────────

def load_text(path: Path) -> str:
    """Load plain text or markdown files."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_csv(path: Path) -> str:
    # TODO: check is this is the best way of feeding a csv to an LLM. 
    #       cosnider other other alternatives such as marksown, html tables, etc. instead of " | "
    """Convert CSV to a readable text table."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(" | ".join(row))
    return "\n".join(rows)


def load_pdf(path: Path) -> str:
    """
    Extract text from PDF using pypdf (pure Python, no binary deps).
    Falls back to a placeholder if pypdf is not installed.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "pypdf is required to load PDF files. Install it with:\n"
            "  uv add pypdf"
        )


def load_docx(path: Path) -> str:
    """
    Extract text from .docx using python-docx.
    Falls back gracefully if not installed.
    """
    try:
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError(
            "python-docx is required to load .docx files. Install it with:\n"
            "  uv add python-docx"
        )


# ── Registry ────────────────────────────────────────────────────────────────────

LOADERS = {
    ".md":   load_text,
    ".txt":  load_text,
    ".pdf":  load_pdf,
    ".csv":  load_csv,
    ".docx": load_docx,
}


# ── Public API ──────────────────────────────────────────────────────────────────

def load_file(path: Path) -> str:
    """
    Load a single file and return its text content, truncated to max_chars.
    Raises ValueError for unsupported extensions.
    """
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file extension '{ext}' for file: {path}\n"
            f"Supported: {list(LOADERS.keys())}"
        )
    return LOADERS[ext](path)


def discover_files(kb_path: str, supported_extensions: list[str]) -> list[Path]:
    """
    Walk kb_path (file or directory) and return all supported files.
    Ignores hidden files and __pycache__ directories.
    """
    root = Path(kb_path)
    exts = {e.lower() for e in supported_extensions}

    if root.is_file():
        if root.suffix.lower() in exts:
            return [root]
        raise ValueError(f"File {root} has unsupported extension '{root.suffix}'")

    if not root.is_dir():
        raise ValueError(f"Path does not exist: {root}")

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and pycache
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for filename in filenames:
            p = Path(dirpath) / filename
            if p.suffix.lower() in exts and not filename.startswith("."):
                files.append(p)

    return sorted(files)
