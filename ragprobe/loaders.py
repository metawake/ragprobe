"""Corpus and query loaders — files, directories, JSON, plain text."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Union


def load_passages_from_directory(path: Union[str, Path],
                                pre_chunked: bool = False,
                                ) -> List[str]:
    """Load text files from a directory.

    If pre_chunked is True, each file is treated as one passage.
    Otherwise, paragraphs (double-newline separated) are split into passages.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")

    passages: List[str] = []
    for fpath in sorted(path.iterdir()):
        if fpath.is_file() and fpath.suffix in (".txt", ".md", ".rst", ".text"):
            text = fpath.read_text(encoding="utf-8", errors="replace")
            if pre_chunked:
                stripped = text.strip()
                if stripped:
                    passages.append(stripped)
            else:
                for para in _split_paragraphs(text):
                    passages.append(para)
    return passages


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines, dropping blanks."""
    parts = text.split("\n\n")
    return [p.strip() for p in parts if p.strip()]


def load_passages_from_strings(texts: List[str]) -> List[str]:
    """Pass-through for the Python API — accept a list of strings."""
    return [t for t in texts if t.strip()]


def load_queries_from_json(path: Union[str, Path]) -> List[str]:
    """Load queries from a JSON file.

    Supports two formats:
      - ragtune format: {"queries": [{"text": "..."}, ...]}
      - simple list:    ["query1", "query2", ...]
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [str(item) for item in data if str(item).strip()]
    if isinstance(data, dict) and "queries" in data:
        return [q["text"] for q in data["queries"] if q.get("text", "").strip()]
    raise ValueError(
        f"Unrecognized JSON format in {path}. "
        "Expected a list of strings or {{\"queries\": [{{\"text\": \"...\"}}]}}."
    )


def load_queries_from_text(path: Union[str, Path]) -> List[str]:
    """Load queries from a plain text file, one per line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_queries(source: Union[str, Path, List[str]]) -> List[str]:
    """Unified query loader — detects format automatically."""
    if isinstance(source, list):
        return [q for q in source if q.strip()]

    path = Path(source)
    if path.suffix == ".json":
        return load_queries_from_json(path)
    return load_queries_from_text(path)


def load_corpus(source: Union[str, Path, List[str]],
                pre_chunked: bool = False) -> List[str]:
    """Unified corpus loader — directory of files or list of strings."""
    if isinstance(source, list):
        return load_passages_from_strings(source)

    path = Path(source)
    if path.is_dir():
        return load_passages_from_directory(path, pre_chunked=pre_chunked)
    if path.is_file() and path.suffix in (".txt", ".md", ".rst", ".text"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if pre_chunked:
            return [text.strip()] if text.strip() else []
        return _split_paragraphs(text)

    raise ValueError(f"Cannot load corpus from: {source}")
