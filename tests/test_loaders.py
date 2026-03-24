"""Tests for ragprobe.loaders."""

import json
import os
import tempfile

import pytest

from ragprobe.loaders import (
    load_corpus,
    load_passages_from_directory,
    load_queries,
    load_queries_from_json,
    load_queries_from_text,
)


@pytest.fixture
def tmp_corpus(tmp_path):
    """Create a temporary corpus directory with text files."""
    (tmp_path / "doc1.txt").write_text("First paragraph.\n\nSecond paragraph.")
    (tmp_path / "doc2.txt").write_text("Another document content.")
    (tmp_path / "ignore.csv").write_text("should,be,ignored")
    return tmp_path


@pytest.fixture
def tmp_queries_json(tmp_path):
    """Create a temp queries JSON file in ragtune format."""
    data = {"queries": [{"text": "What is GDPR?"}, {"text": "Data rights"}]}
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def tmp_queries_list_json(tmp_path):
    """Create a temp queries JSON file as a simple list."""
    data = ["What is GDPR?", "Data rights"]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def tmp_queries_text(tmp_path):
    """Create a temp queries text file."""
    path = tmp_path / "queries.txt"
    path.write_text("What is GDPR?\nData rights\n\nController obligations\n")
    return path


class TestLoadPassagesFromDirectory:
    def test_loads_txt_files(self, tmp_corpus):
        passages = load_passages_from_directory(tmp_corpus)
        assert len(passages) == 3  # 2 paragraphs + 1 doc

    def test_ignores_non_text_files(self, tmp_corpus):
        passages = load_passages_from_directory(tmp_corpus)
        assert not any("should,be,ignored" in p for p in passages)

    def test_pre_chunked_mode(self, tmp_corpus):
        passages = load_passages_from_directory(tmp_corpus, pre_chunked=True)
        assert len(passages) == 2  # one per .txt file

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_passages_from_directory("/nonexistent/path")


class TestLoadQueriesFromJson:
    def test_ragtune_format(self, tmp_queries_json):
        queries = load_queries_from_json(tmp_queries_json)
        assert queries == ["What is GDPR?", "Data rights"]

    def test_simple_list_format(self, tmp_queries_list_json):
        queries = load_queries_from_json(tmp_queries_list_json)
        assert queries == ["What is GDPR?", "Data rights"]

    def test_invalid_format_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"something": "else"}))
        with pytest.raises(ValueError):
            load_queries_from_json(path)


class TestLoadQueriesFromText:
    def test_one_per_line(self, tmp_queries_text):
        queries = load_queries_from_text(tmp_queries_text)
        assert len(queries) == 3
        assert "What is GDPR?" in queries
        assert "Controller obligations" in queries

    def test_skips_blank_lines(self, tmp_queries_text):
        queries = load_queries_from_text(tmp_queries_text)
        assert "" not in queries


class TestUnifiedLoaders:
    def test_load_queries_detects_json(self, tmp_queries_json):
        queries = load_queries(tmp_queries_json)
        assert len(queries) == 2

    def test_load_queries_detects_text(self, tmp_queries_text):
        queries = load_queries(tmp_queries_text)
        assert len(queries) == 3

    def test_load_queries_from_list(self):
        queries = load_queries(["q1", "q2", "q3"])
        assert queries == ["q1", "q2", "q3"]

    def test_load_corpus_from_dir(self, tmp_corpus):
        passages = load_corpus(tmp_corpus)
        assert len(passages) >= 2

    def test_load_corpus_from_list(self):
        passages = load_corpus(["passage one", "passage two"])
        assert passages == ["passage one", "passage two"]
