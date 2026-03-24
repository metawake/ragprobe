# ragprobe

**Pre-deployment domain difficulty diagnostic for RAG. Know if your benchmark transfers — before you deploy.**

[![PyPI version](https://img.shields.io/pypi/v/ragprobe.svg)](https://pypi.org/project/ragprobe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The problem

RAG pipelines get benchmarked on one domain and deployed on another. A system that scores 95% recall on financial documents scores 28% on legal text — same architecture, same embedding model, same settings. The failure is invisible until production users report wrong answers.

There is no standard way to predict, before deploying, whether your benchmark results will hold on your actual domain.

## The fix

Measure the domain, not the pipeline. **Vocabulary specificity** — how uniquely query terms identify target passages — predicts retrieval difficulty in seconds, without running a single embedding. A corpus where query terms appear in 5 passages is trivially retrievable. A corpus where query terms appear in 500 passages will defeat any embedding model.

ragprobe quantifies this gap before you deploy, not after.

```bash
pip install ragprobe
```

## Quick start

```bash
ragprobe score --corpus ./docs --queries queries.json
```

```
Domain Difficulty Report
========================

Overall specificity:  0.177  (HARD)
Reference match:      closest to GDPR regulatory text
                      Expect NeedleCoverage@5 in 15-30% range.

                      WARNING: If your benchmarks used HotpotQA (0.95)
                      or FinanceBench (0.98), results will NOT transfer.

Per-query breakdown:
  EASY  (3 queries)   specificity > 0.7
  HARD  (17 queries)  specificity < 0.3

Top ambiguous terms (appear in 100+ passages):
  "data" (838), "processing" (412), "controller" (389),
  "subject" (301), "personal" (287)
```

No embeddings. No vector store. No API keys. Runs in seconds.

## Python API

```python
from ragprobe import DomainProbe

probe = DomainProbe(
    corpus=["path/to/docs/"],
    queries=["What are the controller's obligations?", "What fines apply?"],
)
report = probe.score()

print(report.specificity)              # 0.177
print(report.difficulty)               # "hard"
print(report.closest_reference)        # "GDPR regulatory text"
print(report.expected_recall_range)    # (0.15, 0.35) — reference value
print(report.hardest_queries[:3])      # per-query breakdown
print(report.ambiguous_terms[:5])      # terms causing collisions
```

## What the scores mean

| Specificity | Difficulty | Expected Recall@5 | What to expect |
|-------------|------------|-------------------|----------------|
| > 0.8 | **Easy** | 85–99% | Your benchmark probably transfers. Any decent embedder will work. |
| 0.3 – 0.8 | **Medium** | 35–80% | Benchmark may partially transfer. Build 10-20 domain-specific test queries before deploying. |
| < 0.3 | **Hard** | 5–35% | Your benchmark is lying to you. Build domain-specific needle annotations. Don't deploy without domain-specific evaluation. |

*Expected Recall@5 ranges are reference values from measured benchmarks at similar specificity levels, not predictions for your specific corpus.*

## Built-in reference profiles

ragprobe ships with measured profiles from real corpora so you can see where your domain sits:

| Domain | Specificity | Difficulty | Recall@5 | Source |
|--------|-------------|------------|----------|--------|
| CaseHOLD (legal holdings) | 0.985 | Easy | 90–99% | Case names, statute numbers act as unique identifiers |
| HotpotQA (Wikipedia) | 0.946 | Easy | 85–95% | Named entities, dates, specific facts |
| Financial (SEC filings) | ~0.95 | Easy | 85–95% | Company names, dates, figures |
| Technical docs (product-specific) | 0.70–0.90 | Medium | 55–80% | Product terms provide moderate specificity |
| Medical (clinical) | 0.40–0.70 | Medium | 35–60% | Clinical terminology helps but overlaps |
| GDPR (regulatory) | 0.177 | Hard | 15–35% | Generic vocabulary: "data," "processing," "controller" everywhere |
| RFC (technical standards) | 0.024 | Hard | 5–20% | "client," "server," "request," "response" in every passage |

```bash
ragprobe score --corpus ./my-docs --queries my-queries.json --compare-references
```

## Try it on example corpora

ragprobe ships with three sample domains so you can see the tiers in action immediately:

```bash
# Easy domain — Wikipedia factoid passages (specificity ~0.97)
ragprobe score --corpus examples/wiki/corpus --queries examples/wiki/queries.txt --compare-references

# Hard domain — GDPR regulatory text (specificity ~0.28)
ragprobe score --corpus examples/regulatory/corpus --queries examples/regulatory/queries.txt --compare-references

# Medium domain — technical product docs (specificity ~0.70)
ragprobe score --corpus examples/technical/corpus --queries examples/technical/queries.txt --compare-references
```

## When ragprobe is useful

- **Before choosing a benchmark.** "Should I trust my HotpotQA results for this legal corpus?" Run ragprobe. 5 seconds. Answer: no.
- **Before deploying to a new domain.** You built a RAG system on product docs (medium difficulty). Now the team wants to add compliance policies (hard). How much will retrieval degrade? Measure it.
- **In CI/CD.** Gate deployment if domain difficulty exceeds a threshold without domain-specific evaluation queries.
- **For mixed corpora.** Real knowledge bases aren't single-domain. ragprobe tells you which parts of your corpus are easy and which will fail silently.

```bash
ragprobe score --corpus ./docs --queries queries.json --ci --max-difficulty hard
```

## When ragprobe is NOT useful

- **Single well-known domain.** If you know you're deploying on GDPR and you've already built domain-specific evaluation, you don't need ragprobe to tell you it's hard.
- **Predicting exact recall numbers.** ragprobe predicts a difficulty *tier*, not a precise metric. It tells you "this is hard" not "you'll get 23.7% recall."
- **Comparing retrieval architectures.** ragprobe measures domain difficulty, not pipeline quality. Use [ragtune](https://github.com/metawake/ragtune) for retrieval benchmarking.

## How it works

1. **Tokenize** each query into non-stopword terms
2. **Build an inverted index** of the corpus (which terms appear in which passages)
3. **Compute specificity** per query: fraction of terms appearing in fewer than 5 passages
4. **Compute IDF statistics**: average and max inverse document frequency per query
5. **Identify ambiguous terms**: terms with highest document frequency
6. **Compare** against built-in reference profiles
7. **Report** difficulty tier, per-query breakdown, and actionable recommendations

The core insight: if the text is lexically ambiguous (many passages share the same vocabulary), no retrieval method — keyword, dense, or hybrid — will have an easy time. Embeddings compress text into vectors; they don't invent semantic distinctions that aren't in the text. ragprobe measures a **difficulty floor** that applies regardless of architecture.

## CLI reference

```bash
# Score a corpus against queries
ragprobe score --corpus ./docs --queries queries.json

# JSON output for CI/CD
ragprobe score --corpus ./docs --queries queries.json --format json

# Compare against built-in reference profiles
ragprobe score --corpus ./docs --queries queries.json --compare-references

# CI mode: exit 1 if difficulty exceeds threshold without domain-specific eval
ragprobe score --corpus ./docs --queries queries.json --ci --max-difficulty hard

# Score pre-chunked text (one file per chunk)
ragprobe score --corpus ./chunks/ --queries queries.json --pre-chunked

# Read queries from a plain text file (one per line)
ragprobe score --corpus ./docs --queries questions.txt
```

## Part of the RAG measurement ecosystem

ragprobe is one of three independent tools for RAG retrieval quality:

| Tool | Layer | Question it answers |
|------|-------|-------------------|
| [chunkweaver](https://github.com/metawake/chunkweaver) | Ingestion | "Are my chunks structurally coherent?" |
| [ragtune](https://github.com/metawake/ragtune) | Evaluation | "How does my retrieval actually perform?" |
| **ragprobe** | Pre-deployment | "Will my benchmark results transfer to this domain?" |

They compose through standard formats (text files, JSON), not shared dependencies:

```bash
chunkweaver legal_doc.txt --preset legal-eu --format jsonl > chunks.jsonl
ragprobe score --corpus ./chunks/ --queries queries.json --pre-chunked
ragtune ingest ./chunks/ --collection test --pre-chunked
ragtune simulate --collection test --queries queries.json
```

## Research

Vocabulary specificity is a pre-retrieval difficulty metric rooted in Query Performance Prediction (QPP), an established area of information retrieval research. The core insight — that retrieval difficulty is predictable from corpus statistics before any embedding is computed — has been validated across TREC benchmarks since the early 2000s.

Key references:
- Hauff, Hiemstra & de Jong, ["A Survey of Pre-Retrieval Query Performance Predictors"](https://djoerdhiemstra.com/2008/a-survey-of-pre-retrieval-query-performance-predictors/) (CIKM 2008) — foundational survey of pre-retrieval QPP methods
- Thakur et al., ["BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"](https://github.com/beir-cellar/beir) (NeurIPS 2021) — demonstrates the benchmark transfer problem across 18 domains
- Alexapolsky, ["Your Benchmark Doesn't Generalize"](https://medium.com/@TheWake) — cross-domain RAG experiments showing vocabulary specificity predicts NeedleCoverage collapse from 95% to 28%

ragprobe is, to our knowledge, the first pip-installable tool that makes pre-retrieval domain difficulty metrics accessible to RAG practitioners. The QPP research community produced 20 years of validated metrics; ragprobe packages the most actionable ones for modern retrieval pipelines.

## Architecture

```
ragprobe/
├── __init__.py      # Public API: DomainProbe
├── scorer.py        # Core: inverted index, specificity, IDF, difficulty tiers
├── models.py        # DomainReport, QueryDifficulty dataclasses
├── profiles.py      # Built-in reference profiles (GDPR, RFC, HotpotQA, etc.)
├── loaders.py       # Corpus and query loaders (files, JSON, directories)
└── cli.py           # CLI entry point
```

**Design principles:**
- Zero dependencies for core — stdlib only, no heavy ML frameworks
- CLI requires only `click` (`pip install ragprobe[cli]`)
- All scores are deterministic and reproducible
- JSON output for CI/CD integration

## Limitations

- **Lexical only.** ragprobe measures word-level specificity, not semantic similarity. Two passages with identical vocabulary but different meaning (e.g., "shall erase" vs "may erase") will appear equally specific. This means ragprobe predicts a difficulty floor — actual retrieval may perform slightly better with strong embedding models.
- **Correlation, not causation.** Vocabulary specificity correlates with retrieval difficulty (validated on GDPR, RFC, HotpotQA, CaseHOLD) but is one of several factors. Answer dispersion (how many passages contain the answer) and semantic role diversity also matter.
- **English-centric stopword list.** The default stopword list is English. For other languages, pass a custom stopword set.

## License

MIT

## Author

[Oleksii Alexapolsky](https://github.com/metawake) — Senior Python & Applied AI Engineer. Building measurement tools for RAG retrieval: [ragtune](https://github.com/metawake/ragtune), [chunkweaver](https://github.com/metawake/chunkweaver), ragprobe. Writing about what actually works at [medium.com/@TheWake](https://medium.com/@TheWake).
