"""CLI entry point — requires ``pip install ragprobe[cli]`` for click."""

from __future__ import annotations

import sys

try:
    import click
except ImportError:
    def main() -> None:
        print("ragprobe CLI requires click. Install with: pip install ragprobe[cli]",
              file=sys.stderr)
        sys.exit(1)
else:
    from ragprobe.loaders import load_corpus, load_queries
    from ragprobe.models import DifficultyTier
    from ragprobe.scorer import score_corpus

    @click.command()
    @click.option("--corpus", required=True, type=click.Path(exists=True),
                  help="Path to corpus directory or file.")
    @click.option("--queries", required=True, type=click.Path(exists=True),
                  help="Path to queries file (JSON or text, one per line).")
    @click.option("--format", "output_format", default="text",
                  type=click.Choice(["text", "json"]),
                  help="Output format.")
    @click.option("--compare-references", is_flag=True, default=False,
                  help="Show comparison with built-in reference profiles.")
    @click.option("--ci", is_flag=True, default=False,
                  help="CI mode — exit 1 if difficulty exceeds --max-difficulty.")
    @click.option("--max-difficulty",
                  type=click.Choice(["easy", "medium", "hard"]),
                  default="hard",
                  help="Maximum allowed difficulty tier in CI mode.")
    @click.option("--pre-chunked", is_flag=True, default=False,
                  help="Treat each file as a single passage.")
    def main(corpus: str, queries: str, output_format: str,
             compare_references: bool, ci: bool, max_difficulty: str,
             pre_chunked: bool) -> None:
        """ragprobe — pre-deployment domain difficulty diagnostic for RAG."""
        passages = load_corpus(corpus, pre_chunked=pre_chunked)
        query_list = load_queries(queries)

        if not passages:
            click.echo("Error: no passages found in corpus.", err=True)
            sys.exit(1)
        if not query_list:
            click.echo("Error: no queries found.", err=True)
            sys.exit(1)

        report = score_corpus(passages, query_list,
                              compare_references=compare_references)

        if output_format == "json":
            click.echo(report.to_json())
        else:
            _print_text_report(report, compare_references)

        if ci:
            threshold = DifficultyTier(max_difficulty)
            tier_order = {DifficultyTier.EASY: 0,
                          DifficultyTier.MEDIUM: 1,
                          DifficultyTier.HARD: 2}
            if tier_order[report.difficulty] > tier_order[threshold]:
                click.echo(
                    f"\nCI FAIL: domain difficulty '{report.difficulty.value}' "
                    f"exceeds threshold '{threshold.value}'.",
                    err=True,
                )
                sys.exit(1)

    def _print_text_report(report, compare_references: bool) -> None:
        click.echo("Domain Difficulty Report")
        click.echo("=" * 40)
        click.echo()
        click.echo(f"Overall specificity:  {report.specificity}  "
                   f"({report.difficulty.value.upper()})")

        if compare_references and report.closest_reference:
            click.echo(f"Reference match:      closest to {report.closest_reference}")

        if report.expected_recall_range and report.expected_recall_range != (0.0, 0.0):
            lo = int(report.expected_recall_range[0] * 100)
            hi = int(report.expected_recall_range[1] * 100)
            click.echo(f"Expected Recall@5:    {lo}–{hi}%  "
                       f"(reference value from benchmarks at similar specificity)")

        click.echo()
        counts = report.summary_counts()
        click.echo("Per-query breakdown:")
        if counts["easy"]:
            click.echo(f"  EASY   ({counts['easy']} queries)   specificity > 0.8")
        if counts["medium"]:
            click.echo(f"  MEDIUM ({counts['medium']} queries)   specificity 0.3–0.8")
        if counts["hard"]:
            click.echo(f"  HARD   ({counts['hard']} queries)   specificity < 0.3")

        if report.ambiguous_terms:
            click.echo()
            click.echo("Top ambiguous terms (highest document frequency):")
            terms_display = ", ".join(
                f'"{t}" ({c})' for t, c in report.ambiguous_terms[:10]
            )
            click.echo(f"  {terms_display}")

        if report.recommendations:
            click.echo()
            click.echo("Recommendations:")
            for rec in report.recommendations:
                click.echo(f"  • {rec}")
