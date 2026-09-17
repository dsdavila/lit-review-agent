"""First-glance academic summaries for every downloaded paper.

For every paper that was actually downloaded (a PDF saved under some
data/{conference}/papers/ - see downloader.py), asks Claude to read the PDF
and produce a short first-glance summary: the big idea, actual
contributions, deficiencies, and any notable methods/findings. Summaries
are cached one JSON file per paper under data/{conference}/summaries/ -
rerunning only calls Claude for papers without a cached summary, unless
--force is given. Once done, writes a single combined markdown doc grouped
by conference (default: data/SUMMARIES.md).

Usage:
    python summarize.py                            # every conference under data/
    python summarize.py --conference cvpr2024       # just one (repeatable)
    python summarize.py --force                     # regenerate cached summaries too
    python summarize.py --concurrency 8 --effort high
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from scrapers.base import PaperMetadata
from storage import load_metadata, load_summary, papers_dir, save_summary, summary_path
from summarizer import ClaudeSummarizer, PaperSummary

logger = logging.getLogger("lit_review_agent")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize every downloaded paper's PDF with Claude and write a combined markdown doc."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument(
        "--conference",
        action="append",
        help="Restrict to this conference folder name (repeatable). Default: every subfolder of "
        "--data-dir with a metadata.json",
    )
    p.add_argument("--model", help="Claude model (default: $ANTHROPIC_MODEL or claude-opus-5)")
    p.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort level (default: medium)",
    )
    p.add_argument("--concurrency", type=int, default=4, help="Parallel Claude calls (default: 4)")
    p.add_argument("--force", action="store_true", help="Regenerate summaries even if already cached")
    p.add_argument("--output", type=Path, default=None, help="Output markdown path (default: <data-dir>/SUMMARIES.md)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def discover_conference_dirs(data_dir: Path, names: Optional[list[str]]) -> list[Path]:
    if names:
        return [data_dir / name for name in names]
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists())


def summarizable_papers(conf_dir: Path) -> list[tuple[PaperMetadata, Path]]:
    """Papers that were actually downloaded (PDF on disk) - the only ones worth summarizing."""
    pdir = papers_dir(conf_dir)
    out = []
    for paper in load_metadata(conf_dir):
        pdf_path = pdir / f"{paper.slug()}.pdf"
        if pdf_path.exists():
            out.append((paper, pdf_path))
    return out


_AUTHOR_INSTITUTION_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")


def _split_author(author: str) -> tuple[str, Optional[str]]:
    """Some scrapers (BMVC) embed affiliation in the author string, e.g. 'Name (Institution)' -
    others (CVPR/ICCV/WACV/ECCV) only ever give a bare name. Split it out where present."""
    m = _AUTHOR_INSTITUTION_RE.match(author)
    if not m:
        return author.strip(), None
    name, institution = m.group(1).strip(), m.group(2).strip()
    return name, (institution or None)


def render_markdown(entries: list[tuple[str, list[tuple[PaperMetadata, PaperSummary]]]]) -> str:
    lines = ["# Literature Summaries", ""]
    total = sum(len(papers) for _, papers in entries)
    lines.append(f"{total} paper(s) across {len(entries)} conference-year(s).")
    lines.append("")
    for conf_name, papers in entries:
        lines.append(f"## {conf_name} ({len(papers)} papers)")
        lines.append("")
        ranked = sorted(papers, key=lambda t: t[0].final_score or 0.0, reverse=True)
        for paper, summary in ranked:
            lines.append(f"### {paper.title}")
            if paper.authors:
                names, institutions = [], []
                for a in paper.authors:
                    name, institution = _split_author(a)
                    names.append(name)
                    if institution and institution not in institutions:
                        institutions.append(institution)
                lines.append(f"*Authors:* {', '.join(names)}")
                if institutions:
                    lines.append(f"*Institutions:* {', '.join(institutions)}")
            venue_year = " ".join(x for x in [paper.venue, str(paper.year) if paper.year else None] if x)
            score_bit = f" · score: {paper.final_score:.1f}" if paper.final_score is not None else ""
            venue_bit = f"{venue_year} · " if venue_year else ""
            lines.append(f"{venue_bit}[paper]({paper.url}){score_bit}")
            lines.append("")
            lines.append(f"**Nutshell:** {summary.nutshell}")
            lines.append("")
            if summary.contributions:
                lines.append("**Contributions:**")
                for c in summary.contributions:
                    lines.append(f"- {c}")
                lines.append("")
            if summary.deficiencies:
                lines.append("**Deficiencies:**")
                for d in summary.deficiencies:
                    lines.append(f"- {d}")
                lines.append("")
            if summary.notable:
                lines.append(f"**Notable:** {summary.notable}")
                lines.append("")
            if summary.results:
                lines.append("**Results:**")
                lines.append("")
                lines.append("| Dataset | Metric | Value | Setting |")
                lines.append("|---|---|---|---|")
                for row in summary.results:
                    setting = row.get("setting", "")
                    lines.append(f"| {row['dataset']} | {row['metric']} | {row['value']} | {setting} |")
                lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


_LEADING_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _parse_numeric(value: str) -> Optional[float]:
    m = _LEADING_NUMBER_RE.search(value)
    return float(m.group()) if m else None


def render_leaderboard(entries: list[tuple[str, list[tuple[PaperMetadata, PaperSummary]]]]) -> str:
    """Groups every paper's extracted results by (dataset, metric) so methods can be racked and
    stacked across the whole corpus - auto-extracted, so treat as a starting point, not ground truth."""
    buckets: dict[tuple[str, str], list[tuple[PaperMetadata, str, str]]] = defaultdict(list)
    for _, papers in entries:
        for paper, summary in papers:
            for row in summary.results:
                key = (row["dataset"].strip(), row["metric"].strip())
                buckets[key].append((paper, row["value"], row.get("setting", "")))

    lines = [
        "# Results Leaderboard",
        "",
        "Headline numbers auto-extracted from each paper's own main results table, grouped by "
        "dataset and metric. This is a starting point for comparison, not ground truth - protocols, "
        "splits, and backbones can differ between rows even within the same dataset/metric; check the "
        "`setting` column and the source paper before citing.",
        "",
    ]
    for (dataset, metric), rows in sorted(buckets.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1].lower())):
        lines.append(f"## {dataset} — {metric}")
        lines.append("")
        lines.append("| Value | Paper | Venue/Year | Setting |")
        lines.append("|---|---|---|---|")

        def sort_key(row: tuple[PaperMetadata, str, str]) -> tuple[bool, float]:
            v = _parse_numeric(row[1])
            return (v is None, -(v if v is not None else 0.0))

        for paper, value, setting in sorted(rows, key=sort_key):
            venue_year = " ".join(x for x in [paper.venue, str(paper.year) if paper.year else None] if x)
            lines.append(f"| {value} | [{paper.title}]({paper.url}) | {venue_year} | {setting} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Root stays at WARNING so the anthropic/httpx client libraries don't dump full request
    # bodies (base64-encoded PDFs included) into the log - only our own logger gets verbose.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    conf_dirs = discover_conference_dirs(args.data_dir, args.conference)
    if not conf_dirs:
        logger.error("No conference folders with metadata.json found under %s", args.data_dir)
        return 1

    summarizer = ClaudeSummarizer(model=args.model, effort=args.effort)

    jobs = []  # (conf_dir, paper, pdf_path, cache_path)
    for conf_dir in conf_dirs:
        if not conf_dir.exists():
            logger.error("No such conference folder: %s", conf_dir)
            continue
        for paper, pdf_path in summarizable_papers(conf_dir):
            jobs.append((conf_dir, paper, pdf_path, summary_path(conf_dir, paper)))

    if not jobs:
        logger.error("No downloaded PDFs found under %s (nothing to summarize)", args.data_dir)
        return 1

    to_generate = [j for j in jobs if args.force or not j[3].exists()]
    cached = len(jobs) - len(to_generate)
    logger.info(
        "%d paper(s) total: %d already cached, %d to summarize via Claude (model=%s, effort=%s)",
        len(jobs), cached, len(to_generate), summarizer.model, args.effort,
    )

    failed = 0
    if to_generate:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            future_to_job = {
                pool.submit(summarizer.summarize, paper, pdf_path): (paper, cache_path)
                for _, paper, pdf_path, cache_path in to_generate
            }
            done = 0
            for future in as_completed(future_to_job):
                paper, cache_path = future_to_job[future]
                done += 1
                try:
                    summary = future.result()
                    save_summary(summary, cache_path)
                    logger.info("[%d/%d] Summarized %r", done, len(to_generate), paper.title)
                except Exception as exc:
                    failed += 1
                    logger.error("[%d/%d] Failed to summarize %r: %s", done, len(to_generate), paper.title, exc)

    # Assemble the combined doc from every cached summary (old + newly written this run)
    entries = []
    for conf_dir in conf_dirs:
        pairs = []
        for paper, _ in summarizable_papers(conf_dir):
            summary = load_summary(summary_path(conf_dir, paper))
            if summary is not None:
                pairs.append((paper, summary))
        if pairs:
            entries.append((conf_dir.name, pairs))

    output_path = args.output or (args.data_dir / "SUMMARIES.md")
    output_path.write_text(render_markdown(entries), encoding="utf-8")

    leaderboard_path = output_path.parent / "LEADERBOARD.md"
    leaderboard_path.write_text(render_leaderboard(entries), encoding="utf-8")

    total_summarized = sum(len(p) for _, p in entries)
    logger.info(
        "Wrote %d summaries to %s and a results leaderboard to %s (%d failed this run)",
        total_summarized, output_path, leaderboard_path, failed,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
