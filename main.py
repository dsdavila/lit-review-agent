"""Metadata-First literature review agent.

Single conference:
    python main.py --url [REPO_URL] --query "your research topic"

Batch, across a flat list of conference repos (each gets its own folder):
    python main.py --config conferences.json
    # conferences.json: {"query": "...", "search_terms": ["...", "..."],
    #                     "conferences": [{"name": "...", "url": "..."}, ...]}
    # see examples/conferences.example.json

Flow per conference:
    1. Fetch index -> save metadata.json                            (Scraper Layer)
    2. Prefilter ALL papers by case-insensitive search-term match    (Filter Layer, stage 1)
       (deterministic string matching by default; --prefilter-method
       embeddings opts into semantic cosine-similarity search instead)
    3. Claude (the LLM adjudicator) judges only the shortlist        (Filter Layer, stage 2)
    4. Print sorted candidates, prompt to confirm                    (user gate)
    5. Selective download of only confirmed PDFs                     (Selective Downloader)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from downloader import download_pdf
from evaluator import Evaluator
from logging_config import configure_logging
from scrapers import get_scraper_by_name, get_scraper_for_url
from scrapers.base import BaseScraper, PaperMetadata
from scrapers.generic_html import GenericHTMLScraper
from storage import conference_dir, load_metadata, papers_dir, save_metadata, write_run_summary

import logging

logger = logging.getLogger("lit_review_agent")


@dataclass
class ConferenceJob:
    url: str
    name: Optional[str] = None
    scraper: Optional[str] = None
    selectors: Optional[Path] = None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Metadata-first literature review agent: scrape metadata, filter with a "
        "local prefilter + Claude judge, and download only the PDFs that actually matter.",
    )
    p.add_argument("--url", help="Single conference index/listing URL to scrape (one-off run)")
    p.add_argument("--config", type=Path, help="JSON config with a flat list of {name, url} conference "
                   "entries to batch-process - see examples/conferences.example.json")
    p.add_argument("--query", help="Research topic to score papers against (required unless set in --config)")
    p.add_argument("--conference", help="Folder name for --url mode (default: derived from the scraper)")
    p.add_argument("--scraper", help="Force a specific scraper by name instead of auto-detecting from the URL")
    p.add_argument("--selectors", type=Path, help="JSON selector config file (for --scraper generic_html)")
    p.add_argument("--js", action="store_true", help="Render pages with Playwright before scraping (generic_html only)")

    p.add_argument("--prefilter-method", choices=["strings", "embeddings"], default="strings",
                    help="Stage-1 local filter method (default: strings - case-insensitive search-term "
                    "matching; embeddings opts into semantic cosine-similarity search instead)")
    p.add_argument("--search-terms", help='Comma-separated, case-insensitive strings/phrases to match in '
                    'title/abstract, e.g. "sparse routing,mixture of experts". Required for the default '
                    '--prefilter-method strings (or set `search_terms` in --config).')
    p.add_argument("--search-fields", default="title,abstract",
                    help="Comma-separated fields to search for --prefilter-method strings (default: title,abstract)")
    p.add_argument("--match-all", action="store_true",
                    help="Require ALL --search-terms to match, not just any one of them")
    p.add_argument("--prefilter-k", type=int, default=40,
                    help="How many top prefilter candidates to send to the Claude judge (default: 40)")
    p.add_argument("--min-prefilter-score", type=float, default=None,
                    help="Also require this minimum prefilter score to reach the judge stage "
                    "(match count for --prefilter-method strings, 0-100 cosine similarity for embeddings)")
    p.add_argument("--no-judge", action="store_true",
                    help="Skip the Claude judge stage entirely - rank on the local prefilter score only")
    p.add_argument("--claude-model", help="Claude model for the judge stage (default: $ANTHROPIC_MODEL or claude-opus-5)")

    p.add_argument("--top-n", type=int, default=10, help="Number of top-ranked papers to propose downloading (default: 10)")
    p.add_argument("--limit", type=int, default=200,
                    help="Max metadata records to fetch per conference (default: 200; 0 = no limit, fetch everything)")
    p.add_argument("--titles-only", action="store_true",
                    help="Skip per-paper abstract fetching (CVF/ECVA/ACL Anthology/BMVC 2025+/generic_html "
                    "abstract_page all need one extra request per paper for it) - much faster scraping at "
                    "the cost of empty abstracts for prefilter/judge scoring")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root output directory (default: ./data)")
    p.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt; download the top candidates automatically")
    p.add_argument("--skip-scrape", action="store_true", help="Reuse an existing metadata.json instead of re-scraping")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose console logging")
    return p.parse_args(argv)


def parse_search_terms(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    return terms or None


def resolve_jobs(args: argparse.Namespace) -> tuple[str, Optional[list[str]], list[ConferenceJob]]:
    cli_search_terms = parse_search_terms(args.search_terms)

    if args.config:
        cfg: dict[str, Any] = json.loads(args.config.read_text())
        query = args.query or cfg.get("query")
        if not query:
            raise SystemExit("No --query given and none found in --config's `query` field.")
        search_terms = cli_search_terms or cfg.get("search_terms")
        jobs = []
        for entry in cfg.get("conferences", []):
            selectors = entry.get("selectors")
            jobs.append(
                ConferenceJob(
                    url=entry["url"],
                    name=entry.get("name"),
                    scraper=entry.get("scraper"),
                    selectors=Path(selectors) if selectors else None,
                )
            )
        if not jobs:
            raise SystemExit(f"--config {args.config} has no `conferences` entries.")
        return query, search_terms, jobs

    if not args.url:
        raise SystemExit("Provide either --url (single conference) or --config (batch of conferences).")
    if not args.query:
        raise SystemExit("--query is required.")
    jobs = [ConferenceJob(url=args.url, name=args.conference, scraper=args.scraper, selectors=args.selectors)]
    return args.query, cli_search_terms, jobs


def resolve_scraper(job: ConferenceJob, js: bool) -> BaseScraper:
    if job.scraper == "generic_html" or (job.scraper is None and job.selectors):
        selectors = json.loads(job.selectors.read_text()) if job.selectors else {}
        return GenericHTMLScraper(selectors=selectors, js=js)
    if job.scraper:
        return get_scraper_by_name(job.scraper)
    return get_scraper_for_url(job.url)


def print_candidates(papers: list[PaperMetadata], highlight_n: int) -> None:
    print(f"\n{'#':>3}  {'score':>6}  {'stage':>7}  {'year':>4}  title")
    print("-" * 100)
    for i, paper in enumerate(papers, start=1):
        marker = "*" if i <= highlight_n else " "
        year = paper.year or "?"
        stage = "judge" if paper.judge_score is not None else "prefilt"
        score = paper.final_score or 0.0
        title = paper.title[:70]
        print(f"{marker}{i:>2}  {score:6.1f}  {stage:>7}  {year:>4}  {title}")
    print("-" * 100)
    print(f"* = top {highlight_n} candidates proposed for download\n")


def confirm_selection(papers: list[PaperMetadata], top_n: int, auto_yes: bool) -> list[PaperMetadata]:
    default_selection = papers[:top_n]
    if auto_yes:
        return default_selection

    print_candidates(papers, top_n)
    answer = input(
        f"Download the top {min(top_n, len(papers))} papers above? "
        "[Y]es / [n]o / comma-separated ranks to download instead: "
    ).strip().lower()
    if answer in ("", "y", "yes"):
        return default_selection
    if answer in ("n", "no"):
        return []
    try:
        ranks = [int(x) for x in answer.replace(" ", "").split(",") if x]
        return [papers[r - 1] for r in ranks if 1 <= r <= len(papers)]
    except ValueError:
        print("Could not parse that response - downloading nothing.")
        return []


def process_conference(job: ConferenceJob, query: str, evaluator: Evaluator, args: argparse.Namespace) -> dict[str, Any]:
    scraper = resolve_scraper(job, js=args.js)
    conference_name = job.name or scraper.name
    conf_dir = conference_dir(args.data_dir, conference_name)
    configure_logging(conf_dir, verbose=args.verbose)

    logger.info("=== %s (%s) ===", conference_name, job.url)

    if args.skip_scrape:
        papers = load_metadata(conf_dir)
        if not papers:
            logger.error("--skip-scrape given but no existing metadata.json found at %s", conf_dir)
            return {"name": conference_name, "url": job.url, "error": "no cached metadata.json"}
        logger.info("Loaded %d cached papers from %s", len(papers), conf_dir / "metadata.json")
    else:
        effective_limit = None if args.limit == 0 else args.limit
        logger.info(
            "Fetching metadata using scraper=%s (limit=%s, titles_only=%s) ...",
            scraper.name,
            effective_limit if effective_limit is not None else "none",
            args.titles_only,
        )
        try:
            papers = scraper.fetch_metadata(job.url, limit=effective_limit, fetch_abstracts=not args.titles_only)
        except Exception:
            logger.exception("Scraping failed for %s", job.url)
            return {"name": conference_name, "url": job.url, "error": "scrape failed"}
        if not papers:
            logger.error("No papers found for %s - nothing to score or download.", job.url)
            return {"name": conference_name, "url": job.url, "error": "no papers found while scraping"}
        path = save_metadata(papers, conf_dir)
        logger.info("Saved metadata for %d papers to %s", len(papers), path)

    ranked = evaluator.run(query, papers)
    save_metadata(ranked, conf_dir)  # persist prefilter/judge scores back into metadata.json (every paper, incl. rejects)

    # Only papers that actually survived the prefilter are real download candidates -
    # never let --top-n pad the selection with never-shortlisted papers just to fill the count.
    candidates = [p for p in ranked if p.shortlisted]
    if not candidates:
        logger.info("No papers matched the prefilter - nothing to download.")
        logger.info("Full detail on rejections/failures: %s", conf_dir / "filtering.log")
        return {
            "name": conference_name,
            "url": job.url,
            "scraped": len(ranked),
            "shortlisted": 0,
            "downloaded": 0,
            "top_matches": [],
        }

    selected = confirm_selection(candidates, args.top_n, args.yes)
    downloaded = 0
    if selected:
        out_dir = papers_dir(conf_dir)
        logger.info("Downloading %d selected PDFs to %s ...", len(selected), out_dir)
        for paper in selected:
            result = download_pdf(paper, out_dir)
            downloaded += int(result.ok)
        logger.info("Downloaded %d/%d selected papers successfully.", downloaded, len(selected))
    else:
        logger.info("No papers selected for download.")
    logger.info("Full detail on rejections/failures: %s", conf_dir / "filtering.log")

    return {
        "name": conference_name,
        "url": job.url,
        "scraped": len(ranked),
        "shortlisted": sum(1 for p in ranked if p.shortlisted),
        "downloaded": downloaded,
        "top_matches": [
            {"title": p.title, "score": p.final_score, "url": p.url}
            for p in candidates[: args.top_n]
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    query, search_terms, jobs = resolve_jobs(args)

    if args.prefilter_method == "strings" and not search_terms:
        raise SystemExit(
            'No search terms given. Pass --search-terms "term1,term2,..." (or a `search_terms` '
            "array in --config), or switch to --prefilter-method embeddings for semantic-only "
            "matching from --query alone."
        )

    search_fields = tuple(f.strip() for f in args.search_fields.split(",") if f.strip())

    evaluator = Evaluator(
        prefilter_method=args.prefilter_method,
        search_terms=search_terms,
        search_fields=search_fields,
        match_all=args.match_all,
        judge=not args.no_judge,
        claude_model=args.claude_model,
        prefilter_k=args.prefilter_k,
        min_prefilter_score=args.min_prefilter_score,
    )

    summary = [process_conference(job, query, evaluator, args) for job in jobs]

    print("\n=== Run summary ===")
    for entry in summary:
        if "error" in entry:
            print(f"  {entry['name']:<20} FAILED: {entry['error']}")
        else:
            print(
                f"  {entry['name']:<20} scraped={entry['scraped']:<5} "
                f"shortlisted={entry['shortlisted']:<5} downloaded={entry['downloaded']}"
            )
    if len(jobs) > 1:
        path = write_run_summary(summary, args.data_dir)
        print(f"\nFull run summary written to {path}")

    return 0 if all("error" not in e for e in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
