"""Filesystem layout helpers.

    /data/{conference_name}/metadata.json
    /data/{conference_name}/papers/{selected_paper}.pdf
    /data/{conference_name}/summaries/{selected_paper}.json  # see summarizer.py
    /data/{conference_name}/filtering.log
    /data/run_summary.json   # written once per multi-conference batch run
    /data/SUMMARIES.md       # written by summarize.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from scrapers.base import PaperMetadata
from summarizer import PaperSummary


def slugify(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "-", name.strip()).strip("-").lower()
    return name or "conference"


def conference_dir(data_dir: Path, conference_name: str) -> Path:
    d = data_dir / slugify(conference_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def papers_dir(conf_dir: Path) -> Path:
    d = conf_dir / "papers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def metadata_path(conf_dir: Path) -> Path:
    return conf_dir / "metadata.json"


def save_metadata(papers: Iterable[PaperMetadata], conf_dir: Path) -> Path:
    path = metadata_path(conf_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in papers], f, indent=2, ensure_ascii=False)
    return path


def load_metadata(conf_dir: Path) -> list[PaperMetadata]:
    path = metadata_path(conf_dir)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [PaperMetadata.from_dict(d) for d in raw]


def summaries_dir(conf_dir: Path) -> Path:
    d = conf_dir / "summaries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_path(conf_dir: Path, paper: PaperMetadata) -> Path:
    return summaries_dir(conf_dir) / f"{paper.slug()}.json"


def save_summary(summary: PaperSummary, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)


def load_summary(path: Path) -> Optional[PaperSummary]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return PaperSummary.from_dict(json.load(f))


def write_run_summary(entries: list[dict[str, Any]], data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "run_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    return path
