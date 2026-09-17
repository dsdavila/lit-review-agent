"""Selective Downloader - the only place actual PDF bytes get fetched.

Only ever called for papers the user has explicitly confirmed after the
two-stage filter - nothing here is reachable from the scraper or evaluator
layers, and the caller is expected to have already applied the relevance
threshold before handing papers to download_pdf().
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scrapers.base import PaperMetadata

logger = logging.getLogger("lit_review_agent")

HEADERS = {"User-Agent": "lit-review-agent/0.1 (metadata-only research tool)"}
CHUNK_SIZE = 1 << 16


class DownloadResult:
    def __init__(self, paper: PaperMetadata, path: Path | None, ok: bool, reason: str):
        self.paper = paper
        self.path = path
        self.ok = ok
        self.reason = reason


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _stream_download(url: str, dest: Path) -> None:
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def download_pdf(paper: PaperMetadata, dest_dir: Path, overwrite: bool = False) -> DownloadResult:
    if not paper.pdf_url:
        reason = "no pdf_url in metadata for this paper - visit `url` manually"
        logger.warning("Skipping download for %r: %s", paper.title, reason)
        return DownloadResult(paper, None, False, reason)

    dest = dest_dir / f"{paper.slug()}.pdf"
    if dest.exists() and not overwrite:
        reason = f"already downloaded at {dest}"
        logger.info("Skipping %r: %s", paper.title, reason)
        return DownloadResult(paper, dest, True, reason)

    try:
        _stream_download(paper.pdf_url, dest)
    except requests.RequestException as exc:
        reason = f"download failed after retries: {exc}"
        logger.error("Failed to download %r from %s: %s", paper.title, paper.pdf_url, exc)
        return DownloadResult(paper, None, False, reason)

    reason = f"downloaded to {dest}"
    logger.info("Downloaded %r: %s", paper.title, reason)
    return DownloadResult(paper, dest, True, reason)
