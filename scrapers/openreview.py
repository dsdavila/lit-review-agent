"""OpenReview scraper - metadata only, via the OpenReview API v2.

Handles venue/group pages like
https://openreview.net/group?id=ICLR.cc/2024/Conference
by querying api2.openreview.net/notes?content.venueid=<venue id>.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

API_URL = "https://api2.openreview.net/notes"
PAGE_SIZE = 100


@register_scraper
class OpenReviewScraper(BaseScraper):
    name = "openreview"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "openreview.net" in urlparse(url).netloc

    def _venue_id_for_url(self, url: str) -> str:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return unquote(qs["id"][0])
        # fall back: last path segment, e.g. /venue/ICLR.cc%2F2024
        segment = parsed.path.strip("/").split("/")[-1]
        return unquote(segment)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _fetch_page(self, venue_id: str, offset: int, batch: int) -> dict:
        params = {
            "content.venueid": venue_id,
            "offset": offset,
            "limit": batch,
        }
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        # The OpenReview API always returns the abstract for free - fetch_abstracts is a no-op here.
        venue_id = self._venue_id_for_url(url)
        logger.info("OpenReview venueid: %s", venue_id)
        want = limit or 200
        papers: list[PaperMetadata] = []
        offset = 0
        while len(papers) < want:
            batch = min(PAGE_SIZE, want - len(papers))
            try:
                data = self._fetch_page(venue_id, offset, batch)
            except requests.RequestException as exc:
                logger.error("OpenReview API request failed at offset %d: %s", offset, exc)
                break
            notes = data.get("notes", [])
            if not notes:
                break
            for note in notes:
                paper = self._parse_note(note, venue_id)
                if paper is not None:
                    papers.append(paper)
            offset += len(notes)
            if len(notes) < batch:
                break
        if not papers:
            logger.warning(
                "No OpenReview submissions found for venueid=%r - double-check the venue id "
                "(copy the `id=` query param from the venue's OpenReview URL)",
                venue_id,
            )
        return papers[: limit or len(papers)]

    def _parse_note(self, note: dict, venue_id: str) -> Optional[PaperMetadata]:
        content = note.get("content", {})

        def get(key: str):
            v = content.get(key, {})
            return v.get("value", "") if isinstance(v, dict) else (v or "")

        title = (get("title") or "").strip()
        abstract = (get("abstract") or "").strip()
        if not title:
            return None
        note_id = note.get("id", "")
        authors = get("authors")
        authors_list = authors if isinstance(authors, list) else ([authors] if authors else [])
        pdate = note.get("pdate") or note.get("cdate")
        year = None
        if pdate:
            try:
                year = datetime.datetime.utcfromtimestamp(pdate / 1000).year
            except (OverflowError, OSError, ValueError):
                year = None

        return PaperMetadata(
            title=title,
            abstract=abstract,
            url=f"https://openreview.net/forum?id={note_id}",
            year=year,
            pdf_url=f"https://openreview.net/pdf?id={note_id}",
            authors=authors_list,
            venue=venue_id,
            paper_id=note_id,
        )
