"""CVF Open Access scraper - metadata only, for CVPR / ICCV / WACV.

openaccess.thecvf.com hosts the official proceedings for CVPR, ICCV, and
WACV. One listing page per conference-year (pass the URL with `?day=all`,
e.g. https://openaccess.thecvf.com/CVPR2024?day=all) gives every paper's
title + authors + a link to its detail page; the abstract lives on that
much smaller detail page, alongside a predictable PDF path derived from the
same URL - so this scraper still never touches a PDF.

Fetching the abstract costs one extra request per paper, so - like
ACLAnthologyScraper - this only fetches abstracts for the first `limit`
papers on the listing (main.py --limit, default 200), not the whole
conference (CVPR alone runs ~3000-4000 papers/year).
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

HEADERS = {"User-Agent": "lit-review-agent/0.1 (metadata-only research tool)"}


@register_scraper
class CVFScraper(BaseScraper):
    name = "cvf"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "openaccess.thecvf.com" in urlparse(url).netloc

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, url: str) -> BeautifulSoup:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def _venue_year(self, url: str) -> tuple[str, Optional[int]]:
        m = re.search(r"/([A-Za-z]+?)(\d{4})(?:$|[/?])", urlparse(url).path)
        if m:
            return m.group(1).upper(), int(m.group(2))
        return "CVF", None

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        venue, year = self._venue_year(url)
        soup = self._get(url)
        entries = soup.select("dt.ptitle")
        if not entries:
            logger.warning(
                "No paper entries found on %s - pass the `...?day=all` listing URL "
                "(the site paginates by day without it)",
                url,
            )
            return []

        papers: list[PaperMetadata] = []
        for dt in entries:
            if limit and len(papers) >= limit:
                break
            link = dt.select_one("a")
            if link is None or not link.get("href"):
                continue
            title = re.sub(r"\s+", " ", link.get_text()).strip()
            paper_url = urljoin(url, link["href"])
            pdf_url = re.sub(r"\.html$", ".pdf", paper_url.replace("/html/", "/papers/"))

            authors: list[str] = []
            authors_dd = dt.find_next_sibling("dd")
            if authors_dd is not None:
                authors = [
                    " ".join(a.split()).strip("* ")
                    for a in authors_dd.get_text().split(",")
                    if a.strip(" *")
                ]

            abstract = ""
            if fetch_abstracts:
                try:
                    paper_soup = self._get(paper_url)
                    abstract_el = paper_soup.select_one("#abstract")
                    if abstract_el:
                        abstract = re.sub(r"\s+", " ", abstract_el.get_text()).strip()
                except requests.RequestException as exc:
                    logger.warning("Could not fetch abstract for %s: %s (keeping title-only metadata)", paper_url, exc)

            paper_id = re.sub(r"\.html$", "", paper_url.rstrip("/").rsplit("/", 1)[-1])
            papers.append(
                PaperMetadata(
                    title=title,
                    abstract=abstract,
                    url=paper_url,
                    year=year,
                    pdf_url=pdf_url,
                    authors=authors,
                    venue=f"{venue}{year}" if year else venue,
                    paper_id=paper_id,
                )
            )
        return papers
