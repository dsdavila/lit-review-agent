"""ACL Anthology scraper - metadata only, via BeautifulSoup HTML parsing.

Handles event pages (https://aclanthology.org/events/acl-2024/) and volume
pages (https://aclanthology.org/volumes/2024.acl-long/). The listing page
gives title/authors/id; the abstract lives on the much smaller per-paper
page, which we fetch instead of the PDF.

Anthology markup does drift over time - if `_get`'s selectors stop
matching, fetch_metadata logs a warning and returns an empty list rather
than silently fabricating results.
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
class ACLAnthologyScraper(BaseScraper):
    name = "acl_anthology"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "aclanthology.org" in urlparse(url).netloc

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

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        soup = self._get(url)
        entries = soup.select("p.d-sm-flex.align-items-stretch")
        if not entries:
            logger.warning(
                "No paper entries found on %s - the ACL Anthology page structure may have "
                "changed, or this isn't an events/volumes listing page",
                url,
            )
            return []

        papers: list[PaperMetadata] = []
        for entry in entries:
            if limit and len(papers) >= limit:
                break
            title_link = entry.select_one("strong a")
            if title_link is None or not title_link.get("href"):
                continue
            paper_url = urljoin(url, title_link["href"])
            title = re.sub(r"\s+", " ", title_link.get_text()).strip()
            paper_id = title_link["href"].strip("/").rsplit("/", 1)[-1]

            authors = [
                re.sub(r"\s+", " ", a.get_text()).strip() for a in entry.select("a[href*='/people/']")
            ]

            year_match = re.match(r"(\d{4})\.", paper_id)
            year = int(year_match.group(1)) if year_match else None

            abstract = ""
            if fetch_abstracts:
                try:
                    paper_soup = self._get(paper_url)
                    abstract_el = paper_soup.select_one("div.acl-abstract span")
                    if abstract_el:
                        abstract = re.sub(r"\s+", " ", abstract_el.get_text()).strip()
                except requests.RequestException as exc:
                    logger.warning("Could not fetch abstract for %s: %s (keeping title-only metadata)", paper_url, exc)

            papers.append(
                PaperMetadata(
                    title=title,
                    abstract=abstract,
                    url=paper_url,
                    year=year,
                    pdf_url=urljoin(url, f"/{paper_id}.pdf"),
                    authors=authors,
                    venue="ACL Anthology",
                    paper_id=paper_id,
                )
            )
        return papers
