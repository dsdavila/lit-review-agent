"""ECVA (European Computer Vision Association) scraper - metadata only, for ECCV.

www.ecva.net/papers.php lists every ECCV year's papers on ONE page, each
year collapsed into its own accordion section ("ECCV {year} Papers"). Since
one URL covers every year, jobs must select a year with a `year=` query
param, e.g.:

    https://www.ecva.net/papers.php?year=2024

(The real site ignores unknown query params - this is just a local
convention this scraper reads before fetching the same real page.)

Like CVFScraper, only the first `limit` papers get their abstract fetched
(one request per paper) - ECCV runs 1500-2000+ papers/year.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

HEADERS = {"User-Agent": "lit-review-agent/0.1 (metadata-only research tool)"}
PAPERS_URL = "https://www.ecva.net/papers.php"


@register_scraper
class ECVAScraper(BaseScraper):
    name = "ecva"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "ecva.net" in urlparse(url).netloc

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

    def _group_records(self, dl: Tag) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        current: Optional[dict[str, Any]] = None
        for el in dl.find_all(["dt", "dd"], recursive=False):
            if el.name == "dt":
                current = {"dt": el, "dds": []}
                records.append(current)
            elif current is not None:
                current["dds"].append(el)
        return records

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        year_values = parse_qs(urlparse(url).query).get("year")
        if not year_values:
            raise ValueError(
                "ECVAScraper needs a `?year=YYYY` query param, e.g. "
                "https://www.ecva.net/papers.php?year=2024"
            )
        year = int(year_values[0])

        soup = self._get(PAPERS_URL)
        section = None
        for button in soup.select("button.accordion"):
            if f"ECCV {year}" in button.get_text():
                section = button.find_next_sibling("div", class_="accordion-content")
                break
        if section is None:
            logger.warning("No 'ECCV %d Papers' section found on %s - that year may not be posted yet", year, PAPERS_URL)
            return []

        dl = section.find("dl")
        records = self._group_records(dl) if dl else []
        if not records:
            logger.warning("ECCV %d section on %s had no dt/dd paper entries", year, PAPERS_URL)
            return []

        papers: list[PaperMetadata] = []
        for record in records:
            if limit and len(papers) >= limit:
                break
            link = record["dt"].select_one("a")
            if link is None or not link.get("href"):
                continue
            title = re.sub(r"\s+", " ", link.get_text()).strip()
            paper_url = urljoin(PAPERS_URL, link["href"])

            authors: list[str] = []
            pdf_url = None
            if record["dds"]:
                authors = [
                    " ".join(a.split()).strip("* ")
                    for a in record["dds"][0].get_text().split(",")
                    if a.strip(" *")
                ]
            for dd in record["dds"][1:]:
                pdf_link = dd.select_one("a[href*='.pdf']")
                if pdf_link and pdf_link.get("href"):
                    pdf_url = urljoin(PAPERS_URL, pdf_link["href"])
                    break

            abstract = ""
            if fetch_abstracts:
                try:
                    paper_soup = self._get(paper_url)
                    abstract_el = paper_soup.select_one("#abstract")
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
                    pdf_url=pdf_url,
                    authors=authors,
                    venue=f"ECCV{year}",
                    paper_id=re.sub(r"\W+", "-", title.lower()).strip("-")[:80],
                )
            )
        return papers
