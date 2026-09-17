"""arXiv scraper - metadata only, via the arXiv Atom API (export.arxiv.org).

Never touches a PDF: the Atom API returns title/abstract/authors/links per
entry, which is exactly what the metadata-first architecture needs. No
BeautifulSoup dependency here - the Atom feed is parsed with the stdlib
xml.etree, which keeps this scraper working with zero extra installs.
"""
from __future__ import annotations

import calendar
import logging
import re
from typing import Optional
from urllib.parse import urlparse, parse_qs
from xml.etree import ElementTree as ET

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100


@register_scraper
class ArxivScraper(BaseScraper):
    name = "arxiv"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "arxiv.org" in urlparse(url).netloc

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _fetch_page(self, search_query: str, start: int, max_results: int) -> ET.Element:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return ET.fromstring(resp.content)

    def _search_query_for_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        qs = parse_qs(parsed.query)

        # https://arxiv.org/list/cs.CL/2024-01  (or /2024 for the whole year)
        m = re.match(r"list/([\w.\-]+)(?:/(\d{4})(?:-(\d{2}))?)?", path)
        if m:
            category, year, month = m.groups()
            query = f"cat:{category}"
            # arXiv's submittedDate range wants YYYYMMDDHHMM (12 digits, no seconds).
            if year and month:
                last_day = calendar.monthrange(int(year), int(month))[1]
                query += f" AND submittedDate:[{year}{month}010000 TO {year}{month}{last_day:02d}2359]"
            elif year:
                query += f" AND submittedDate:[{year}01010000 TO {year}12312359]"
            return query

        # https://arxiv.org/abs/2401.12345 or /pdf/2401.12345
        m = re.match(r"(?:abs|pdf)/([\w.\-]+?)(?:v\d+)?(?:\.pdf)?$", path)
        if m:
            return f"id:{m.group(1)}"

        # explicit ?search_query= / ?query= override
        if "search_query" in qs:
            return qs["search_query"][0]
        if "query" in qs:
            return f"all:{qs['query'][0]}"

        # last resort: treat the remaining path as a free-text search
        free_text = path.replace("/", " ").strip() or parsed.netloc
        return f"all:{free_text}"

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        # The arXiv API always returns the abstract for free - fetch_abstracts is a no-op here.
        search_query = self._search_query_for_url(url)
        logger.info("arXiv search query: %s", search_query)
        want = limit or 200
        papers: list[PaperMetadata] = []
        start = 0
        while len(papers) < want:
            batch = min(PAGE_SIZE, want - len(papers))
            try:
                root = self._fetch_page(search_query, start, batch)
            except requests.RequestException as exc:
                logger.error("arXiv API request failed at offset %d: %s", start, exc)
                break
            entries = root.findall(f"{ATOM_NS}entry")
            if not entries:
                break
            for entry in entries:
                papers.append(self._parse_entry(entry))
            start += len(entries)
            if len(entries) < batch:
                break
        if not papers:
            logger.warning("No arXiv results for query %r - check the URL / category / date range", search_query)
        return papers[: limit or len(papers)]

    def _parse_entry(self, entry: ET.Element) -> PaperMetadata:
        def text(tag: str, ns: str = ATOM_NS) -> str:
            el = entry.find(f"{ns}{tag}")
            return (el.text or "").strip() if el is not None else ""

        arxiv_id_url = text("id")
        arxiv_id = arxiv_id_url.rsplit("/", 1)[-1]
        title = re.sub(r"\s+", " ", text("title"))
        abstract = re.sub(r"\s+", " ", text("summary"))
        published = text("published")
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.find(f"{ATOM_NS}name").text or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
            if a.find(f"{ATOM_NS}name") is not None
        ]
        primary_cat_el = entry.find(f"{ARXIV_NS}primary_category")
        venue = primary_cat_el.get("term") if primary_cat_el is not None else None

        return PaperMetadata(
            title=title,
            abstract=abstract,
            url=arxiv_id_url,
            year=year,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            authors=authors,
            venue=venue,
            paper_id=arxiv_id,
        )
