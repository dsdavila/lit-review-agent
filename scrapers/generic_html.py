"""Template for a new metadata-only conference scraper.

Copy this file, rename the class, set `name`, implement `can_handle`, and
register it - that's the entire integration surface for a new conference
that has a nice API to hit (see arxiv.py / openreview.py for that pattern).

GenericHTMLScraper itself is a config-driven, CSS-selector-based scraper
for sites that only have a plain HTML listing (no API). It is intentionally
NOT auto-selected by URL (can_handle always returns False) because it needs
selector configuration; invoke it explicitly:

    python main.py --url <listing-url> --query "..." \\
        --scraper generic_html --selectors examples/generic_selectors.example.json

Playwright is used instead of requests+BeautifulSoup only when the page
requires JS rendering (pass --js) - most conference proceedings pages are
static HTML and don't need it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

HEADERS = {"User-Agent": "lit-review-agent/0.1 (metadata-only research tool)"}


@register_scraper
class GenericHTMLScraper(BaseScraper):
    """Configurable fallback for a conference site with plain HTML listings.

    Selector config (JSON) shape - see examples/generic_selectors.example.json:
        {
          "item_selector": "div.paper-entry",
          "title_selector": "h3.title a",
          "link_selector": "h3.title a",        # optional, defaults to title_selector
          "abstract_selector": "p.abstract",     # optional
          "abstract_page": true,                 # optional: fetch link_selector's href for the abstract
          "year": 2024,                          # optional static year, if not derivable from the page
          "pdf_suffix": ".pdf"                   # optional: appended to the paper URL to build pdf_url
        }
    """

    name = "generic_html"

    def __init__(self, selectors: Optional[dict] = None, js: bool = False):
        self.selectors = selectors or {}
        self.js = js

    @classmethod
    def can_handle(cls, url: str) -> bool:
        # Never auto-selected - requires --scraper generic_html --selectors ...
        return False

    def _fetch_html(self, url: str) -> str:
        if self.js:
            return self._fetch_html_playwright(url)
        return self._fetch_html_requests(url)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _fetch_html_requests(self, url: str) -> str:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _fetch_html_playwright(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install playwright` and "
                "`playwright install chromium`, or drop --js to use plain requests instead."
            ) from exc
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60_000)
            html = page.content()
            browser.close()
            return html

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        if not self.selectors.get("item_selector") or not self.selectors.get("title_selector"):
            raise ValueError(
                "GenericHTMLScraper requires at least item_selector and title_selector in its "
                "selector config - see the docstring on this class / examples/generic_selectors.example.json"
            )
        soup = BeautifulSoup(self._fetch_html(url), "html.parser")
        link_selector = self.selectors.get("link_selector", self.selectors["title_selector"])
        items = soup.select(self.selectors["item_selector"])
        if not items:
            logger.warning("item_selector %r matched nothing on %s", self.selectors["item_selector"], url)

        papers: list[PaperMetadata] = []
        for item in items:
            if limit and len(papers) >= limit:
                break
            title_el = item.select_one(self.selectors["title_selector"])
            if title_el is None:
                continue
            title = re.sub(r"\s+", " ", title_el.get_text()).strip()

            link_el = item.select_one(link_selector)
            href = link_el.get("href") if link_el else None
            paper_url = urljoin(url, href) if href else url

            abstract = ""
            abstract_selector = self.selectors.get("abstract_selector")
            if abstract_selector and (fetch_abstracts or not self.selectors.get("abstract_page")):
                abstract_el = None
                if self.selectors.get("abstract_page") and href:
                    try:
                        page_soup = BeautifulSoup(self._fetch_html(paper_url), "html.parser")
                        abstract_el = page_soup.select_one(abstract_selector)
                    except requests.RequestException as exc:
                        logger.warning("Could not fetch abstract page %s: %s", paper_url, exc)
                else:
                    abstract_el = item.select_one(abstract_selector)
                if abstract_el:
                    abstract = re.sub(r"\s+", " ", abstract_el.get_text()).strip()

            pdf_suffix = self.selectors.get("pdf_suffix")
            pdf_url = f"{paper_url}{pdf_suffix}" if pdf_suffix else None

            papers.append(
                PaperMetadata(
                    title=title,
                    abstract=abstract,
                    url=paper_url,
                    year=self.selectors.get("year"),
                    pdf_url=pdf_url,
                    paper_id=re.sub(r"\W+", "-", title.lower()).strip("-")[:80],
                )
            )
        return papers
