"""BMVC scraper - metadata only, for the British Machine Vision Conference.

BMVC proceedings move to a new domain almost every year - e.g.
proceedings.bmvc2023.org, bmvc2024.org/proceedings/conference-proceedings/,
bmvc2025.bmva.org/proceedings/conference-proceedings/ - so this scraper
matches on "bmvc" appearing anywhere in the URL rather than a fixed domain,
and a job just points at whichever year's proceedings URL you found (see
examples/cv_conferences.example.json for known-good ones).

The listing table (`<tr id="paper">` rows) is stable across years, but
what the title links to isn't: 2023/2024 link the title straight to the
PDF (no abstract available anywhere); 2025 links it to a per-paper detail
page that *does* publish an abstract under an `<h2 id="abstract">` heading.
This scraper detects which shape it's looking at per-row and fetches the
abstract when there's a detail page to fetch it from - like CVFScraper,
that costs one extra request per paper, bounded by `limit`.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseScraper, PaperMetadata, register_scraper

logger = logging.getLogger("lit_review_agent")

HEADERS = {"User-Agent": "lit-review-agent/0.1 (metadata-only research tool)"}
_HEADING_TAGS = ("h1", "h2", "h3")


@register_scraper
class BMVCScraper(BaseScraper):
    name = "bmvc"

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "bmvc" in url.lower()

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

    def _clean_authors(self, text: str) -> list[str]:
        # 2023/2024 join authors with commas; 2025 switched to semicolons - split on either.
        return [" ".join(a.split()).strip("* ") for a in re.split(r"[,;]", text) if a.strip(" *")]

    def _fetch_abstract(self, detail_url: str) -> str:
        try:
            soup = self._get(detail_url)
        except requests.RequestException as exc:
            logger.warning("Could not fetch detail page %s: %s (keeping title-only metadata)", detail_url, exc)
            return ""
        heading = soup.find(id="abstract")
        if heading is None:
            return ""
        parts: list[str] = []
        for sib in heading.next_siblings:
            if isinstance(sib, Tag) and sib.name in _HEADING_TAGS:
                break
            if isinstance(sib, NavigableString):
                parts.append(str(sib))
            elif isinstance(sib, Tag):
                parts.append(sib.get_text(" ", strip=True))
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        year_match = re.search(r"20\d{2}", url)
        year = int(year_match.group()) if year_match else None

        soup = self._get(url)
        rows = soup.select('tr[id="paper"]')
        if not rows:
            logger.warning("No paper rows found on %s - the BMVC page structure may have changed", url)
            return []

        papers: list[PaperMetadata] = []
        for row in rows:
            if limit and len(papers) >= limit:
                break
            title_link = row.select_one("td strong a")
            if title_link is None or not title_link.get("href"):
                continue
            title = re.sub(r"\s+", " ", title_link.get_text()).strip()
            title_href = urljoin(url, title_link["href"])

            if title_href.lower().endswith(".pdf"):
                # 2023/2024 shape: title links straight to the PDF, no abstract available
                pdf_url = title_href
                paper_url = pdf_url
                abstract = ""
            else:
                # 2025+ shape: title links to a per-paper detail page with an abstract
                paper_url = title_href
                pdf_link = row.select_one("a[href$='.pdf']")
                pdf_url = urljoin(url, pdf_link["href"]) if pdf_link and pdf_link.get("href") else None
                abstract = self._fetch_abstract(paper_url) if fetch_abstracts else ""

            authors: list[str] = []
            cell = title_link.find_parent("td")
            if cell is not None:
                text_parts = list(cell.stripped_strings)
                if title in text_parts:
                    idx = text_parts.index(title)
                    author_text = text_parts[idx + 1] if idx + 1 < len(text_parts) else ""
                    authors = self._clean_authors(author_text)

            papers.append(
                PaperMetadata(
                    title=title,
                    abstract=abstract,
                    url=paper_url,
                    year=year,
                    pdf_url=pdf_url,
                    authors=authors,
                    venue=f"BMVC{year}" if year else "BMVC",
                    paper_id=re.sub(r"\W+", "-", title.lower()).strip("-")[:80],
                )
            )
        return papers
