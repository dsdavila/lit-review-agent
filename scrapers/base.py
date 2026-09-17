"""Base classes and registry for metadata-only conference scrapers.

Every scraper module in this package registers itself with the global
registry by subclassing BaseScraper and decorating the class with
@register_scraper. main.py resolves the correct scraper for a given URL
via get_scraper_for_url() - or by name, for an explicit override.
"""
from __future__ import annotations

import abc
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("lit_review_agent")


@dataclass
class PaperMetadata:
    """Metadata-only record for a single paper. No PDF bytes ever live here.

    Scoring is two-stage (see evaluator.py):
      - prefilter_score/reason: cheap local pass over EVERY paper
      - judge_score/reason:     Claude's pass, only over the prefilter shortlist
      - shortlisted:            True if this paper made it to the judge stage
    """

    title: str
    abstract: str
    url: str
    year: Optional[int] = None
    pdf_url: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    venue: Optional[str] = None
    paper_id: Optional[str] = None  # scraper-local identifier, used for filenames
    extra: dict[str, Any] = field(default_factory=dict)

    prefilter_score: Optional[float] = None
    prefilter_reason: Optional[str] = None
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None
    shortlisted: bool = False

    @property
    def final_score(self) -> Optional[float]:
        """The score to rank/select on: Claude's judgement if available, else the prefilter score."""
        return self.judge_score if self.judge_score is not None else self.prefilter_score

    @property
    def final_reason(self) -> Optional[str]:
        return self.judge_reason if self.judge_reason is not None else self.prefilter_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "year": self.year,
            "pdf_url": self.pdf_url,
            "authors": self.authors,
            "venue": self.venue,
            "paper_id": self.paper_id,
            "prefilter_score": self.prefilter_score,
            "prefilter_reason": self.prefilter_reason,
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "shortlisted": self.shortlisted,
            "final_score": self.final_score,
            "final_reason": self.final_reason,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperMetadata":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def slug(self) -> str:
        """A filesystem-safe stem for the PDF filename."""
        base = self.paper_id or self.title
        base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
        base = re.sub(r"[^\w\-]+", "-", base).strip("-").lower()
        return base[:120] or "paper"


class BaseScraper(abc.ABC):
    """Subclass + @register_scraper to add a new metadata-only conference scraper.

    That's the entire integration surface for a new conference: one file
    under scrapers/, one class, one import line in scrapers/__init__.py.
    """

    #: short, unique registry key; also the default conference folder name
    #: when the caller doesn't supply an explicit `name` for a job
    name: str = "base"

    @classmethod
    @abc.abstractmethod
    def can_handle(cls, url: str) -> bool:
        """Return True if this scraper knows how to fetch metadata from `url`."""

    @abc.abstractmethod
    def fetch_metadata(
        self, url: str, limit: Optional[int] = None, fetch_abstracts: bool = True
    ) -> list[PaperMetadata]:
        """Fetch (Title, Abstract, URL, Year, ...) for papers found at `url`.

        Must NEVER download a PDF - only lightweight HTML/XML/JSON index pages.
        `limit`, when given, caps the number of records returned/fetched.

        `fetch_abstracts`: some venues (CVF, ECVA, ACL Anthology, BMVC 2025+)
        only publish the abstract on a per-paper detail page, costing one
        extra HTTP request per paper. Pass False to skip that entirely and
        return title/authors/pdf_url only - turns an O(papers) scrape into
        O(1) per conference-year. Scrapers that get the abstract for free
        from their listing/API response (arXiv, OpenReview) ignore this.
        """


_REGISTRY: dict[str, type[BaseScraper]] = {}


def register_scraper(cls: type[BaseScraper]) -> type[BaseScraper]:
    """Class decorator: adds `cls` to the scraper registry."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        logger.warning("Scraper name %r re-registered (was %s, now %s)", cls.name, _REGISTRY[cls.name], cls)
    _REGISTRY[cls.name] = cls
    return cls


def all_scrapers() -> dict[str, type[BaseScraper]]:
    return dict(_REGISTRY)


def get_scraper_for_url(url: str) -> BaseScraper:
    for scraper_cls in _REGISTRY.values():
        try:
            if scraper_cls.can_handle(url):
                return scraper_cls()
        except Exception:  # a broken can_handle() shouldn't crash discovery
            logger.exception("can_handle() raised in %s", scraper_cls)
    available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
    raise ValueError(
        f"No registered scraper can handle URL: {url!r}. Available scrapers: {available}. "
        "Add one by subclassing BaseScraper in scrapers/ - see scrapers/generic_html.py - "
        "or pass --scraper generic_html --selectors <config.json> for a one-off site."
    )


def get_scraper_by_name(name: str) -> BaseScraper:
    try:
        return _REGISTRY[name]()
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown scraper {name!r}. Available: {available}")
