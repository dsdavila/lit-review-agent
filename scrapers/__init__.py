"""Modular metadata-only scraper registry.

Importing this package registers every built-in scraper. To add a new
conference: create scrapers/<name>.py with a BaseScraper subclass decorated
with @register_scraper, then add one import line below.
"""
from .base import (  # noqa: F401
    BaseScraper,
    PaperMetadata,
    all_scrapers,
    get_scraper_by_name,
    get_scraper_for_url,
)

from . import acl_anthology  # noqa: F401
from . import arxiv  # noqa: F401
from . import bmvc  # noqa: F401
from . import cvf  # noqa: F401
from . import ecva  # noqa: F401
from . import generic_html  # noqa: F401
from . import openreview  # noqa: F401

__all__ = [
    "BaseScraper",
    "PaperMetadata",
    "get_scraper_for_url",
    "get_scraper_by_name",
    "all_scrapers",
]
