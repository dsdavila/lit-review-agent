"""Per-paper first-glance summaries - the Summary Layer.

For every paper that was actually downloaded (a PDF confirmed and saved by
downloader.py), asks Claude to read the PDF itself and produce a short
first-glance academic summary: the big idea, the actual contributions, the
deficiencies, and any interesting methods/findings not otherwise captured.
This is deliberately not a deep review - it's the kind of skim a reviewer
does before a close read, meant to get a researcher oriented across many
papers fast.

See summarize.py for the script that walks data/, calls this on every
downloaded paper (with caching), and writes a combined markdown doc.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scrapers.base import PaperMetadata

logger = logging.getLogger("lit_review_agent")

# Messages API request body limit is 32MB and PDFs go in base64 (~1.33x the
# binary size) alongside the rest of the request - stay well under that.
MAX_PDF_BYTES = 24 * 1024 * 1024

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "nutshell": {
            "type": "string",
            "description": "The big idea in a nutshell - 1-3 sentences: what problem, what's the core idea.",
        },
        "contributions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "The actual, concrete contributions (not the paper's own marketing claims) - "
            "new method, dataset, theoretical result, benchmark, etc.",
        },
        "deficiencies": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Weaknesses, limitations, or gaps a critical reviewer would flag - weak "
            "baselines, narrow evaluation, unsupported claims, missing ablations, etc. If the paper "
            "is genuinely solid, say so briefly here rather than inventing a complaint.",
        },
        "notable": {
            "type": "string",
            "description": "Brief note on any interesting methods, tricks, or findings not already "
            "captured above - empty string if nothing stands out beyond the contributions.",
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "Benchmark dataset name, e.g. 'Market1501', 'MSMT17'"},
                    "metric": {"type": "string", "description": "Metric name, e.g. 'Rank-1', 'mAP', 'HOTA'"},
                    "value": {"type": "string", "description": "The reported value exactly as printed in the paper's table, e.g. '88.4' or '95.2%'"},
                    "setting": {
                        "type": "string",
                        "description": "Short note on protocol/backbone/split if it affects comparability "
                        "(e.g. 'single-query', 'ResNet50 backbone') - empty string if not applicable.",
                    },
                },
                "required": ["dataset", "metric", "value", "setting"],
                "additionalProperties": False,
            },
            "description": "This paper's own proposed method's headline results from its main results "
            "table(s) - one row per dataset+metric pair. Not competitor/baseline numbers, just what this "
            "paper's own method achieves, taken from the table it uses to claim its main results (usually "
            "labeled 'Ours' or the paper's method name). Empty array if the paper has no such table "
            "(e.g. a pure benchmark/dataset paper).",
        },
    },
    "required": ["nutshell", "contributions", "deficiencies", "notable", "results"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are an experienced computer vision researcher doing a first-pass read of a paper - the "
    "kind of skim a reviewer does before a close read, not a full deep review. Be direct and "
    "specific, and ground every claim in what the paper actually says. Contributions should be the "
    "real, concrete novelty - not the paper's own marketing language. Deficiencies should be genuine "
    "limitations a critical reviewer would flag; if the paper is genuinely solid, say so briefly "
    "rather than inventing a complaint. Also extract the paper's own headline results (its proposed "
    "method's numbers, not competitors') from its main results table(s), so results can be compared "
    "across papers later."
)


@dataclass
class PaperSummary:
    nutshell: str
    contributions: list[str]
    deficiencies: list[str]
    notable: str
    results: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nutshell": self.nutshell,
            "contributions": self.contributions,
            "deficiencies": self.deficiencies,
            "notable": self.notable,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperSummary":
        return cls(
            nutshell=d["nutshell"],
            contributions=list(d["contributions"]),
            deficiencies=list(d["deficiencies"]),
            notable=d["notable"],
            results=list(d.get("results", [])),
        )


class ClaudeSummarizer:
    """Reads a paper's actual PDF and produces a first-glance academic summary."""

    def __init__(self, model: Optional[str] = None, effort: str = "medium"):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        self.effort = effort
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        import anthropic
                    except ImportError as exc:
                        raise RuntimeError(
                            "The `anthropic` package is required for summarization. Install it "
                            "with `pip install anthropic`."
                        ) from exc
                    self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / an `ant auth login` profile
        return self._client

    def summarize(self, paper: PaperMetadata, pdf_path: Path) -> PaperSummary:
        client = self._get_client()

        pdf_bytes = pdf_path.read_bytes()
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(
                f"{pdf_path} is {len(pdf_bytes)} bytes, over the {MAX_PDF_BYTES}-byte inline limit "
                "for this script"
            )
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode()

        context_lines = [f"Title: {paper.title}"]
        if paper.authors:
            context_lines.append(f"Authors: {', '.join(paper.authors)}")
        venue_year = " ".join(x for x in [paper.venue, str(paper.year) if paper.year else None] if x)
        if venue_year:
            context_lines.append(f"Venue: {venue_year}")

        response = client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=_SYSTEM_PROMPT,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": _SUMMARY_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
                        },
                        {
                            "type": "text",
                            "text": "\n".join(context_lines) + "\n\nGive your first-glance read of this paper.",
                        },
                    ],
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return PaperSummary.from_dict(json.loads(text))
