"""Two-stage relevance filtering - the Filter Layer.

Stage 1 (prefilter): cheap, local, deterministic filtering across EVERY
scraped paper. This is what keeps the pipeline from blowing through
token/context budgets on a large conference - nothing goes to the LLM until
it has survived this pass.

  - "strings" (default): case-insensitive matching of a user-supplied set
    of search terms/phrases against the title (and, by default, abstract).
    No scoring model, no embeddings, no latent space - a paper is
    shortlisted only if it literally contains one of your terms (or all of
    them, with --match-all). You decide exactly what counts as a hit.
  - "embeddings": SentenceTransformers cosine similarity against a
    free-text --query, for when you want semantic recall instead of exact
    string matches. Opt-in - requires `sentence-transformers` installed.

Stage 2 (judge): Claude looks at (title, abstract) for only the papers that
survived stage 1 (the "shortlist", default top 40), and returns a
calibrated 0-100 relevance score plus a one-sentence reason via structured
outputs. This is the score actually used to rank and select papers for
download. Disable with judge=False / --no-judge to run prefilter-only
(fully local, no API key required).

Evaluator.run() mutates each PaperMetadata in place with both stages'
scores/reasons and returns the list re-sorted by final_score (judge score
if present, else prefilter score) descending. Papers that don't clear the
prefilter (score <= 0 - i.e. no search term matched) never reach the judge,
regardless of --prefilter-k.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from scrapers.base import PaperMetadata

logger = logging.getLogger("lit_review_agent")

DEFAULT_SEARCH_FIELDS = ("title", "abstract")


class PrefilterScorer:
    """Stage 1: fast, local, no API cost."""

    def __init__(
        self,
        method: str = "strings",
        search_terms: Optional[list[str]] = None,
        search_fields: tuple[str, ...] = DEFAULT_SEARCH_FIELDS,
        match_all: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.method = method
        self.search_terms = [t.strip() for t in (search_terms or []) if t.strip()]
        self.search_fields = search_fields
        self.match_all = match_all
        self.embedding_model_name = embedding_model
        self._st_model = None

    def score(self, query: str, papers: list[PaperMetadata]) -> list[tuple[PaperMetadata, float, str]]:
        """Returns (paper, score, reason) aligned to `papers`' order (not sorted).

        For "strings", score is the count of matched terms (0 = no match,
        never shortlisted). For "embeddings", score is 0-100 cosine similarity.
        """
        if self.method == "strings":
            if not self.search_terms:
                raise ValueError(
                    "--prefilter-method strings (the default) needs --search-terms - a "
                    'comma-separated list of case-insensitive strings/phrases to match against '
                    'title/abstract, e.g. --search-terms "sparse routing,mixture of experts". '
                    "Pass --prefilter-method embeddings instead for semantic-only matching from "
                    "--query alone."
                )
            logger.info(
                "Prefilter: matching %d papers against %d search term(s) (match %s) in %s",
                len(papers),
                len(self.search_terms),
                "ALL" if self.match_all else "ANY",
                "+".join(self.search_fields),
            )
            return self._score_strings(papers)

        if self.method == "embeddings":
            if not self._try_load_embeddings():
                raise RuntimeError(
                    "sentence-transformers is required for --prefilter-method embeddings. "
                    "Install it with `pip install sentence-transformers`, or use the default "
                    "--prefilter-method strings instead."
                )
            logger.info("Prefilter: scoring %d papers via embeddings cosine similarity", len(papers))
            return self._score_embeddings(query, papers)

        raise ValueError(f"Unknown prefilter method {self.method!r}")

    def _score_strings(self, papers: list[PaperMetadata]) -> list[tuple[PaperMetadata, float, str]]:
        terms = [(t, t.lower()) for t in self.search_terms]
        results = []
        for paper in papers:
            haystack_parts = []
            if "title" in self.search_fields:
                haystack_parts.append(paper.title or "")
            if "abstract" in self.search_fields:
                haystack_parts.append(paper.abstract or "")
            haystack = " ".join(haystack_parts).lower()

            matched = [original for original, lowered in terms if lowered in haystack]
            is_hit = len(matched) == len(terms) if self.match_all else len(matched) > 0
            score = float(len(matched)) if is_hit else 0.0

            if is_hit:
                reason = f"Matched search term(s): {', '.join(matched)}"
            elif self.match_all:
                missing = [t for t in self.search_terms if t not in matched]
                reason = f"Missing required term(s): {', '.join(missing)}"
            else:
                reason = "No search terms matched"
            results.append((paper, score, reason))
        return results

    def _try_load_embeddings(self) -> bool:
        if self._st_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.debug("sentence-transformers not installed")
            return False
        try:
            self._st_model = SentenceTransformer(self.embedding_model_name)
            return True
        except Exception as exc:  # model download failure, offline, corrupt cache, etc.
            logger.warning("Could not load embedding model %r: %s", self.embedding_model_name, exc)
            return False

    def _score_embeddings(self, query: str, papers: list[PaperMetadata]) -> list[tuple[PaperMetadata, float, str]]:
        texts = [f"{p.title}. {p.abstract}" for p in papers]
        query_vec = self._st_model.encode([query], normalize_embeddings=True)[0]
        doc_vecs = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        results = []
        for paper, doc_vec in zip(papers, doc_vecs):
            sim = float(doc_vec @ query_vec)  # cosine similarity (vectors are normalized)
            score = max(0.0, min(1.0, (sim + 1) / 2)) * 100  # map [-1,1] -> [0,100]
            reason = f"Semantic cosine similarity to query: {sim:.3f}"
            results.append((paper, score, reason))
        return results


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "0-based index of the paper in this batch"},
                    "score": {"type": "integer", "description": "0-100 relevance score"},
                    "reason": {"type": "string", "description": "one-sentence justification"},
                },
                "required": ["index", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


class ClaudeJudge:
    """Stage 2: Claude relevance judgement (the LLM adjudicator). Only called on the prefilter shortlist."""

    def __init__(self, model: Optional[str] = None, batch_size: int = 15):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The `anthropic` package is required for LLM judging. Install it with "
                    "`pip install anthropic`, or pass --no-judge to rank on the prefilter score only."
                ) from exc
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / an `ant auth login` profile
        return self._client

    def score(self, query: str, papers: list[PaperMetadata]) -> list[Optional[tuple[float, str]]]:
        """Returns (score, reason) aligned to `papers`, or None per-paper on a failed batch."""
        client = self._get_client()
        out: list[Optional[tuple[float, str]]] = [None] * len(papers)
        for start in range(0, len(papers), self.batch_size):
            batch = papers[start : start + self.batch_size]
            try:
                batch_results = self._score_batch(client, query, batch)
            except Exception as exc:
                logger.error(
                    "Claude judging failed for papers %d-%d (%s); these keep their prefilter score",
                    start,
                    start + len(batch) - 1,
                    exc,
                )
                continue
            for i, result in enumerate(batch_results):
                out[start + i] = result
        return out

    def _score_batch(self, client, query: str, batch: list[PaperMetadata]) -> list[tuple[float, str]]:
        papers_block = "\n\n".join(
            f"[{i}] Title: {p.title}\nAbstract: {p.abstract or '(no abstract available)'}"
            for i, p in enumerate(batch)
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are screening papers for a literature review. Research query: "
                        f"{query!r}\n\nFor each numbered paper below, score how relevant its abstract "
                        "is to the research query, from 0 (irrelevant) to 100 (directly on-topic and "
                        "highly useful). Judge only from the title and abstract given - do not assume "
                        "anything else about the paper.\n\n"
                        f"{papers_block}"
                    ),
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        by_index = {item["index"]: item for item in data["scores"]}
        results = []
        for i in range(len(batch)):
            item = by_index.get(i)
            if item is None:
                raise ValueError(f"Claude response missing a score for batch index {i}")
            results.append((float(item["score"]), item["reason"]))
        return results


class Evaluator:
    """Orchestrates the two-stage prefilter -> judge pipeline."""

    def __init__(
        self,
        prefilter_method: str = "strings",
        search_terms: Optional[list[str]] = None,
        search_fields: tuple[str, ...] = DEFAULT_SEARCH_FIELDS,
        match_all: bool = False,
        judge: bool = True,
        claude_model: Optional[str] = None,
        prefilter_k: int = 40,
        min_prefilter_score: Optional[float] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        judge_batch_size: int = 15,
    ):
        self.prefilter = PrefilterScorer(
            method=prefilter_method,
            search_terms=search_terms,
            search_fields=search_fields,
            match_all=match_all,
            embedding_model=embedding_model,
        )
        self.judge_enabled = judge
        self.judge = ClaudeJudge(model=claude_model, batch_size=judge_batch_size) if judge else None
        self.prefilter_k = prefilter_k
        self.min_prefilter_score = min_prefilter_score

    def run(self, query: str, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        if not papers:
            return []

        for paper, score, reason in self.prefilter.score(query, papers):
            paper.prefilter_score = round(score, 2)
            paper.prefilter_reason = reason

        ranked = sorted(papers, key=lambda p: p.prefilter_score or 0.0, reverse=True)
        # A score of 0 means nothing matched (strings) or a genuinely null signal
        # (embeddings) - never shortlist those just to pad out --prefilter-k.
        shortlist = [p for p in ranked[: self.prefilter_k] if (p.prefilter_score or 0.0) > 0]
        if self.min_prefilter_score is not None:
            shortlist = [p for p in shortlist if (p.prefilter_score or 0.0) >= self.min_prefilter_score]

        shortlist_ids = {id(p) for p in shortlist}
        cut = [p for p in ranked if id(p) not in shortlist_ids]
        for p in cut:
            logger.debug("Prefilter cut %r (score %.1f): %s", p.title, p.prefilter_score or 0.0, p.prefilter_reason)
        if cut:
            logger.info(
                "Prefilter shortlisted %d/%d papers (%d cut - see filtering.log for each reason)",
                len(shortlist),
                len(papers),
                len(cut),
            )

        if not shortlist:
            logger.warning("Nothing survived the prefilter shortlist - nothing to rank or download")
        elif self.judge_enabled:
            logger.info("Judge (Claude, model=%s): scoring %d shortlisted papers ...", self.judge.model, len(shortlist))
            judge_results = self.judge.score(query, shortlist)
            for paper, result in zip(shortlist, judge_results):
                paper.shortlisted = True
                if result is not None:
                    paper.judge_score, paper.judge_reason = result
                else:
                    logger.warning("No Claude judgement for %r - keeping prefilter score for ranking", paper.title)
        else:
            for paper in shortlist:
                paper.shortlisted = True

        return sorted(papers, key=lambda p: p.final_score or 0.0, reverse=True)
