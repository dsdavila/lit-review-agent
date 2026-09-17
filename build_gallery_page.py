"""Builds the "Gallery Set" leaderboard page from every cached paper summary.

Reads every data/{conference}/summaries/*.json (see summarize.py), applies
light normalization on top, and writes a single self-contained HTML file
ready to hand to Claude Code's Artifact tool for publishing (this script does
not publish anything itself - that requires Claude/the Artifact tool, there is
no API for it).

Normalization applied (deliberately conservative - see README):
  - dataset names: case/punctuation/whitespace-insensitive grouping only
    (fixes "Market-1501" vs "Market1501" vs "Market 1501"). Does not attempt
    semantic merges (e.g. "AG-ReID" vs "AG-ReID.v1" stay separate).
  - metric names: a small, high-confidence alias table for the Rank-K family
    only (Rank-1/R1/R@1/CMC-1/Top-1/... all mean the same thing in ReID).
    Never merges anything containing "lower" (inverted/attack/privacy metrics
    would otherwise get sorted as if higher were better).
  - task tagging: rule-based, from each paper's (title + nutshell) text, into
    a fixed set of categories - not from the dataset name, which is too
    ambiguous/plentiful to classify reliably one string at a time.

Usage:
    python build_gallery_page.py
    python build_gallery_page.py --data-dir data --output data/gallery_set.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from summarize import discover_conference_dirs, summarizable_papers
from storage import load_summary, summary_path

TEMPLATE_PATH = Path(__file__).parent / "gallery_set_template.html"

# ---------------------------------------------------------------------------
# Dataset name canonicalization: pure normalization, no semantic guessing.
# ---------------------------------------------------------------------------


def dataset_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Metric alias table - Rank-K family only, high confidence. Never touches
# anything with "lower" in it (inverted/attack/privacy metrics).
# ---------------------------------------------------------------------------

RANK_ALIASES = {
    1: ["rank-1", "rank1", "r1", "r@1", "rank@1", "cmc-1", "cmc1", "cmc@1", "cmc-r1", "top-1", "top1"],
    3: ["rank-3", "rank3", "top-3", "top3"],
    5: ["rank-5", "rank5", "r5", "r@5", "cmc@5", "top-5", "top5"],
    10: ["rank-10", "rank10", "cmc@10", "top-10", "top10"],
    20: ["rank-20", "rank20", "top-20", "top20"],
}
_RANK_LOOKUP = {alias: f"Rank-{k}" for k, aliases in RANK_ALIASES.items() for alias in aliases}


def metric_key(metric: str) -> str:
    m = metric.strip()
    if "lower" in m.lower():
        return m  # never merge inverted/attack metrics into the normal ranking
    norm = re.sub(r"\s+", "", m.lower())
    return _RANK_LOOKUP.get(norm, m)


def parse_numeric(value: str) -> Optional[float]:
    m = re.search(r"-?\d+\.?\d*", value)
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Task tagging - rule-based, applied to (title + nutshell) lowercased text,
# first matching rule wins. Order matters: more specific rules first.
# ---------------------------------------------------------------------------

TASK_RULES = [
    ("Animal / Wildlife Re-ID", [
        "wildlife", "animal re-identification", "animal identification", "animal re-id",
        "tiger", "turtle", "gorilla", "zebra", "macaque", "honey bee", "honeybee",
        "cattle", "fish counting", "pet identification", "petface",
    ]),
    ("Visible-Infrared / Cross-Modality ReID", [
        "visible-infrared", "visible infrared", "cross-modality", "cross modality", "rgb-ir",
        "rgb-nir", "nir-vis", "vi-reid", "thermal",
    ]),
    ("Multi-Modal Object Re-ID", [
        "multi-modal object re-identification", "multimodal object re-identification",
        "rgbnt", "multi-modal re-id", "ship re-identification", "sar imagery",
    ]),
    ("Clothes-Changing ReID", [
        "clothes-changing", "clothes changing", "cloth-changing", "cloth changing",
        "clothing change", "cc-reid",
    ]),
    ("Text-to-Image / Language-Guided ReID", [
        "text-to-image person", "text-image person", "text-guided", "language guidance person",
        "pedes", "natural language description",
    ]),
    ("Occluded / Partial ReID", [
        "occluded person", "partial person", "partial re-id", "occlusion",
    ]),
    ("Vehicle Re-ID", [
        "vehicle re-identification", "vehicle re-id",
    ]),
    ("Aerial-Ground ReID", [
        "aerial-ground", "aerial ground", "uav", "drone", "sky and ground",
    ]),
    ("Video Person ReID", [
        "video-based person", "video person re-identification", "video re-id",
    ]),
    ("Person Search", [
        "person search",
    ]),
    ("Lifelong / Continual ReID", [
        "lifelong person", "continual", "catastrophic forgetting", "class-incremental",
    ]),
    ("Domain Generalization / UDA ReID", [
        "domain generaliz", "domain adaptive", "domain adaptation", "unsupervised domain",
        "cross-domain",
    ]),
    ("Privacy / De-identification / Adversarial", [
        "privacy", "de-identification", "de-reidentification", "anonymiz", "adversarial attack",
        "model inversion",
    ]),
    ("Multi-Object Tracking", [
        "multi-object tracking", "multiple object tracking", "multi-person tracking",
        "video instance segmentation", "point tracking", "object tracking",
    ]),
    ("Person Re-ID (general)", [
        "person re-identification", "person re-id", "person reid", "pedestrian re-identification",
        "human recognition", "human re-identification",
    ]),
]


def classify_task(title: str, nutshell: str) -> str:
    text = f"{title} {nutshell}".lower()
    for label, keywords in TASK_RULES:
        if any(kw in text for kw in keywords):
            return label
    return "Other CV Research"


# ---------------------------------------------------------------------------
# Build the consolidated dataset + splice into the HTML template
# ---------------------------------------------------------------------------


def build_papers(data_dir: Path) -> list[dict]:
    conf_dirs = discover_conference_dirs(data_dir, None)

    papers_out = []
    dataset_display: dict[str, dict[str, int]] = {}
    metric_display: dict[str, dict[str, int]] = {}

    for conf_dir in conf_dirs:
        for paper, _ in summarizable_papers(conf_dir):
            s = load_summary(summary_path(conf_dir, paper))
            if s is None:
                continue
            task = classify_task(paper.title, s.nutshell)

            norm_results = []
            for row in s.results:
                dkey = dataset_key(row["dataset"])
                dataset_display.setdefault(dkey, {}).setdefault(row["dataset"].strip(), 0)
                dataset_display[dkey][row["dataset"].strip()] += 1

                mkey = metric_key(row["metric"])
                metric_display.setdefault(mkey, {}).setdefault(row["metric"].strip(), 0)
                metric_display[mkey][row["metric"].strip()] += 1

                norm_results.append({
                    "dataset_key": dkey,
                    "metric_key": mkey,
                    "value": row["value"],
                    "value_numeric": parse_numeric(row["value"]),
                    "setting": row.get("setting", ""),
                })

            papers_out.append({
                "title": paper.title,
                "authors": paper.authors,
                "venue": paper.venue,
                "year": paper.year,
                "url": paper.url,
                "conference_dir": conf_dir.name,
                "score": paper.final_score,
                "task": task,
                "nutshell": s.nutshell,
                "contributions": s.contributions,
                "deficiencies": s.deficiencies,
                "notable": s.notable,
                "results": norm_results,
            })

    dataset_name_for_key = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in dataset_display.items()}
    metric_name_for_key = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in metric_display.items()}

    for p in papers_out:
        for row in p["results"]:
            row["dataset"] = dataset_name_for_key[row["dataset_key"]]
            row["metric"] = metric_name_for_key[row["metric_key"]]
            del row["dataset_key"], row["metric_key"]

    return papers_out


def render_html(papers: list[dict], template_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    if "__DATA_JSON__" not in template:
        raise ValueError(f"{template_path} is missing the __DATA_JSON__ placeholder")
    data_json = json.dumps(papers, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")  # keep </script> out of the embedded JSON
    return template.replace("__DATA_JSON__", data_json)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--template", type=Path, default=TEMPLATE_PATH, help="HTML template with a __DATA_JSON__ placeholder")
    p.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <data-dir>/gallery_set.html)")
    args = p.parse_args(argv)

    papers = build_papers(args.data_dir)
    if not papers:
        print(f"No paper summaries found under {args.data_dir} - run summarize.py first.")
        return 1

    total_results = sum(len(p["results"]) for p in papers)
    unique_pairs = {(r["dataset"], r["metric"]) for p in papers for r in p["results"]}
    task_counts: dict[str, int] = {}
    for paper in papers:
        task_counts[paper["task"]] = task_counts.get(paper["task"], 0) + 1

    print(f"{len(papers)} papers, {total_results} result rows, {len(unique_pairs)} unique dataset/metric buckets")
    for task, count in sorted(task_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {task}")

    output_path = args.output or (args.data_dir / "gallery_set.html")
    output_path.write_text(render_html(papers, args.template), encoding="utf-8")
    print(f"\nWrote {output_path} ({output_path.stat().st_size:,} bytes)")
    print("Publish it with Claude Code's Artifact tool (or ask Claude to) - this script only builds the file.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
