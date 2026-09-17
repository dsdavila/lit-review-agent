# Metadata-First Literature Review Agent

Scrapes conference/arXiv/OpenReview **metadata only** (title, abstract, URL,
year), filters it against your research query in two cheap-then-smart
stages, and downloads PDFs only for the papers that actually survive the
filter. Never fetches a PDF speculatively. From there, two more stages turn
the downloaded PDFs into something you can actually read and compare:
per-paper Claude summaries with extracted results tables (`summarize.py`),
and a consolidated, normalized leaderboard page built from all of them
(`build_gallery_page.py`).

## Architecture

```
Scraper Layer        ->  Filter Layer (2 stages)          ->  Selective Downloader   ->  Summary Layer          ->  Leaderboard page
------------------       -----------------------------        --------------------       ---------------            ----------------
scrapers/*.py             1. Prefilter (local, free)            downloader.py              summarizer.py /            build_gallery_page.py
  arxiv                      case-insensitive search-term          only for papers          summarize.py                normalizes dataset/metric
  openreview                 matching against title/abstract       you confirmed             reads each PDF,             names + tags task category,
  acl_anthology               (deterministic - no scoring model,                             writes nutshell/            writes one self-contained
  cvf (CVPR/ICCV/WACV)        no embeddings, by default)                                      contributions/              HTML file (Gallery Set) -
  ecva (ECCV)                 -> shortlists everything that                                   deficiencies/notable        publish it as a Claude
  bmvc                           matched, capped at top-K                                     + results table per         Artifact to actually use it
  generic_html (template)   2. Judge (Claude - the LLM adjudicator)                            paper (cached JSON)
                                only scores the shortlist
                                -> calibrated 0-100 + reason
```

This is what keeps a large conference from blowing through Claude's context
budget: stage 1 runs over *every* scraped paper for free and is entirely
under your control (you supply the exact terms it matches on - no "latent
space" semantic guessing unless you explicitly opt into that); stage 2 only
ever sees the shortlist (default top 40 matches), regardless of how big the
conference is. Papers that match nothing in stage 1 are never sent to
Claude and never downloadable, no matter what `--top-n` you ask for.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, or run `ant auth login`
```

`sentence-transformers` is only needed if you opt into `--prefilter-method
embeddings`; the default `strings` prefilter has no dependency at all.
`anthropic` is needed for the Claude judge stage (skip with `--no-judge` to
run fully local). Both are in `requirements.txt` for convenience but the
tool runs without either - see "Fallback behavior" below.

## Usage

Single conference:

```bash
python main.py \
  --url https://openreview.net/group?id=ICLR.cc/2024/Conference \
  --query "efficient inference for large language models" \
  --search-terms "mixture of experts,MoE,sparse routing,expert routing"
```

Batch, across a flat list of conference repos - each gets its own output
folder (see `examples/conferences.example.json`):

```bash
python main.py --config examples/conferences.example.json
```

Flow:

1. Fetch the index page/API for each conference -> save `metadata.json`.
2. Prefilter every scraped paper: case-insensitive match against your
   `--search-terms` in title (+ abstract by default) - deterministic, no
   scoring model. (`--prefilter-method embeddings` opts into semantic
   cosine-similarity search from `--query` alone instead, if you want it.)
3. Claude (the LLM adjudicator) judges only the prefilter shortlist ->
   calibrated relevance score + reason.
4. Print the sorted candidate list; prompt you to confirm (or type ranks to
   hand-pick, e.g. `1,3,7`).
5. Download only the confirmed PDFs.

**`--search-terms` is required** for the default `strings` prefilter - it's
a comma-separated, case-insensitive list of terms/phrases matched as plain
substrings (not whole-word) against each paper's title/abstract; a paper is
shortlisted if *any* term hits (`--match-all` requires *all* of them). In
`--config` mode, put the same list in the config's top-level `search_terms`
array instead of repeating it on the command line. `--query` is separate -
it's the natural-language research question handed to Claude in stage 2,
not what stage 1 searches on.

### Computer vision venues (CVPR / WACV / ICCV / ECCV / BMVC)

`examples/cv_conferences.example.json` is a ready-to-run config covering the
last ~3 instances of CVPR, WACV, ICCV, ECCV, and BMVC (ICCV/ECCV are
biennial, so "3 years" is their last 2 editions). Copy it, fill in `query`,
and run:

```bash
python main.py --config examples/cv_conferences.example.json --query "your topic"
```

A few things worth knowing about these five:

- **CVPR/ICCV/WACV** (`scrapers/cvf.py`) are all hosted on
  `openaccess.thecvf.com` - point at the `...?day=all` listing URL for a
  given venue+year (e.g. `CVPR2024?day=all`), not the bare venue page.
- **ECCV** (`scrapers/ecva.py`) is hosted on `www.ecva.net`, which lists
  every year on one page - select a year with a `?year=YYYY` query param
  (this is a local convention this scraper reads; the real site ignores it
  and returns the same full page either way).
- **BMVC** (`scrapers/bmvc.py`) moves to a new domain almost every year
  (`proceedings.bmvc2023.org`, `bmvc2024.org/...`, `bmvc2025.bmva.org/...`)
  with no predictable pattern, so there's no URL formula - the config just
  hard-codes each year's actual proceedings URL. Recent years (2025+)
  publish an abstract per paper; 2023/2024 only publish title + authors +
  PDF, so scoring for those years works off the title alone.
- **CVPR/ICCV/WACV/ECCV fetch one extra request per paper** to get its
  abstract, so scraping is bounded by `--limit` (default 200/conference),
  not the full conference size (CVPR alone runs ~3,000-4,000 papers/year).
  Start with a small `--limit` to test, then raise it - a full run across
  all 13 entries at a high limit will take a while (sequential requests,
  polite retry backoff included).

## Output layout

```
data/
  iclr2024/
    metadata.json     # every scraped paper, with prefilter_score/reason,
                       # judge_score/reason, shortlisted, final_score
    filtering.log      # why each paper was cut / why a download failed
    papers/
      <slug>.pdf        # only the papers you confirmed
    summaries/
      <slug>.json        # one per downloaded paper - see "Summarizing papers"
  acl2024-long/
    ...
  run_summary.json      # written for multi-conference (--config) runs
  SUMMARIES.md           # combined per-paper write-ups, all conferences
  LEADERBOARD.md         # extracted results grouped by dataset+metric
  gallery_set.html       # the interactive leaderboard page - see below
```

## Key flags

| Flag | Default | What |
|---|---|---|
| `--url` / `--config` | - | one conference vs. a batch (see above) |
| `--query` | - | natural-language research question, passed to the Claude judge stage |
| `--search-terms` | - | comma-separated, case-insensitive terms/phrases for the `strings` prefilter (required unless `--prefilter-method embeddings`) |
| `--search-fields` | `title,abstract` | which fields `--search-terms` matches against |
| `--match-all` | off | require *all* `--search-terms` to hit, instead of any one |
| `--prefilter-method` | `strings` | `strings` (deterministic search-term matching) / `embeddings` (semantic cosine similarity, opt-in) |
| `--prefilter-k` | `40` | how many top prefilter matches reach the Claude judge stage |
| `--no-judge` | off | skip Claude entirely, rank on the local prefilter score only |
| `--claude-model` | `$ANTHROPIC_MODEL` or `claude-opus-5` | judge model |
| `--top-n` | `10` | how many top-ranked papers to propose downloading |
| `--limit` | `200` | max metadata records fetched per conference |
| `-y` / `--yes` | off | skip the confirmation prompt |
| `--skip-scrape` | off | reuse a conference's existing `metadata.json` |

## Summarizing papers

Once you've downloaded PDFs (the flow above), `summarize.py` reads each one
with Claude and produces a first-glance academic summary - the kind of skim a
reviewer does before a close read, not a full deep review:

```bash
python summarize.py                          # every conference under data/
python summarize.py --conference cvpr2024     # just one (repeatable)
python summarize.py --force                   # regenerate cached summaries too
python summarize.py --concurrency 8 --effort high
```

For each downloaded paper it writes `data/{conference}/summaries/{slug}.json`
with five fields:

- `nutshell` - the big idea in 1-3 sentences
- `contributions` - the actual, concrete novelty (not the paper's own marketing language)
- `deficiencies` - genuine limitations a critical reviewer would flag, grounded in the paper's own numbers where possible
- `notable` - anything else worth flagging (citation errors, cross-paper naming collisions, a clever trick)
- `results` - the paper's own headline numbers from its main results table: a list of `{dataset, metric, value, setting}` rows (its proposed method's numbers only, never competitors' - empty if the paper has no such table, e.g. a pure benchmark/dataset paper)

Summaries are cached one JSON file per paper, so reruns are cheap and
resumable - only papers without a cached summary (or run with `--force`) hit
the API again. After summarizing, it writes two combined docs:

- `data/SUMMARIES.md` - every paper's full write-up, grouped by conference
- `data/LEADERBOARD.md` - every extracted `results` row grouped by
  dataset+metric, sorted best-to-worst, so you can compare methods across
  papers. This is a raw, unnormalized view (see the next section for the
  cleaned-up version) - useful for eyeballing, not for citing.

**Key flags:** `--model` (default `$ANTHROPIC_MODEL` or `claude-opus-5`),
`--effort` (`low`/`medium`/`high`/`xhigh`/`max`, default `medium`),
`--concurrency` (parallel Claude calls, default 4), `--output` (markdown path
override).

**Cost note:** this calls the Claude API directly (same billing as
`main.py`'s judge stage) - it needs `ANTHROPIC_API_KEY` or `ant auth login`,
and API usage is billed separately from a Claude.ai/Claude Code subscription.
Each call sends the full PDF, so a large batch adds up - test on one
conference (`--conference`) before running the whole corpus.

**Known limits:** PDFs over 24MB are skipped (base64 inline request limit -
see `MAX_PDF_BYTES` in `summarizer.py`); those log as failed rather than
crashing the run. If you're running this via Claude Code, Claude can also
just read the PDFs directly with its own Read tool (which handles larger
files via text extraction) and write the summary JSON itself, skipping the
API call/billing path entirely - ask it to.

## Building the leaderboard page

`build_gallery_page.py` turns every cached summary into one interactive HTML
page - "Gallery Set": a searchable, filterable list of every paper, one card
each, showing its benchmark pills (dataset + headline metric values) up
front. Filter by task/dataset/metric/year and search by title/author; pick a
dataset *and* metric to rank the list by that specific benchmark value
instead of by relevance/year/title. Every card expands to the paper's full
critical summary. (Earlier versions had a separate row-per-result leaderboard
table - dropped in favor of one paper-per-card view, since a paper reporting
several settings/protocols on the same benchmark produced a noisy repeated
row per variant there.)

```bash
python build_gallery_page.py                                # writes data/gallery_set.html
python build_gallery_page.py --output somewhere/else.html
```

This is pure local data processing - no API calls, no cost - reading
`data/{conference}/summaries/*.json` and applying light normalization on top
of the raw `results` rows:

- **Dataset names**: grouped case/punctuation/whitespace-insensitively (fixes
  "Market-1501" vs "Market1501" vs "Market 1501"). Does **not** attempt
  semantic merges of genuinely-differently-spelled variants (e.g. "AG-ReID"
  vs "AG-ReID.v1" stay separate) - that needs human judgment call, not string
  matching.
- **Metric names**: a small, high-confidence alias table merges the Rank-K
  family only (`Rank-1`/`R1`/`R@1`/`CMC-1`/`Top-1`/... all mean the same
  thing in ReID). Anything containing "lower" (inverted/attack/privacy
  metrics, where a low score is the good outcome) is deliberately never
  merged into the normal ranking - conflating those would silently sort
  attack-success rates as if they were legitimate top scores.
- **Task tagging**: every paper gets one category (Visible-Infrared ReID,
  Clothes-Changing ReID, Vehicle ReID, Animal/Wildlife Re-ID, Multi-Object
  Tracking, ...) via keyword rules against its own title+nutshell text, with
  an honest "Other CV Research" bucket for papers pulled in by the broader
  search terms that were never really re-identification work (bird pecking,
  sports field reconstruction, neuron re-ID, etc.). Edit `TASK_RULES` in
  `build_gallery_page.py` to add/adjust categories.

The page itself lives in `gallery_set_template.html` (all CSS/JS, with a
`__DATA_JSON__` placeholder the script splices the consolidated dataset into)
- edit that file to change layout/styling, the Python script only owns the
data pipeline.

**Publishing it is a separate, manual step** - this script only writes the
HTML file to disk; there's no API for publishing a Claude Artifact. Open
Claude Code in this repo and ask it to publish `data/gallery_set.html` as an
Artifact (or, if you're already in a Claude Code session, just ask it
directly - that's how this page was built and updated the first time). Once
published, re-running the build script and asking Claude to republish to the
same Artifact URL updates it in place.

**This is a "quick and dirty" leaderboard, not a citable benchmark table** -
say so on the page itself, and it's worth repeating here: values come from an
LLM reading each paper's tables, protocols/backbones/splits vary even within
one dataset+metric pair (check the `setting` field), and the normalization
above is conservative on purpose. Treat it as a fast way to see what's been
reported and jump to the source paper, not as ground truth to cite from
directly.

## Adding a new conference scraper

Copy `scrapers/generic_html.py` (config-driven, no code) for a plain HTML
listing site, or `scrapers/arxiv.py` / `scrapers/openreview.py` (real API
clients) as a template for a site with an API. Then:

1. Create `scrapers/<name>.py` with a `BaseScraper` subclass, `name = "..."`,
   `can_handle(url)`, and `fetch_metadata(url, limit)`.
2. Decorate the class with `@register_scraper`.
3. Add one import line to `scrapers/__init__.py`.

That's the whole integration surface - `main.py`, the evaluator, and the
downloader never change.

## Fallback behavior (no hard dependency on network/API keys)

- **Prefilter**: the default `strings` method has no dependency at all - it's
  plain Python substring matching, always available, fully deterministic.
  `--prefilter-method embeddings` is opt-in and needs `sentence-transformers`
  installed and a model downloadable; if either isn't available it raises a
  clear error rather than silently swapping methods on you.
- **Judge**: if `anthropic` isn't installed, `ANTHROPIC_API_KEY` isn't set,
  or a batch call fails, those papers simply keep their prefilter score
  instead of erroring the run. Pass `--no-judge` to skip Claude on purpose
  (fully local, no API key needed).
