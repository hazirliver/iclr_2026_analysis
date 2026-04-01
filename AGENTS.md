# AGENTS.md — ICLR 2026 Paper Analysis

## Project Overview

End-to-end analysis of 5,358 accepted ICLR 2026 papers to surface interesting papers across multiple dimensions (not just highest-rated). Data sourced from OpenReview via papercopilot/paperlists.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv
- **Notebooks:** marimo (reactive, git-friendly `.py` files)
- **DataFrame:** Polars (primary, not pandas)
- **Visualization:** Plotly (interactive, preferred)
- **Embeddings:** Qwen/Qwen3-Embedding-8B (4096-dim, via vLLM on GPU VM)
- **ML/Clustering:** scikit-learn (Ward hierarchical, KMeans), UMAP
- **Code quality:** ty (type check)

## Data

- **Source:** `iclr_2026_raw.parquet` (16 MB) — raw conference data
- **Rows:** 5,358 accepted papers (Oral, ConditionalOral, Poster, ConditionalPoster)
- **Key columns:** review scores (rating, soundness, presentation, contribution, confidence) as per-reviewer lists, aggregated `_avg` columns as `[mean, std]` (ddof=0), word counts for review sections, keywords, abstracts, primary areas
- **Embeddings:** `embeddings.parquet` (64 MB) — Qwen3-Embedding-8B, 4096-dim, L2-normalized
- **PCA:** 4096D → 608 components (90% variance)
- **Clustering:** 50 Ward clusters (k=50), sizes 25–277, median 90

## Code Conventions

- Polars, not pandas. Use `pl.` prefix.
- Marimo notebooks are the primary code format. Run with `uv run python <notebook>.py`.
- One logical step per marimo cell.
- Set random seeds for reproducibility (SEED=42).
- No silent mutation of input data — always create new columns/frames.
- Comments explain *why*, not *what*.
- Save intermediate artifacts to parquet.
- When showing papers: always include title, rating_mean, status, primary_area, OpenReview link.

## Project Structure

```
├── marimo_notebooks/                  # Core analysis pipeline (marimo)
│   ├── try1.py                        # Data cleanup → iclr_2026_accepted.parquet
│   ├── 1_data_audit_and_features.py   # Validation + 42 features → iclr_2026_features.parquet
│   ├── 2_eda.py                       # Distributions, correlations → 21 HTML figures
│   ├── 3_embeddings.py                # PCA + UMAP + kNN → iclr_2026_embeddings.parquet
│   ├── 4_clustering.py                # Ward + KMeans → iclr_2026_clustered.parquet
│   ├── 5_interestingness.py           # 8 archetypes → iclr_2026_scored.parquet
│   └── 6_final_shortlist.py           # Diversified reading lists (printed)
├── iclr2026_download.py               # OpenReview API scraper
├── embed-script.py                    # vLLM async embedding client
├── figures/                           # 21 interactive Plotly HTML figures
├── *.parquet                          # Pipeline artifacts (see Data Pipeline below)
├── pyproject.toml                     # Dependencies (gpu extras for vLLM)
├── uv.lock                            # Locked deps
└── AGENTS.md                          # This file
```

### Legacy scripts (root-level `0_*.py` through `7_*.py`)

These are the original non-marimo scripts from the initial development. The `marimo_notebooks/` versions are the authoritative source now.

## Data Pipeline

```
iclr2026_download.py → (JSONL) → iclr_2026_raw.parquet
try1.py → iclr_2026_accepted.parquet (5,358 rows × 31 cols)
1_data_audit_and_features.py → iclr_2026_features.parquet (5,358 × 73)
embed-script.py (GPU VM) → embeddings.parquet (5,358 × 4096-dim)
3_embeddings.py → iclr_2026_embeddings.parquet (5,358 × 684: +608 PCA +UMAP +kNN)
4_clustering.py → iclr_2026_clustered.parquet (+cluster_ward, cluster_kmeans, bridge_ratio)
5_interestingness.py → iclr_2026_scored.parquet (+8 score columns)
6_final_shortlist.py → printed reading lists
2_eda.py → figures/*.html (runs independently on features.parquet)
```

## Analysis Phases

1. **Data Audit & Feature Engineering** — validate data, build flat analysis table (42 features)
2. **Exploratory Data Analysis** — distributions, correlations, effect sizes
3. **Embeddings & Semantic Structure** — Qwen3-Embedding-8B → PCA → UMAP, kNN density, near-duplicates
4. **Clustering & Theme Discovery** — Ward k=50 (primary) + KMeans (baseline), cluster labeling, bridge papers
5. **Interestingness Framework** — 8 archetypes: top overall, hidden gems, controversial, high engagement, semantically novel, bridge papers, area leaders, consensus standouts
6. **Final Shortlist** — diversified reading lists (30 papers, 15 areas, 18 clusters)

## Review Score Ranges

- rating: 1-10
- soundness: 1-4
- presentation: 1-4
- contribution: 1-4
- confidence: 1-5

## Key Findings

- **Status:** 95.6% Poster, 4.2% Oral, 0.3% Conditional
- **Reviewers:** 76% have 4 reviewers, 17% have 3, 7% have 5+
- **`_avg` columns use ddof=0** (population std)
- **Oral vs Poster effect sizes:** rating Cohen's d=+1.38, soundness d=+0.78, contribution d=+0.82 (large effects); confidence d=+0.09 (negligible)
- **Cluster structure:** 50 Ward clusters, well-balanced (25–277 papers), thematically coherent, rating spread 5.08–5.65 (clusters are topic-based, not quality-based)
