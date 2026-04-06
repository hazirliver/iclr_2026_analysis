# AGENTS.md — ICLR 2026 Paper Analysis

## Project Overview

End-to-end analysis of 5,358 accepted ICLR 2026 papers to surface interesting papers across multiple dimensions. Data sourced from OpenReview.

## Tech Stack

- **Python 3.13+**, **uv** package manager
- **Notebooks:** marimo (reactive, git-friendly `.py` files) — the core pipeline
- **DataFrame:** Polars (not pandas)
- **Visualization:** Plotly (interactive)
- **Embeddings:** Qwen/Qwen3-Embedding-8B (4096-dim, via vLLM on 8×H200 GPU VM)
- **Clustering:** Ward hierarchical (primary), KMeans (baseline), UMAP for viz
- **Code quality:** ruff (lint/format), ty (type check), prek (pre-commit)

## Code Conventions

- Polars, not pandas. Use `pl.` prefix.
- Marimo notebooks in `marimo_notebooks/` are the primary code. Run: `uv run python marimo_notebooks/<name>.py`
- Marimo constraint: no duplicate variable names across cells (use unique names per cell).
- SEED=42 for reproducibility.
- No silent mutation of input data.
- Save intermediate artifacts to parquet.
- When showing papers: always include title, rating_mean, status, primary_area, OpenReview link.
- Run `uv run --dev prek --all-files` before committing. `figures/` is excluded from checks.

## Project Structure

```
├── marimo_notebooks/                  # Core analysis pipeline
│   ├── try1.py                        # Data cleanup → iclr_2026_accepted.parquet
│   ├── 1_data_audit_and_features.py   # Validation + 42 features → iclr_2026_features.parquet
│   ├── 1_5_keyword_canonicalization.py# Embed+cluster keywords → keyword_mapping.parquet
│   ├── 2_eda.py                       # Distributions, correlations → figures/*.html
│   ├── 3_embeddings.py                # PCA + UMAP + kNN → iclr_2026_embeddings.parquet
│   ├── 4_clustering.py                # Ward k=50 → iclr_2026_clustered.parquet
│   ├── 5_interestingness.py           # 8 archetypes → iclr_2026_scored.parquet
│   └── 6_final_shortlist.py           # Diversified reading lists (printed)
├── embed-script.py                    # vLLM async embedding client (papers)
├── embed-keywords-script.py           # vLLM async embedding client (keywords)
├── iclr2026_download.py               # OpenReview API scraper
├── 0-7_*.py                           # Legacy root-level scripts (superseded by marimo_notebooks/)
├── figures/                           # 21 interactive Plotly HTML figures
├── *.parquet                          # Pipeline data artifacts
└── pyproject.toml                     # Dependencies (gpu extras for vLLM)
```

## Data Pipeline

```
iclr2026_download.py → iclr_2026_raw.parquet (16MB)
try1.py → iclr_2026_accepted.parquet (5,358 × 31)
1_data_audit_and_features.py → iclr_2026_features.parquet (5,358 × 73)
embed-script.py (GPU) → embeddings.parquet (5,358 × 4096-dim, 64MB)
embed-keywords-script.py (GPU) → keyword_embeddings.parquet (~9K keywords)
1_5_keyword_canonicalization.py → keyword_mapping.parquet (9,246 → 326 canonical)
3_embeddings.py → iclr_2026_embeddings.parquet (5,358 × 684: +608 PCA +UMAP +kNN)
4_clustering.py → iclr_2026_clustered.parquet (+cluster_ward, cluster_kmeans, bridge_ratio)
5_interestingness.py → iclr_2026_scored.parquet (+8 score columns, 696 total cols)
6_final_shortlist.py → printed reading lists (30 papers, 15 areas, 18 clusters)
2_eda.py → figures/*.html (21 figures, runs on features.parquet)
```

## Key Data Facts

- **5,358 papers**: 95.6% Poster, 4.2% Oral, 0.3% Conditional
- **Reviewers**: 76% have 4, 17% have 3, 7% have 5+
- **`_avg` columns use ddof=0** (population std)
- **Embeddings**: Qwen3-Embedding-8B → 4096-dim → PCA to 608 components (90% variance)
- **Clusters**: 50 Ward clusters, sizes 25–277 (median 90), rating spread 5.08–5.65
- **Keywords**: 9,246 unique → 326 canonical (via embedding clustering + manual abbreviation merges)
- **Score columns**: score_top_overall, score_hidden_gem, score_controversial, score_high_engagement, score_semantic_novel, score_bridge, score_area_leader, score_consensus

## Review Score Ranges

- rating: 1-10, soundness: 1-4, presentation: 1-4, contribution: 1-4, confidence: 1-5
