# AGENTS.md — ICLR 2026 Paper Analysis

## Project Overview

End-to-end analysis of 5,358 accepted ICLR 2026 papers to surface interesting papers across multiple dimensions (not just highest-rated). Data sourced from OpenReview via papercopilot/paperlists.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv
- **DataFrame:** Polars (primary, not pandas)
- **Visualization:** Plotly (interactive, preferred), Matplotlib/Seaborn (static)
- **ML/Clustering:** scikit-learn, HDBSCAN, UMAP
- **Code quality:** Ruff (lint/format), ty (type check), pre-commit hooks

## Data

- **Source:** `iclr_2026_raw.parquet` (21 MB) — raw conference data
- **Rows:** 5,358 accepted papers (Oral, ConditionalOral, Poster, ConditionalPoster)
- **Key columns:** review scores (rating, soundness, presentation, contribution, confidence) as per-reviewer lists, aggregated `_avg` columns as `[mean, std]`, word counts for review sections, keywords, abstracts, primary areas
- **No embeddings yet** — to be computed in Phase 3

## Code Conventions

- Polars, not pandas. Use `pl.` prefix.
- One logical step per cell/script section.
- Set random seeds for reproducibility.
- No silent mutation of input data — always create new columns/frames.
- Comments explain *why*, not *what*.
- Save intermediate artifacts to parquet.
- When showing papers: always include title, rating_mean, status, primary_area, OpenReview link.

## Project Structure

```
├── 0_First_glimpse_at_the_data.py    # Initial data fetch + cleaning
├── iclr_2026_raw.parquet             # Raw data (21 MB)
├── pyproject.toml                    # Dependencies
├── uv.lock                           # Locked deps
├── .pre-commit-config.yml            # Code quality hooks
└── AGENTS.md                         # This file
```

## Analysis Phases

1. **Data Audit & Feature Engineering** — validate data, build flat analysis table
2. **Exploratory Data Analysis** — distributions, correlations, relationships
3. **Embeddings & Semantic Structure** — compute embeddings, nearest neighbors, outliers
4. **Clustering & Theme Discovery** — HDBSCAN + KMeans, cluster labeling
5. **Interestingness Framework** — 9 archetypes (top overall, hidden gems, controversial, etc.)
6. **Final Shortlist** — diversified reading lists
7. **Visualizations** — UMAP scatters, violin plots, correlation heatmaps, tables

## Review Score Ranges

- rating: 1-10
- soundness: 1-4
- presentation: 1-4
- contribution: 1-4
- confidence: 1-5
