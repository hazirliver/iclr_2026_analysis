"""Phase 2: Exploratory Data Analysis.

Distributions, correlations, reviewer behavior, and area-level comparisons.
All plots saved as interactive HTML to figures/.
"""

from pathlib import Path

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats as sp_stats

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

df = pl.read_parquet("iclr_2026_features.parquet")
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Status breakdown ──────────────────────────────────────────────────
status_df = (
    df.group_by("status")
    .agg(pl.len().alias("count"))
    .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("pct"))
    .sort("count", descending=True)
)
print("\n=== Status Breakdown ===")
print(status_df)

fig = px.bar(
    status_df.to_pandas(),
    x="status",
    y="count",
    text="pct",
    title="Paper Count by Acceptance Status",
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.write_html(FIGURES / "status_breakdown.html")

# ── 2. Primary area distribution ─────────────────────────────────────────
area_stats = (
    df.group_by("primary_area")
    .agg(
        pl.len().alias("count"),
        pl.col("rating_mean").mean().alias("mean_rating"),
        pl.col("rating_mean").std().alias("std_rating"),
    )
    .sort("count", descending=True)
)
print(f"\n=== Primary Areas ({area_stats.shape[0]} unique) ===")
print(area_stats.head(15))

fig = px.bar(
    area_stats.to_pandas(),
    x="count",
    y="primary_area",
    orientation="h",
    color="mean_rating",
    color_continuous_scale="RdYlGn",
    title="Papers per Primary Area (colored by mean rating)",
)
fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
fig.write_html(FIGURES / "area_distribution.html")

# Tier distribution per area (top 15 areas)
top_areas = area_stats["primary_area"].head(15).to_list()
area_status = (
    df.filter(pl.col("primary_area").is_in(top_areas))
    .group_by("primary_area", "status")
    .len()
    .sort("primary_area")
)
fig = px.bar(
    area_status.to_pandas(),
    x="primary_area",
    y="len",
    color="status",
    barmode="stack",
    title="Status Distribution per Area (Top 15)",
)
fig.update_layout(xaxis_tickangle=-45, height=500)
fig.write_html(FIGURES / "area_status_distribution.html")

# ── 3. Score distributions ───────────────────────────────────────────────
score_cols = [
    "rating_mean",
    "soundness_mean",
    "presentation_mean",
    "contribution_mean",
    "confidence_mean",
]

# Violin plots by status
for col in score_cols:
    fig = px.violin(
        df.to_pandas(),
        x="status",
        y=col,
        box=True,
        points="outliers",
        title=f"{col} by Status",
        color="status",
    )
    fig.write_html(FIGURES / f"violin_{col}_by_status.html")

# Combined score distributions
fig = make_subplots(rows=1, cols=5, subplot_titles=score_cols)
for i, col in enumerate(score_cols, 1):
    fig.add_trace(
        go.Histogram(x=df[col].to_list(), name=col, nbinsx=30),
        row=1,
        col=i,
    )
fig.update_layout(
    title="Score Distributions (paper-level means)", height=350, width=1400
)
fig.write_html(FIGURES / "score_distributions.html")

# ── 4. Review verbosity ─────────────────────────────────────────────────
wc_cols = [
    "wc_review_mean",
    "wc_strengths_mean",
    "wc_weaknesses_mean",
    "wc_questions_mean",
    "wc_summary_mean",
]

fig = make_subplots(rows=1, cols=5, subplot_titles=wc_cols)
for i, col in enumerate(wc_cols, 1):
    fig.add_trace(
        go.Histogram(x=df[col].to_list(), name=col, nbinsx=30),
        row=1,
        col=i,
    )
fig.update_layout(title="Review Section Word Counts", height=350, width=1400)
fig.write_html(FIGURES / "review_verbosity.html")

# ── 5. Keyword frequency ────────────────────────────────────────────────
# Explode keywords, normalize, count
kw_counts = (
    df.select("keywords")
    .explode("keywords")
    .with_columns(pl.col("keywords").str.to_lowercase().str.strip_chars())
    .group_by("keywords")
    .len()
    .sort("len", descending=True)
)
print("\n=== Top 30 Keywords ===")
print(kw_counts.head(30))

fig = px.bar(
    kw_counts.head(30).to_pandas(),
    x="len",
    y="keywords",
    orientation="h",
    title="Top 30 Keywords",
)
fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
fig.write_html(FIGURES / "keyword_frequency.html")

# ══════════════════════════════════════════════════════════════════════════
# RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════

# ── 6. Score correlations (Spearman) ─────────────────────────────────────
all_numeric = score_cols + [
    "rating_std",
    "n_reviewers",
    "n_replies",
    "n_authors",
    "total_review_wc",
    "wc_review_mean",
    "wc_strengths_mean",
    "wc_weaknesses_mean",
    "wc_questions_mean",
    "strengths_weaknesses_ratio",
    "questions_review_ratio",
    "corr_rating_confidence",
]

numeric_df = df.select(all_numeric).to_pandas()
corr_matrix = numeric_df.corr(method="spearman")

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Spearman Correlation Matrix",
)
fig.update_layout(height=700, width=800)
fig.write_html(FIGURES / "correlation_heatmap.html")

# ── 7. Reviewer disagreement ────────────────────────────────────────────
print("\n=== Reviewer Disagreement (rating_std) ===")
print(df.select("rating_std").describe())

fig = px.histogram(
    df.to_pandas(),
    x="rating_std",
    nbins=30,
    title="Rating Standard Deviation Distribution",
)
fig.write_html(FIGURES / "rating_std_distribution.html")

# Disagreement vs rating level
fig = px.scatter(
    df.to_pandas(),
    x="rating_mean",
    y="rating_std",
    opacity=0.3,
    title="Rating Mean vs Std (Disagreement)",
    trendline="lowess",
)
fig.write_html(FIGURES / "disagreement_vs_rating.html")

# Disagreement vs review length
fig = px.scatter(
    df.to_pandas(),
    x="rating_std",
    y="wc_review_mean",
    opacity=0.3,
    title="Disagreement vs Review Length",
    trendline="lowess",
)
fig.write_html(FIGURES / "disagreement_vs_review_length.html")

# ── 8. Confidence vs rating ─────────────────────────────────────────────
print("\n=== corr_rating_confidence Distribution ===")
print(df.select("corr_rating_confidence").describe())

fig = px.histogram(
    df.to_pandas(),
    x="corr_rating_confidence",
    nbins=30,
    title="Per-Paper Rating-Confidence Correlation",
)
fig.write_html(FIGURES / "corr_rating_confidence_dist.html")

# Do confident reviewers rate higher or lower?
fig = px.scatter(
    df.to_pandas(),
    x="confidence_mean",
    y="rating_mean",
    opacity=0.3,
    title="Confidence Mean vs Rating Mean",
    trendline="ols",
)
fig.write_html(FIGURES / "confidence_vs_rating.html")

# ── 9. Review effort vs rating ──────────────────────────────────────────
fig = px.scatter(
    df.to_pandas(),
    x="rating_mean",
    y="wc_review_mean",
    opacity=0.3,
    title="Rating vs Mean Review Word Count",
    trendline="lowess",
)
fig.write_html(FIGURES / "review_effort_vs_rating.html")

# ── 10. Discussion activity vs rating ────────────────────────────────────
fig = px.scatter(
    df.to_pandas(),
    x="rating_mean",
    y="n_replies",
    opacity=0.3,
    title="Rating vs Reply Count",
    trendline="lowess",
)
fig.write_html(FIGURES / "discussion_vs_rating.html")

# ── 11. Strengths-to-weaknesses ratio ───────────────────────────────────
fig = px.scatter(
    df.to_pandas(),
    x="rating_mean",
    y="strengths_weaknesses_ratio",
    opacity=0.3,
    title="Rating vs Strengths/Weaknesses Word Count Ratio",
    trendline="lowess",
)
fig.write_html(FIGURES / "sw_ratio_vs_rating.html")

# ── 12. Area-level comparisons ──────────────────────────────────────────
area_comparison = (
    df.group_by("primary_area")
    .agg(
        pl.len().alias("count"),
        pl.col("rating_mean").mean().alias("mean_rating"),
        pl.col("rating_mean").median().alias("median_rating"),
        pl.col("soundness_mean").mean().alias("mean_soundness"),
        pl.col("contribution_mean").mean().alias("mean_contribution"),
        pl.col("rating_std").mean().alias("mean_disagreement"),
        pl.col("wc_review_mean").mean().alias("mean_review_wc"),
        pl.col("n_replies").mean().alias("mean_replies"),
    )
    .sort("count", descending=True)
)

fig = px.scatter(
    area_comparison.to_pandas(),
    x="mean_rating",
    y="mean_soundness",
    size="count",
    hover_name="primary_area",
    color="mean_contribution",
    color_continuous_scale="Viridis",
    title="Area Comparison: Rating vs Soundness (size=count, color=contribution)",
)
fig.write_html(FIGURES / "area_comparison.html")

# ── 13. Effect sizes: Oral vs Poster ────────────────────────────────────
print("\n=== Effect Sizes: Oral vs Poster ===")
oral = df.filter(pl.col("status") == "Oral")
poster = df.filter(pl.col("status") == "Poster")

for col in score_cols + ["rating_std", "wc_review_mean", "n_replies"]:
    o = oral[col].drop_nulls().to_numpy()
    p = poster[col].drop_nulls().to_numpy()

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(o) - 1) * o.std(ddof=1) ** 2 + (len(p) - 1) * p.std(ddof=1) ** 2)
        / (len(o) + len(p) - 2)
    )
    d = (o.mean() - p.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U (rank-biserial)
    u_stat, p_val = sp_stats.mannwhitneyu(o, p, alternative="two-sided")
    rank_biserial = 1 - (2 * u_stat) / (len(o) * len(p))

    print(
        f"  {col:30s}  Cohen's d={d:+.3f}  rank-biserial={rank_biserial:+.3f}  p={p_val:.2e}"
    )

# ── Summary ──────────────────────────────────────────────────────────────
n_figures = len(list(FIGURES.glob("*.html")))
print(f"\nSaved {n_figures} interactive figures to {FIGURES}/")
print("\nKey observations to investigate further:")
print("  - Check if high-disagreement papers cluster in certain areas")
print("  - Examine the relationship between strengths/weaknesses ratio and status")
print("  - Look for areas with unusually high or low mean confidence")
