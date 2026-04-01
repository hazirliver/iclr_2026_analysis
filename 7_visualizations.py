"""Phase 7: Visualizations.

High-value interactive plots: score distributions, correlations,
UMAP scatters, cluster summaries, and ranked tables.
"""

from pathlib import Path

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

df = pl.read_parquet("iclr_2026_scored.parquet")
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

has_umap = "umap_x" in df.columns
has_clusters = "cluster_ward" in df.columns
pdf = df.to_pandas()

# ══════════════════════════════════════════════════════════════════════════
# 1. Score distributions: violin/ridge by status
# ══════════════════════════════════════════════════════════════════════════

score_cols = [
    "rating_mean",
    "soundness_mean",
    "presentation_mean",
    "contribution_mean",
    "confidence_mean",
]

fig = make_subplots(rows=1, cols=5, subplot_titles=score_cols)
for i, col in enumerate(score_cols, 1):
    for status in pdf["status"].unique():
        subset = pdf[pdf["status"] == status]
        fig.add_trace(
            go.Violin(
                y=subset[col],
                name=status,
                legendgroup=status,
                showlegend=(i == 1),
                box_visible=True,
                meanline_visible=True,
            ),
            row=1,
            col=i,
        )
fig.update_layout(
    title="Score Distributions by Status",
    height=500,
    width=1600,
    violinmode="overlay",
)
fig.write_html(FIGURES / "scores_by_status_violin.html")
print("✓ Score violin plots by status")

# ── By area (top 10 areas) ──────────────────────────────────────────────
top_areas = pdf["primary_area"].value_counts().head(10).index.tolist()
area_pdf = pdf[pdf["primary_area"].isin(top_areas)]

for col in ["rating_mean", "soundness_mean", "contribution_mean"]:
    fig = px.violin(
        area_pdf,
        x="primary_area",
        y=col,
        box=True,
        color="primary_area",
        title=f"{col} by Area (Top 10)",
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    fig.write_html(FIGURES / f"{col}_by_area_violin.html")
print("✓ Score violin plots by area")

# ══════════════════════════════════════════════════════════════════════════
# 2. Correlation heatmap of all scalar review features
# ══════════════════════════════════════════════════════════════════════════

numeric_cols = [
    "rating_mean",
    "rating_std",
    "soundness_mean",
    "presentation_mean",
    "contribution_mean",
    "confidence_mean",
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
corr = pdf[numeric_cols].corr(method="spearman")

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Spearman Correlation: All Scalar Review Features",
)
fig.update_layout(height=700, width=800)
fig.write_html(FIGURES / "full_correlation_heatmap.html")
print("✓ Full correlation heatmap")

# ══════════════════════════════════════════════════════════════════════════
# 3. UMAP scatters (if embeddings available)
# ══════════════════════════════════════════════════════════════════════════

if has_umap:
    # By cluster
    if has_clusters:
        pdf["cluster_label"] = pdf["cluster_ward"].apply(lambda x: f"C{x}")
        fig = px.scatter(
            pdf,
            x="umap_x",
            y="umap_y",
            color="cluster_label",
            hover_name="title",
            hover_data=["rating_mean", "primary_area", "status"],
            title="UMAP: Colored by HDBSCAN Cluster",
            opacity=0.6,
        )
        fig.update_layout(height=700, width=900)
        fig.write_html(FIGURES / "umap_by_cluster.html")

    # By primary area
    fig = px.scatter(
        pdf,
        x="umap_x",
        y="umap_y",
        color="primary_area",
        hover_name="title",
        hover_data=["rating_mean", "status"],
        title="UMAP: Colored by Primary Area",
        opacity=0.5,
    )
    fig.update_layout(height=700, width=1000)
    fig.write_html(FIGURES / "umap_by_area.html")

    # By rating mean
    fig = px.scatter(
        pdf,
        x="umap_x",
        y="umap_y",
        color="rating_mean",
        hover_name="title",
        hover_data=["primary_area", "status"],
        color_continuous_scale="RdYlGn",
        title="UMAP: Colored by Rating Mean",
        opacity=0.6,
    )
    fig.update_layout(height=700, width=900)
    fig.write_html(FIGURES / "umap_by_rating.html")

    # By interestingness archetype (top overall score)
    fig = px.scatter(
        pdf,
        x="umap_x",
        y="umap_y",
        color="score_top_overall",
        hover_name="title",
        hover_data=["rating_mean", "primary_area", "status"],
        color_continuous_scale="Plasma",
        title="UMAP: Colored by Top Overall Score",
        opacity=0.6,
    )
    fig.update_layout(height=700, width=900)
    fig.write_html(FIGURES / "umap_by_interestingness.html")

    print("✓ UMAP scatter plots (cluster, area, rating, interestingness)")
else:
    print("⚠ Skipping UMAP plots (no embeddings)")

# ══════════════════════════════════════════════════════════════════════════
# 4. Cluster size + quality summary
# ══════════════════════════════════════════════════════════════════════════

if has_clusters:
    cluster_summary = (
        df.group_by("cluster_ward")
        .agg(
            pl.len().alias("size"),
            pl.col("rating_mean").mean().alias("mean_rating"),
            pl.col("soundness_mean").mean().alias("mean_soundness"),
            pl.col("contribution_mean").mean().alias("mean_contribution"),
            pl.col("rating_std").mean().alias("mean_disagreement"),
        )
        .sort("size", descending=True)
    )

    cluster_pdf = cluster_summary.to_pandas()
    fig = px.scatter(
        cluster_pdf,
        x="size",
        y="mean_rating",
        size="size",
        color="mean_contribution",
        hover_data=["cluster_ward", "mean_soundness", "mean_disagreement"],
        color_continuous_scale="Viridis",
        title="Cluster Size vs Mean Rating (color=contribution)",
        text="cluster_ward",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500, width=700)
    fig.write_html(FIGURES / "cluster_summary.html")
    print("✓ Cluster summary scatter")

# ══════════════════════════════════════════════════════════════════════════
# 5. Area summary table
# ══════════════════════════════════════════════════════════════════════════

area_summary = (
    df.group_by("primary_area")
    .agg(
        pl.len().alias("count"),
        pl.col("rating_mean").mean().alias("mean_rating"),
        pl.col("rating_mean").median().alias("median_rating"),
        pl.col("soundness_mean").mean().alias("mean_soundness"),
        pl.col("contribution_mean").mean().alias("mean_contribution"),
        pl.col("rating_std").mean().alias("mean_disagreement"),
        pl.col("n_replies").mean().alias("mean_replies"),
        (pl.col("status") == "Oral").sum().alias("n_oral"),
    )
    .with_columns(
        (pl.col("n_oral") / pl.col("count") * 100).alias("oral_pct"),
    )
    .sort("count", descending=True)
)

fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=area_summary.columns,
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[area_summary[c].to_list() for c in area_summary.columns],
                fill_color="lavender",
                align="left",
                format=[
                    None,
                    None,
                    ".2f",
                    ".2f",
                    ".2f",
                    ".2f",
                    ".2f",
                    ".1f",
                    None,
                    ".1f",
                ],
            ),
        )
    ]
)
fig.update_layout(title="Area Summary Table", height=600, width=1200)
fig.write_html(FIGURES / "area_summary_table.html")
print("✓ Area summary table")

# ══════════════════════════════════════════════════════════════════════════
# 6. Archetype score distributions
# ══════════════════════════════════════════════════════════════════════════

archetype_cols = [c for c in df.columns if c.startswith("score_")]
if archetype_cols:
    fig = make_subplots(
        rows=1,
        cols=len(archetype_cols),
        subplot_titles=[c.replace("score_", "") for c in archetype_cols],
    )
    for i, col in enumerate(archetype_cols, 1):
        fig.add_trace(
            go.Histogram(x=df[col].to_list(), nbinsx=40, name=col),
            row=1,
            col=i,
        )
    fig.update_layout(
        title="Archetype Score Distributions",
        height=350,
        width=300 * len(archetype_cols),
        showlegend=False,
    )
    fig.write_html(FIGURES / "archetype_distributions.html")
    print("✓ Archetype score distributions")

# ── Done ─────────────────────────────────────────────────────────────────
n_figures = len(list(FIGURES.glob("*.html")))
print(f"\nTotal interactive figures in {FIGURES}/: {n_figures}")
