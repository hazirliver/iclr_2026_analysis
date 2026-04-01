import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy import stats as sp_stats

    FIGURES = Path("figures")
    FIGURES.mkdir(exist_ok=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    df = pl.read_parquet("iclr_2026_features.parquet")
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.1 Status breakdown
    """)
    return


@app.cell
def _(df):
    status_df = (
        df.group_by("status")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("pct"))
        .sort("count", descending=True)
    )
    status_df
    return (status_df,)


@app.cell
def _(status_df):
    fig1 = px.bar(
        status_df,
        x="status",
        y="count",
        text="pct",
        title="Paper Count by Acceptance Status",
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1
    return (fig1,)


@app.cell
def _(fig1):
    fig1.write_html(FIGURES / "status_breakdown.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.2 Primary area distribution
    """)
    return


@app.cell
def _(df):
    area_stats = (
        df.group_by("primary_area")
        .agg(
            pl.len().alias("count"),
            pl.col("rating_mean").mean().alias("mean_rating"),
            pl.col("rating_mean").std().alias("std_rating"),
        )
        .sort("count", descending=True)
    )
    area_stats
    return (area_stats,)


@app.cell
def _(area_stats):
    fig2 = px.bar(
        area_stats,
        x="count",
        y="primary_area",
        orientation="h",
        color="mean_rating",
        color_continuous_scale="RdYlGn",
        title="Papers per Primary Area (colored by mean rating)",
    )
    fig2.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    fig2
    return (fig2,)


@app.cell
def _(fig2):
    fig2.write_html(FIGURES / "area_distribution.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.3 Tier distribution per area (top 15 areas)
    """)
    return


@app.cell
def _(area_stats, df):
    top_areas = area_stats["primary_area"].head(15).to_list()
    area_status = (
        df.filter(pl.col("primary_area").is_in(top_areas))
        .group_by("primary_area", "status")
        .len()
        .sort("primary_area")
    )
    area_status
    return (area_status,)


@app.cell
def _(area_status):
    fig3 = px.bar(
        area_status.to_pandas(),
        x="primary_area",
        y="len",
        color="status",
        barmode="stack",
        title="Status Distribution per Area (Top 15)",
    )
    fig3.update_layout(xaxis_tickangle=-45, height=500)
    fig3
    return (fig3,)


@app.cell
def _(fig3):
    fig3.write_html(FIGURES / "area_status_distribution.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.4 Score distributions
    """)
    return


@app.cell
def _():
    score_cols = [
        "rating_mean",
        "soundness_mean",
        "presentation_mean",
        "contribution_mean",
        "confidence_mean",
    ]
    return (score_cols,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.4.1 Violin plots by status
    """)
    return


@app.cell
def _(df, score_cols):
    for col in score_cols:
        fig5 = px.violin(
            df.to_pandas(),
            x="status",
            y=col,
            box=True,
            points="outliers",
            title=f"{col} by Status",
            color="status",
        )
        fig5.write_html(FIGURES / f"violin_{col}_by_status.html")
    return (fig5,)


@app.cell
def _(fig5):
    fig5
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.4.2 Combined score distributions
    """)
    return


@app.cell
def _(df, score_cols):
    fig6 = make_subplots(rows=1, cols=5, subplot_titles=score_cols)
    for i, _col in enumerate(score_cols, 1):
        fig6.add_trace(
            go.Histogram(x=df[_col].to_list(), name=_col, nbinsx=35),
            row=1,
            col=i,
        )
    fig6.update_layout(
        title="Score Distributions (paper-level means)", height=350, width=1400
    )
    fig6
    return (fig6,)


@app.cell
def _(fig6):
    fig6.write_html(FIGURES / "score_distributions.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.5 Review verbosity
    """)
    return


@app.cell
def _(df):
    wc_cols = [
        "wc_review_mean",
        "wc_strengths_mean",
        "wc_weaknesses_mean",
        "wc_questions_mean",
        "wc_summary_mean",
    ]

    fig7 = make_subplots(rows=1, cols=5, subplot_titles=wc_cols)
    for _i, _col in enumerate(wc_cols, 1):
        fig7.add_trace(
            go.Histogram(x=df[_col].to_list(), name=_col, nbinsx=30),
            row=1,
            col=_i,
        )
    fig7.update_layout(title="Review Section Word Counts", height=350, width=1400)
    fig7
    return (fig7,)


@app.cell
def _(fig7):
    fig7.write_html(FIGURES / "review_verbosity.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.6 Keyword frequency (canonical)

    Uses canonicalized keywords from notebook 1.5 (synonym clusters merged via
    embedding-based agglomerative clustering).
    """)
    return


@app.cell
def _(df):
    kw_counts = (
        df.select("canonical_keywords")
        .explode("canonical_keywords")
        .filter(pl.col("canonical_keywords") != "")
        .group_by("canonical_keywords")
        .len()
        .sort("len", descending=True)
    )
    kw_counts
    return (kw_counts,)


@app.cell
def _(kw_counts):
    fig8 = px.bar(
        kw_counts.head(30).to_pandas(),
        x="len",
        y="canonical_keywords",
        orientation="h",
        title="Top 30 Canonical Keywords",
    )
    fig8.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    fig8
    return (fig8,)


@app.cell
def _(fig8):
    fig8.write_html(FIGURES / "keyword_frequency.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Relationships
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 Score correlations (Spearman)
    """)
    return


@app.cell
def _(df, score_cols):
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

    fig9 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Spearman Correlation Matrix",
    )
    fig9.update_layout(height=700, width=800)
    fig9
    return (fig9,)


@app.cell
def _(fig9):
    fig9.write_html(FIGURES / "correlation_heatmap.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.2 Reviewer disagreement
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Reviewer Disagreement (`rating_std`)
    """)
    return


@app.cell
def _(df):
    df.select("rating_std").describe()
    return


@app.cell
def _(df):
    fig10 = px.histogram(
        df.to_pandas(),
        x="rating_std",
        nbins=30,
        title="Rating Standard Deviation Distribution",
    )
    fig10
    return (fig10,)


@app.cell
def _(fig10):
    fig10.write_html(FIGURES / "rating_std_distribution.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Disagreement vs rating level
    """)
    return


@app.cell
def _(df):
    fig11 = px.scatter(
        df,
        x="rating_mean",
        y="rating_std",
        opacity=0.3,
        title="Rating Mean vs Std (Disagreement)",
        trendline="lowess",
    )
    fig11
    return (fig11,)


@app.cell
def _(fig11):
    fig11.write_html(FIGURES / "disagreement_vs_rating.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Disagreement vs review length
    """)
    return


@app.cell
def _(df):
    fig12 = px.scatter(
        df.to_pandas(),
        x="rating_std",
        y="wc_review_mean",
        opacity=0.3,
        title="Disagreement vs Review Length",
        trendline="lowess",
    )
    fig12
    return (fig12,)


@app.cell
def _(fig12):
    fig12.write_html(FIGURES / "disagreement_vs_review_length.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.3 Confidence vs rating
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `corr_rating_confidence` Distribution
    """)
    return


@app.cell
def _(df):
    df.select("corr_rating_confidence").describe()
    return


@app.cell
def _(df):
    fig13 = px.histogram(
        df,
        x="corr_rating_confidence",
        nbins=50,
        title="Per-Paper Rating-Confidence Correlation",
    )
    fig13
    return (fig13,)


@app.cell
def _(fig13):
    fig13.write_html(FIGURES / "corr_rating_confidence_dist.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Do confident reviewers rate higher or lower?
    """)
    return


@app.cell
def _(df):
    fig14 = px.scatter(
        df.to_pandas(),
        x="confidence_mean",
        y="rating_mean",
        opacity=0.3,
        title="Confidence Mean vs Rating Mean",
        trendline="ols",
    )
    fig14
    return (fig14,)


@app.cell
def _(fig14):
    fig14.write_html(FIGURES / "confidence_vs_rating.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.4 Review effort vs rating
    """)
    return


@app.cell
def _(df):
    fig15 = px.scatter(
        df,
        x="rating_mean",
        y="wc_review_mean",
        opacity=0.3,
        title="Rating vs Mean Review Word Count",
        trendline="lowess",
    )
    fig15
    return (fig15,)


@app.cell
def _(fig15):
    fig15.write_html(FIGURES / "review_effort_vs_rating.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.5 Discussion activity vs rating
    """)
    return


@app.cell
def _(df):
    fig16 = px.scatter(
        df,
        x="rating_mean",
        y="n_replies",
        opacity=0.3,
        title="Rating vs Reply Count",
        trendline="lowess",
    )
    fig16
    return (fig16,)


@app.cell
def _(fig16):
    fig16.write_html(FIGURES / "discussion_vs_rating.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.6 Strengths-to-weaknesses ratio
    """)
    return


@app.cell
def _(df):
    fig17 = px.scatter(
        df.to_pandas(),
        x="rating_mean",
        y="strengths_weaknesses_ratio",
        opacity=0.3,
        title="Rating vs Strengths/Weaknesses Word Count Ratio",
        trendline="lowess",
    )
    fig17
    return (fig17,)


@app.cell
def _(fig17):
    fig17.write_html(FIGURES / "sw_ratio_vs_rating.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.7 Area-level comparisons
    """)
    return


@app.cell
def _(df):
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

    fig18 = px.scatter(
        area_comparison,
        x="mean_rating",
        y="mean_soundness",
        size="count",
        hover_name="primary_area",
        color="mean_contribution",
        color_continuous_scale="Viridis",
        title="Area Comparison: Rating vs Soundness (size=count, color=contribution)",
    )
    fig18
    return (fig18,)


@app.cell
def _(fig18):
    fig18.write_html(FIGURES / "area_comparison.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.8 Effect sizes: Oral vs Poster
    """)
    return


@app.cell
def _(df, score_cols):
    oral = df.filter(pl.col("status") == "Oral")
    poster = df.filter(pl.col("status") == "Poster")

    for _col in score_cols + ["rating_std", "wc_review_mean", "n_replies"]:
        o = oral[_col].drop_nulls().to_numpy()
        p = poster[_col].drop_nulls().to_numpy()

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
            f"  {_col:30s}  Cohen's d={d:+.3f}  rank-biserial={rank_biserial:+.3f}  p={p_val:.2e}"
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _():
    n_figures = len(list(FIGURES.glob("*.html")))
    print(f"\nSaved {n_figures} interactive figures to {FIGURES}/")
    print("\nKey observations to investigate further:")
    print("  - Check if high-disagreement papers cluster in certain areas")
    print("  - Examine the relationship between strengths/weaknesses ratio and status")
    print("  - Look for areas with unusually high or low mean confidence")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
