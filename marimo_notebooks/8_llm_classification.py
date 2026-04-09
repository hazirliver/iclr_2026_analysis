import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import plotly.express as px
    import polars as pl
    import marimo as mo

    CLASSIFICATION_FILE = Path("classification_results_with_reasoning.parquet")
    FEATURES_FILE = Path("iclr_2026_accepted.parquet")
    OUTPUT_FILE = "iclr_2026_classified.parquet"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 8. LLM-based Paper Classification

    Semantic classification of ICLR 2026 papers into 8 thematic categories
    using **Kimi-K2.5** served via vLLM. Each paper is classified based on
    its title and abstract via a structured-output chat completion.

    Categories: AI Agents, RL, Inference Optimisation, Infrastructure,
    AI Safety/Ethics, AI for Life Sciences, Robotics, Media.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run inference script

    On the GPU VM with vLLM serving Kimi-K2.5:

    ```bash
    vllm serve moonshotai/Kimi-K2.5 \
        --tensor-parallel-size 8 \
        --max-model-len 8192 \
        --max-num-seqs 256 \
        --max-num-batched-tokens 16384 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --gpu-memory-utilization 0.95 \
        --disable-log-requests \
        --port 8000
    ```

    Then run classification:

    ```bash
    uv run python classify-script.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load classification results
    """)
    return


@app.cell
def _():
    papers = pl.read_parquet(FEATURES_FILE)
    clf = pl.read_parquet(CLASSIFICATION_FILE)
    print(
        f"Papers: {papers.shape[0]} rows, Classification results: {clf.shape[0]} rows"
    )

    df = papers.join(clf, on="openreview_id", how="left")
    print(f"Joined: {df.shape[0]} rows, {df.shape[1]} columns")
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation
    """)
    return


@app.cell
def _(df):
    n_total = df.shape[0]
    n_failed = df.filter(pl.col("llm_category") == "UNCLASSIFIED").shape[0]
    n_missing = df.filter(pl.col("llm_category").is_null()).shape[0]
    n_classified = n_total - n_failed - n_missing

    mo.output.append(
        mo.md(
            f"**Classified:** {n_classified}/{n_total} "
            f"({n_classified / n_total:.1%})\n\n"
            f"**Failed (UNCLASSIFIED):** {n_failed}\n\n"
            f"**Missing (no result):** {n_missing}"
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Category distribution
    """)
    return


@app.cell
def _(df):
    cat_counts = (
        df.filter(pl.col("llm_category") != "UNCLASSIFIED")
        .group_by("llm_category")
        .agg(
            count=pl.len(),
            mean_rating=pl.col("rating").list.mean().mean(),
        )
        .sort("count", descending=True)
        .with_columns(
            pct=(pl.col("count") / pl.col("count").sum() * 100).round(1),
        )
    )
    cat_counts
    return (cat_counts,)


@app.cell
def _(cat_counts):
    fig_cat = px.bar(
        cat_counts.to_pandas(),
        x="llm_category",
        y="count",
        text="count",
        color="llm_category",
        title="Papers per LLM category",
    )
    fig_cat.update_layout(showlegend=False, xaxis_tickangle=-30)
    fig_cat
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Confidence distribution
    """)
    return


@app.cell
def _(df):
    conf_counts = (
        df.group_by("llm_confidence").agg(count=pl.len()).sort("llm_confidence")
    )
    conf_counts
    return


@app.cell
def _(df):
    conf_by_cat = (
        df.filter(pl.col("llm_category") != "UNCLASSIFIED")
        .group_by("llm_category", "llm_confidence")
        .agg(count=pl.len())
        .sort("llm_category", "llm_confidence")
    )
    fig_conf = px.bar(
        conf_by_cat.to_pandas(),
        x="llm_category",
        y="count",
        color="llm_confidence",
        barmode="stack",
        title="Confidence breakdown per category",
        color_discrete_map={"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"},
    )
    fig_conf.update_layout(xaxis_tickangle=-30)
    fig_conf
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## LLM category vs OpenReview primary area

    Sanity-check: do LLM assignments align with the original primary-area labels?
    """)
    return


@app.cell
def _(df):
    cross = (
        df.filter(pl.col("llm_category") != "UNCLASSIFIED")
        .group_by("llm_category", "primary_area")
        .agg(count=pl.len())
        .sort("count", descending=True)
    )

    pivot = cross.pivot(
        on="llm_category", index="primary_area", values="count"
    ).fill_null(0)
    cat_cols = [c for c in pivot.columns if c != "primary_area"]
    pivot = pivot.with_columns(total=pl.sum_horizontal(*cat_cols)).sort(
        "total", descending=True
    )
    pivot
    return (cross,)


@app.cell
def _(cross):
    fig_heat = px.density_heatmap(
        cross.to_pandas(),
        x="llm_category",
        y="primary_area",
        z="count",
        histfunc="sum",
        title="LLM category vs primary area",
        color_continuous_scale="YlOrRd",
    )
    fig_heat.update_layout(
        height=700,
        xaxis_tickangle=-30,
        yaxis=dict(categoryorder="total ascending"),
    )
    fig_heat
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sample papers per category
    """)
    return


@app.cell
def _(df):
    for cat in sorted(df["llm_category"].unique().drop_nulls().to_list()):
        if cat == "UNCLASSIFIED":
            continue
        subset = df.filter(pl.col("llm_category") == cat)
        sample = subset.head(5).select("title", "llm_confidence", "llm_reasoning")
        mo.output.append(mo.md(f"### {cat} ({subset.shape[0]} papers)"))
        mo.output.append(sample)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save enriched dataset
    """)
    return


@app.cell
def _(df):
    df.write_parquet(OUTPUT_FILE)
    print(f"Saved {df.shape[0]} rows to {OUTPUT_FILE}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
