import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl

    SEED = 42
    np.random.seed(SEED)


@app.cell
def _():
    df = pl.read_parquet("iclr_2026_clustered.parquet")
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return (df,)


@app.cell
def _(df):
    has_embeddings = "mean_knn_distance" in df.columns
    has_bridge = "bridge_ratio" in df.columns
    has_clusters = "cluster_ward" in df.columns
    print(f"Embeddings: {has_embeddings}, Bridge scores: {has_bridge}, Clusters: {has_clusters}")
    return (has_bridge,)


@app.cell
def _():
    def z_score(col: pl.Expr) -> pl.Expr:
        """Standardize a column to zero mean, unit variance."""
        return (col - col.mean()) / col.std()


    def area_z(col_name: str) -> pl.Expr:
        """Area-normalized z-score."""
        return (
            (pl.col(col_name) - pl.col(col_name).mean().over("primary_area")) / pl.col(col_name).std().over("primary_area")
        ).fill_nan(0.0)


    def print_top_papers(scored_df: pl.DataFrame, score_col: str, label: str, n: int = 20):
        """Print top-N papers for an archetype."""
        top = scored_df.sort(score_col, descending=True).head(n)
        print(f"\n{'=' * 60}")
        print(f"TOP {n} — {label}")
        print(f"{'=' * 60}")
        for i, row in enumerate(top.iter_rows(named=True), 1):
            print(
                f"  {i:2d}. [{row['status']:8s}] rating={row['rating_mean']:.1f} "
                f"score={row[score_col]:.3f}  {row['title'][:70]}"
            )
            print(f"      area={row['primary_area'][:50]}  {row['site']}")

    return area_z, print_top_papers, z_score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Top overall
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    High rating + soundness + contribution, low disagreement, decent confidence
    """)
    return


@app.cell
def _(df, print_top_papers, z_score):
    df1 = df.with_columns(
        score_top_overall=(
            z_score(pl.col("rating_mean")) * 0.35
            + z_score(pl.col("soundness_mean")) * 0.25
            + z_score(pl.col("contribution_mean")) * 0.25
            - z_score(pl.col("rating_std")) * 0.10
            + z_score(pl.col("confidence_mean")) * 0.05
        )
    )
    print_top_papers(df1, "score_top_overall", "TOP OVERALL")
    return (df1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Hidden Gems
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Poster papers with strong area-normalized scores but low engagement
    """)
    return


@app.cell
def _(area_z, df1, print_top_papers, z_score):
    df2 = df1.with_columns(
        score_hidden_gem=(
            area_z("rating_mean") * 0.30
            + area_z("soundness_mean") * 0.25
            + area_z("contribution_mean") * 0.25
            - z_score(pl.col("n_replies")) * 0.10  # low engagement
            - z_score(pl.col("total_review_wc")) * 0.10  # low attention
        )
    )
    # Only consider Poster papers
    hidden_gem_df = df2.filter(pl.col("status").str.contains("Poster"))
    print_top_papers(hidden_gem_df, "score_hidden_gem", "HIDDEN GEMS (Poster only)", 15)
    return (df2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Controversial
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    High disagreement, many questions, active discussion
    """)
    return


@app.cell
def _(df2, print_top_papers, z_score):
    df3 = df2.with_columns(
        score_controversial=(
            z_score(pl.col("rating_std")) * 0.25
            + z_score(pl.col("rating_range").cast(pl.Float64)) * 0.25
            + z_score(pl.col("wc_questions_mean")) * 0.20
            + z_score(pl.col("n_replies").cast(pl.Float64)) * 0.15
            + z_score(pl.col("corr_rating_confidence").abs()) * 0.15
        )
    )
    print_top_papers(df3, "score_controversial", "CONTROVERSIAL", 15)
    return (df3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. High Engagement
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Lots of discussion, long reviews, many questions
    """)
    return


@app.cell
def _(area_z, df3, print_top_papers):
    df4 = df3.with_columns(
        score_high_engagement=(
            area_z("n_replies") * 0.30
            + area_z("wc_questions_mean") * 0.25
            + area_z("total_review_wc") * 0.25
            + area_z("wc_review_mean") * 0.20
        )
    )
    print_top_papers(df4, "score_high_engagement", "HIGH ENGAGEMENT", 15)
    return (df4,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Semantically novel
    """)
    return


@app.cell
def _(df4, has_bridge, print_top_papers, z_score):
    df5 = df4.with_columns(
        score_semantic_novel=(
            z_score(pl.col("mean_knn_distance")) * 0.50 + z_score(pl.col("dist_to_nearest_centroid")) * 0.50
            if has_bridge
            else z_score(pl.col("mean_knn_distance"))
        )
    )
    print_top_papers(df5, "score_semantic_novel", "SEMANTICALLY NOVEL", 15)
    return (df5,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Bridge papers
    """)
    return


@app.cell
def _(df5, print_top_papers, z_score):
    df6 = df5.with_columns(score_bridge=(-z_score(pl.col("bridge_ratio"))))
    print_top_papers(df6, "score_bridge", "BRIDGE PAPERS", 15)
    return (df6,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7. Cluster exemplars
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Nearest to centroid among top-rated papers in each cluster
    """)
    return


@app.cell
def _(df, df6, has_bridge):
    cluster_col = "cluster_ward"
    clusters = df6.filter(pl.col(cluster_col) != -1)[cluster_col].unique().sort().to_list()

    exemplars = []
    for cid in clusters:
        cluster_df = df.filter(pl.col(cluster_col) == cid)
        # Top 25% by rating within cluster
        threshold = cluster_df["rating_mean"].quantile(0.75)
        top_rated = cluster_df.filter(pl.col("rating_mean") >= threshold)

        if has_bridge and "dist_to_nearest_centroid" in top_rated.columns:
            # Pick the one closest to centroid among top-rated
            best = top_rated.sort("dist_to_nearest_centroid").head(3)
        else:
            best = top_rated.sort("rating_mean", descending=True).head(3)

        exemplars.append(best.with_columns(pl.lit(cid).alias("exemplar_cluster")))

    if exemplars:
        exemplar_df = pl.concat(exemplars)
        print(f"\n{'=' * 60}")
        print(f"CLUSTER EXEMPLARS ({len(clusters)} clusters, up to 3 per cluster)")
        print(f"{'=' * 60}")
        for row in exemplar_df.iter_rows(named=True):
            print(f"  Cluster {row['exemplar_cluster']:3d}: rating={row['rating_mean']:.1f}  {row['title'][:65]}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 8. Area Leaders
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Highest area-normalized composite per primary area
    """)
    return


@app.cell
def _(area_z, df6):
    df7 = df6.with_columns(
        score_area_leader=(
            area_z("rating_mean") * 0.40 + area_z("soundness_mean") * 0.30 + area_z("contribution_mean") * 0.30
        )
    )

    areas = df7["primary_area"].unique().sort().to_list()
    print(f"\n{'=' * 60}")
    print(f"AREA LEADERS (top 3 per area, {len(areas)} areas)")
    print(f"{'=' * 60}")
    for area in areas:
        area_df = df7.filter(pl.col("primary_area") == area)
        top = area_df.sort("score_area_leader", descending=True).head(3)
        print(f"\n  [{area}]")
        for _row in top.iter_rows(named=True):
            print(f"    rating={_row['rating_mean']:.1f} score={_row['score_area_leader']:.2f}  {_row['title'][:65]}")
    return (df7,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 9. Consensus standout
    """)
    return


@app.cell
def _(df7, print_top_papers, z_score):
    df8 = df7.with_columns(
        score_consensus=(
            z_score(pl.col("rating_mean")) * 0.40
            - z_score(pl.col("rating_std")) * 0.30
            + z_score(pl.col("confidence_mean")) * 0.30
        )
    )
    print_top_papers(df8, "score_consensus", "CONSENSUS STANDOUTS", 15)
    return (df8,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Save
    """)
    return


@app.cell
def _(df8):
    score_cols = [c for c in df8.columns if c.startswith("score_")]
    print(f"\nArchetype scores computed: {score_cols}")

    df8.write_parquet("iclr_2026_scored.parquet")
    print(f"Saved to iclr_2026_scored.parquet ({df8.shape[0]} rows × {df8.shape[1]} columns)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
