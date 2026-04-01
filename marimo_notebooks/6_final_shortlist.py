import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import polars as pl


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    df = pl.read_parquet("iclr_2026_scored.parquet")
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    def format_paper(row: dict, reason: str) -> str:
        """Format a paper for display."""
        return (
            f"  [{row['status']:8s}] rating={row['rating_mean']:.1f} "
            f"sound={row['soundness_mean']:.1f} contrib={row['contribution_mean']:.1f}"
            f" cluster={row['cluster_ward']}\n"
            f"    {row['title'][:80]}\n"
            f"    area={row['primary_area'][:50]}\n"
            f"    {row['site']}\n"
            f"    → {reason}"
        )

    def diversified_top_n(
        scored_df: pl.DataFrame,
        score_col: str,
        n: int,
        max_per_area: int = 3,
        max_per_cluster: int | None = 3,
    ) -> pl.DataFrame:
        """Select top-N papers with diversity constraints."""
        ranked = scored_df.sort(score_col, descending=True)
        selected = []
        area_counts: dict[str, int] = {}
        cluster_counts: dict[int, int] = {}

        for row in ranked.iter_rows(named=True):
            if len(selected) >= n:
                break

            area = row["primary_area"]
            cluster = row.get("cluster_ward")

            if area_counts.get(area, 0) >= max_per_area:
                continue
            if (
                max_per_cluster
                and cluster is not None
                and cluster_counts.get(cluster, 0) >= max_per_cluster
            ):
                continue

            selected.append(row)
            area_counts[area] = area_counts.get(area, 0) + 1
            if cluster is not None:
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        return pl.DataFrame(selected)

    def print_list(title: str, papers: pl.DataFrame, score_col: str):
        """Print a shortlist."""
        print(f"\n{'=' * 70}")
        print(f"{title} ({papers.shape[0]} papers)")
        print(f"{'=' * 70}")
        for i, row in enumerate(papers.iter_rows(named=True), 1):
            print(f"\n{i}. {format_paper(row, f'{score_col}={row[score_col]:.3f}')}")

    return (
        df,
        diversified_top_n,
        format_paper,
        print_list,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Shortlists
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Top 20 overall
    """)
    return


@app.cell
def _(df, diversified_top_n, print_list):
    top_overall = diversified_top_n(df, "score_top_overall", 20)
    print_list("TOP 20 OVERALL", top_overall, "score_top_overall")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Top 15 hidden gems (Poster only)
    """)
    return


@app.cell
def _(df, diversified_top_n, print_list):
    poster_df = df.filter(pl.col("status").str.contains("Poster"))
    hidden_gems = diversified_top_n(poster_df, "score_hidden_gem", 15)
    print_list("TOP 15 HIDDEN GEMS", hidden_gems, "score_hidden_gem")
    return (poster_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Top 15 controversial
    """)
    return


@app.cell
def _(df, diversified_top_n, print_list):
    controversial = diversified_top_n(df, "score_controversial", 15, max_per_area=4)
    print_list("TOP 15 CONTROVERSIAL", controversial, "score_controversial")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Top 15 semantically novel
    """)
    return


@app.cell
def _(df, diversified_top_n, print_list):
    semantic = diversified_top_n(df, "score_semantic_novel", 15)
    print_list("TOP 15 SEMANTICALLY NOVEL", semantic, "score_semantic_novel")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Top 3-5 per major cluster
    """)
    return


@app.cell
def _(df):
    cluster_col = "cluster_ward"
    clusters = df.group_by(cluster_col).len().sort("len", descending=True)
    major_clusters = clusters.filter(pl.col("len") >= 50)[cluster_col].to_list()

    print(f"\n{'=' * 70}")
    print(
        f"TOP PAPERS PER MAJOR CLUSTER ({len(major_clusters)} clusters with 50+ papers)"
    )
    print(f"{'=' * 70}")
    for cid in major_clusters:
        cluster_df = df.filter(pl.col(cluster_col) == cid)
        top = cluster_df.sort("score_top_overall", descending=True).head(5)
        print(f"\n--- Cluster {cid} (n={cluster_df.shape[0]}) ---")
        for row in top.iter_rows(named=True):
            print(f"  rating={row['rating_mean']:.1f}  {row['title'][:70]}")
            print(f"    {row['site']}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Top 3-5 per primary area
    """)
    return


@app.cell
def _(df):
    areas = df.group_by("primary_area").len().sort("len", descending=True)

    print(f"\n{'=' * 70}")
    print(f"TOP PAPERS PER PRIMARY AREA ({areas.shape[0]} areas)")
    print(f"{'=' * 70}")
    for _row in areas.iter_rows(named=True):
        area = _row["primary_area"]
        area_df = df.filter(pl.col("primary_area") == area)
        n_show = 5 if _row["len"] >= 50 else 3
        _top = area_df.sort("score_area_leader", descending=True).head(n_show)
        print(f"\n--- {area} (n={_row['len']}) ---")
        for paper in _top.iter_rows(named=True):
            print(f"  rating={paper['rating_mean']:.1f}  {paper['title'][:70]}")
            print(f"    {paper['site']}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Combined unique list
    """)
    return


@app.cell
def _(df, format_paper, poster_df):
    def _():
        print(f"\n{'=' * 70}")
        print("FINAL DIVERSIFIED 'START HERE' LIST")
        print(f"{'=' * 70}")

        candidates: list[
            tuple[str, str, float]
        ] = []  # (openreview_id, reason, priority)

        for row in (
            df.sort("score_top_overall", descending=True).head(30).iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "top overall", row["score_top_overall"])
            )

        for row in (
            poster_df.sort("score_hidden_gem", descending=True)
            .head(20)
            .iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "hidden gem", row["score_hidden_gem"])
            )

        for row in (
            df.sort("score_controversial", descending=True)
            .head(15)
            .iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "controversial", row["score_controversial"])
            )

        for row in (
            df.sort("score_consensus", descending=True).head(15).iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "consensus standout", row["score_consensus"])
            )

        for row in (
            df.sort("score_high_engagement", descending=True)
            .head(15)
            .iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "high engagement", row["score_high_engagement"])
            )

        for row in (
            df.sort("score_semantic_novel", descending=True)
            .head(15)
            .iter_rows(named=True)
        ):
            candidates.append(
                (
                    row["openreview_id"],
                    "semantically novel",
                    row["score_semantic_novel"],
                )
            )

        for row in (
            df.sort("score_bridge", descending=True).head(15).iter_rows(named=True)
        ):
            candidates.append(
                (row["openreview_id"], "bridge paper", row["score_bridge"])
            )

        # Deduplicate: keep first occurrence (highest priority archetype)
        seen = set()
        unique_candidates = []
        for oid, reason, score in candidates:
            if oid not in seen:
                seen.add(oid)
                unique_candidates.append((oid, reason))

        # Apply diversity constraints
        final_list = []
        area_counts: dict[str, int] = {}
        cluster_counts: dict[int, int] = {}
        status_counts: dict[str, int] = {}
        n_clusters = df["cluster_ward"].n_unique()

        MAX_PER_AREA = 5
        MAX_PER_CLUSTER = max(10, 30 // max(n_clusters, 1))
        MAX_PER_STATUS = 25
        TARGET = 30

        for oid, reason in unique_candidates:
            if len(final_list) >= TARGET:
                break

            paper = df.filter(pl.col("openreview_id") == oid).row(0, named=True)
            area = paper["primary_area"]
            status = paper["status"]
            cluster = paper["cluster_ward"]

            if area_counts.get(area, 0) >= MAX_PER_AREA:
                continue
            if status_counts.get(status, 0) >= MAX_PER_STATUS:
                continue
            if cluster_counts.get(cluster, 0) >= MAX_PER_CLUSTER:
                continue

            final_list.append((paper, reason))
            area_counts[area] = area_counts.get(area, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        print(
            f"\nSelected {len(final_list)} papers from {len(unique_candidates)} candidates\n"
        )
        for i, (paper, reason) in enumerate(final_list, 1):
            print(f"{i:2d}. {format_paper(paper, reason)}\n")

        return final_list

    final_list = _()
    return (final_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary stats
    """)
    return


@app.cell
def _(final_list):
    final_areas = [p["primary_area"] for p, _ in final_list]
    final_statuses = [p["status"] for p, _ in final_list]
    final_clusters = [p["cluster_ward"] for p, _ in final_list]
    print("\nDiversity check:")
    print(f"  Unique areas: {len(set(final_areas))}")
    print(f"  Unique statuses: {len(set(final_statuses))}")
    print(f"  Unique clusters: {len(set(final_clusters))}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
