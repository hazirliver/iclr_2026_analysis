import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path

    SEED = 42
    FIGURES = Path("figures")
    FIGURES.mkdir(exist_ok=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # LLM-Based Category Analysis of ICLR 2026

    LLM classification (`llm_category`) into 6 focused categories:
    SWE Agents, RL, Inference Optimisation, Infrastructure,
    AI for Life Sciences, Robotics — plus "Other" for everything else.
    Each paper has **exactly one** category (no multi-category overlaps).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load data & separate benchmarks
    """)
    return


@app.cell
def _():
    df_all = pl.read_parquet("iclr_2026_scored.parquet")
    clf = pl.read_parquet("classification_results_with_reasoning.parquet")
    df_all = df_all.join(clf, on="openreview_id", how="left")
    print(f"Loaded {df_all.shape[0]} rows × {df_all.shape[1]} columns")

    BENCH_AREA = "datasets and benchmarks"
    BENCH_CATEGORY = "Datasets & Benchmarks"

    # Override display_category: benchmarks get their own label regardless of LLM output
    df_all = df_all.with_columns(
        pl.when(pl.col("primary_area") == BENCH_AREA)
        .then(pl.lit(BENCH_CATEGORY))
        .otherwise(pl.col("llm_category"))
        .alias("display_category")
    )

    df_bench = df_all.filter(
        (pl.col("primary_area") == BENCH_AREA)
        & (pl.col("llm_category") != "UNCLASSIFIED")
    )
    df = df_all.filter(
        (pl.col("llm_category") != "UNCLASSIFIED")
        & (pl.col("primary_area") != BENCH_AREA)
    )
    print(
        f"Benchmark papers: {df_bench.shape[0]} ({df_bench.shape[0] / df_all.shape[0]:.1%})"
    )
    print(f"Main analysis set (excl. benchmarks): {df.shape[0]} papers")
    return BENCH_CATEGORY, df, df_all, df_bench


@app.cell
def _(df_all):
    bench_overview = (
        df_all.filter(pl.col("llm_category") != "UNCLASSIFIED")
        .group_by("llm_category")
        .agg(
            total=pl.len(),
            bench=(pl.col("primary_area") == "datasets and benchmarks")
            .sum()
            .cast(pl.Int64),
        )
        .with_columns(bench_pct=(pl.col("bench") / pl.col("total") * 100).round(1))
        .sort("total", descending=True)
    )
    overall_bench_pct = round(
        df_all.filter(pl.col("primary_area") == "datasets and benchmarks").shape[0]
        / df_all.shape[0]
        * 100,
        1,
    )

    fig_bench = px.bar(
        bench_overview.to_pandas(),
        x="llm_category",
        y="bench_pct",
        text="bench",
        title="Benchmark paper share per LLM category",
        color="bench_pct",
        color_continuous_scale="Blues",
    )
    fig_bench.add_hline(
        y=overall_bench_pct,
        line_dash="dash",
        line_color="red",
        annotation_text=f"overall {overall_bench_pct}%",
    )
    fig_bench.update_layout(xaxis_tickangle=-30, yaxis_title="% benchmark papers")
    fig_bench.write_html(str(FIGURES / "llm_benchmark_share.html"))
    fig_bench
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Category metadata & scoring helpers

    We recompute four archetype scores using **category-local** z-scores so that
    a paper can be a "top overall" within its category even if it's average globally.
    """)
    return


@app.cell
def _(BENCH_CATEGORY, df, df_bench):
    SHORT_NAMES = {
        "SWE Agents": "SWE",
        "Inference Optimisation": "Inference",
        "Infrastructure": "Infra",
        "AI for Life Sciences": "Science",
        "Robotics": "Robotics",
        BENCH_CATEGORY: "Data & Bench",
        "Other": "Other",
    }

    COLOR_MAP = {
        "SWE": "#e74c3c",
        "Inference": "#2ecc71",
        "Infra": "#9b59b6",
        "Science": "#1abc9c",
        "Robotics": "#f39c12",
        "Data & Bench": "#e67e22",
        "Other": "#95a5a6",
        "Unclassified": "#d5d8dc",
    }

    CATEGORIES: dict = {}
    for cat_name in sorted(df["llm_category"].unique().drop_nulls().to_list()):
        if cat_name == "Other":
            continue
        subset = df.filter(pl.col("llm_category") == cat_name)
        short = SHORT_NAMES.get(cat_name, cat_name)
        CATEGORIES[cat_name] = {"label": cat_name, "short": short, "df": subset}

    # Benchmarks as a standalone category (based on primary_area, not LLM)
    CATEGORIES[BENCH_CATEGORY] = {
        "label": BENCH_CATEGORY,
        "short": "Data & Bench",
        "df": df_bench,
    }

    for cat_name, cat_data in CATEGORIES.items():
        short = cat_data["short"]
        n_papers = cat_data["df"].shape[0]
        print(f"  {short:>12s}: {n_papers:4d} papers")
    return CATEGORIES, COLOR_MAP, SHORT_NAMES


@app.cell
def _():
    def _z(col: pl.Expr) -> pl.Expr:
        """Standardize a column to zero mean, unit variance."""
        return (col - col.mean()) / col.std()

    def compute_local_scores(subset: pl.DataFrame) -> pl.DataFrame:
        """Recompute 4 archetype scores using category-LOCAL z-scores."""
        if subset.shape[0] < 10:
            return subset.with_columns(
                pl.col("score_top_overall").alias("local_top_overall"),
                pl.col("score_hidden_gem").alias("local_hidden_gem"),
                pl.col("score_controversial").alias("local_controversial"),
                pl.col("score_consensus").alias("local_consensus"),
            )
        return subset.with_columns(
            (
                _z(pl.col("contribution_mean")) * 0.40
                + _z(pl.col("rating_mean")) * 0.35
                + _z(pl.col("confidence_mean")) * 0.10
                + _z(pl.col("soundness_mean")) * 0.10
                - _z(pl.col("rating_std")) * 0.05
            )
            .fill_nan(0.0)
            .alias("local_top_overall"),
            (
                _z(pl.col("rating_mean")) * 0.30
                + _z(pl.col("soundness_mean")) * 0.25
                + _z(pl.col("contribution_mean")) * 0.25
                - _z(pl.col("n_replies").cast(pl.Float64)) * 0.10
                - _z(pl.col("total_review_wc").cast(pl.Float64)) * 0.10
            )
            .fill_nan(0.0)
            .alias("local_hidden_gem"),
            (
                _z(pl.col("rating_std")) * 0.25
                + _z(pl.col("rating_range").cast(pl.Float64)) * 0.25
                + _z(pl.col("wc_questions_mean")) * 0.20
                + _z(pl.col("n_replies").cast(pl.Float64)) * 0.15
                + _z(pl.col("corr_rating_confidence")).abs() * 0.15
            )
            .fill_nan(0.0)
            .alias("local_controversial"),
            (
                _z(pl.col("rating_mean")) * 0.40
                - _z(pl.col("rating_std")) * 0.30
                + _z(pl.col("confidence_mean")) * 0.30
            )
            .fill_nan(0.0)
            .alias("local_consensus"),
        )

    def diversified_top_n_cat(
        scored_df: pl.DataFrame,
        score_col: str,
        n: int,
        max_per_cluster: int = 2,
    ) -> pl.DataFrame:
        """Select top-N papers within a category with cluster diversity."""
        ranked = scored_df.sort(score_col, descending=True)
        selected: list[dict] = []
        cluster_counts: dict[int, int] = {}
        for _r in ranked.iter_rows(named=True):
            if len(selected) >= n:
                break
            cl = _r.get("cluster_ward")
            if cl is not None and cluster_counts.get(cl, 0) >= max_per_cluster:
                continue
            selected.append(_r)
            if cl is not None:
                cluster_counts[cl] = cluster_counts.get(cl, 0) + 1
        return pl.DataFrame(selected)

    def format_paper_cat(row: dict, reason: str) -> str:
        """Format a paper for display."""
        conf = row.get("llm_confidence", "?")
        return (
            f"  [{row['status']:8s}] rating={row['rating_mean']:.1f} "
            f"sound={row['soundness_mean']:.1f} contrib={row['contribution_mean']:.1f}"
            f" cluster={row['cluster_ward']} conf={conf}\n"
            f"    {row['title']}\n"
            f"    area={row['primary_area']}\n"
            f"    {row['site']}\n"
            f"    -> {reason}"
        )

    return compute_local_scores, diversified_top_n_cat, format_paper_cat


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Category overview
    """)
    return


@app.cell
def _(CATEGORIES: dict):
    rows = []
    for _cat_name, _cat in CATEGORIES.items():
        _sub = _cat["df"]
        _n = _sub.shape[0]
        if _n == 0:
            continue
        oral_pct = (_sub.filter(pl.col("status") == "Oral").shape[0] / _n) * 100
        kw_col = (
            "canonical_keywords" if "canonical_keywords" in _sub.columns else "keywords"
        )
        kw_flat = (
            _sub.select(pl.col(kw_col).explode().str.to_lowercase())
            .to_series()
            .drop_nulls()
        )
        top_kw = ", ".join(
            kw_flat.value_counts(sort=True).head(3).get_column(kw_flat.name).to_list()
        )
        conf_high_pct = (
            _sub.filter(pl.col("llm_confidence") == "high").shape[0] / _n * 100
        )
        rows.append(
            {
                "category": _cat["short"],
                "papers": _n,
                "oral_pct": round(oral_pct, 1),
                "mean_rating": round(float(_sub["rating_mean"].mean()), 2),
                "median_rating": round(float(_sub["rating_mean"].median()), 2),
                "mean_soundness": round(float(_sub["soundness_mean"].mean()), 2),
                "mean_contribution": round(float(_sub["contribution_mean"].mean()), 2),
                "confidence_high_pct": round(conf_high_pct, 1),
                "top_keywords": top_kw,
            }
        )
    overview_df = pl.DataFrame(rows)
    overview_df
    return (overview_df,)


@app.cell
def _(overview_df):
    fig_overview = px.bar(
        overview_df.to_pandas(),
        y="category",
        x="papers",
        color="mean_rating",
        color_continuous_scale="RdYlGn",
        orientation="h",
        text="papers",
        hover_data=[
            "oral_pct",
            "mean_soundness",
            "mean_contribution",
            "confidence_high_pct",
            "top_keywords",
        ],
        title="Papers per LLM Category (colored by mean rating)",
    )
    fig_overview.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
    fig_overview.write_html(str(FIGURES / "llm_category_overview.html"))
    fig_overview
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-category top 10 papers

    For each category we select 10 papers via a budget-based approach using
    **category-local** z-scores: 3 top-overall + 2 hidden gems (poster only) +
    2 controversial + 3 consensus standouts, deduplicated and diversified across
    Ward clusters.
    """)
    return


@app.cell
def _(
    CATEGORIES: dict,
    compute_local_scores,
    diversified_top_n_cat,
    format_paper_cat,
):
    all_category_picks: dict[str, list[dict]] = {}

    for _cat_name, _cat in CATEGORIES.items():
        _sub = _cat["df"]
        _n = _sub.shape[0]
        print(f"\n{'=' * 70}")
        print(f"Category: {_cat['label']} ({_n} papers)")
        print(f"{'=' * 70}")

        if _n == 0:
            all_category_picks[_cat_name] = []
            continue

        scored = compute_local_scores(_sub)
        seen: set[str] = set()
        picks: list[dict] = []

        budget = [
            ("local_top_overall", 10, None, "Top overall (local)"),
            ("local_hidden_gem", 0, "Poster", "Hidden gem (local)"),
            ("local_controversial", 0, None, "Controversial (local)"),
            ("local_consensus", 0, None, "Consensus standout (local)"),
        ]
        for sc, want, sf, label in budget:
            pool = scored if not sf else scored.filter(pl.col("status") == sf)
            top = diversified_top_n_cat(pool, sc, want + 5)
            added = 0
            for _r in top.iter_rows(named=True):
                if len(picks) >= 10 or added >= want:
                    break
                if _r["openreview_id"] in seen:
                    continue
                seen.add(_r["openreview_id"])
                rc = dict(_r)
                rc["label"] = label
                rc["_score"] = _r[sc]
                picks.append(rc)
                added += 1

        if len(picks) < 10:
            bf = diversified_top_n_cat(scored, "local_top_overall", 15)
            for _r in bf.iter_rows(named=True):
                if len(picks) >= 10:
                    break
                if _r["openreview_id"] in seen:
                    continue
                seen.add(_r["openreview_id"])
                rc = dict(_r)
                rc["label"] = "Top overall (backfill)"
                rc["_score"] = _r["local_top_overall"]
                picks.append(rc)

        all_category_picks[_cat_name] = picks
        for _i, _p in enumerate(picks, 1):
            _lbl = _p["label"]
            scr = _p["_score"]
            print(f"\n{_i}. {format_paper_cat(_p, f'{_lbl} = {scr:.3f}')}")
    return (all_category_picks,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.filter(pl.col("site") == "https://openreview.net/forum?id=DM0Y0oL33T").select(
        ["llm_confidence", "llm_reasoning", "llm_category", "site"]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Category × Archetype heatmap

    Each cell shows what percentage of a category's papers fall into the global
    top-20% for that archetype. The baseline is 20% — values above indicate
    over-representation.
    """)
    return


@app.cell
def _(CATEGORIES: dict, df):
    arch_cols = [
        "score_top_overall",
        "score_hidden_gem",
        "score_controversial",
        "score_high_engagement",
        "score_semantic_novel",
        "score_bridge",
        "score_area_leader",
        "score_consensus",
    ]
    arch_labels = [
        _c.replace("score_", "").replace("_", " ").title() for _c in arch_cols
    ]
    thresholds = {_c: df[_c].quantile(0.80) for _c in arch_cols}

    hm_data: list[list[float]] = []
    cat_labels_hm = []
    for _cat_name in sorted(CATEGORIES.keys()):
        _cat = CATEGORIES[_cat_name]
        _sub = _cat["df"]
        _n = _sub.shape[0]
        if _n == 0:
            hm_data.append([0.0] * len(arch_cols))
        else:
            row_pcts = []
            for _c in arch_cols:
                pct = (_sub.filter(pl.col(_c) > thresholds[_c]).shape[0] / _n) * 100
                row_pcts.append(round(pct, 1))
            hm_data.append(row_pcts)
        cat_labels_hm.append(f"{_cat['short']} ({_sub.shape[0]})")

    hm_array = np.array(hm_data)
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=hm_array,
            x=arch_labels,
            y=cat_labels_hm,
            colorscale="YlOrRd",
            text=hm_array.astype(str),
            texttemplate="%{text}%",
            textfont={"size": 11},
            colorbar={"title": "% in top 20%"},
        )
    )
    fig_heatmap.update_layout(
        title="Category x Archetype: % of papers in global top-20%<br>(20% = baseline; above = over-indexed)",
        height=500,
        width=900,
        xaxis_title="Archetype",
        yaxis_title="Category",
        yaxis={"autorange": "reversed"},
    )
    fig_heatmap.write_html(str(FIGURES / "llm_category_archetype_heatmap.html"))
    fig_heatmap
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## LLM classification confidence

    Since each paper has exactly one LLM category (no multi-category overlap),
    we instead analyze the model's classification confidence and cross-check
    assignments against OpenReview's primary area labels.
    """)
    return


@app.cell
def _(CATEGORIES: dict, COLOR_MAP):
    conf_rows = []
    for _cat_name, _cat in CATEGORIES.items():
        _sub = _cat["df"]
        for level in ["high", "medium", "low"]:
            cnt = _sub.filter(pl.col("llm_confidence") == level).shape[0]
            conf_rows.append(
                {"category": _cat["short"], "confidence": level, "count": cnt}
            )
    conf_by_cat = pl.DataFrame(conf_rows)

    _cat_order = sorted(COLOR_MAP.keys())
    fig_conf = px.bar(
        conf_by_cat.to_pandas(),
        x="category",
        y="count",
        color="confidence",
        barmode="stack",
        title="LLM confidence breakdown per category",
        color_discrete_map={"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"},
        category_orders={"category": _cat_order},
    )
    fig_conf.update_layout(xaxis_tickangle=-30)
    fig_conf.write_html(str(FIGURES / "llm_confidence_by_category.html"))
    fig_conf
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Low-confidence classifications

    Papers where the LLM was uncertain — these are borderline cases worth inspecting.
    """)
    return


@app.cell
def _(df):
    low_conf = (
        df.filter(
            (pl.col("llm_confidence") == "low")
            & (pl.col("llm_category") != "UNCLASSIFIED")
        )
        .select("title", "llm_category", "llm_reasoning", "rating_mean", "primary_area")
        .sort("rating_mean", descending=True)
    )
    print(f"Low-confidence papers (excluding UNCLASSIFIED): {low_conf.shape[0]}")
    low_conf.head(15)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### LLM category vs OpenReview primary area
    """)
    return


@app.cell
def _(SHORT_NAMES, df):
    cross = (
        df.filter(
            (pl.col("llm_category") != "UNCLASSIFIED")
            & pl.col("llm_category").is_not_null()
        )
        .with_columns(pl.col("llm_category").replace(SHORT_NAMES).alias("cat_short"))
        .group_by("cat_short", "primary_area")
        .agg(count=pl.len())
        .with_columns(
            pct=pl.col("count") / pl.col("count").sum().over("cat_short") * 100
        )
    )

    fig_cross = px.density_heatmap(
        cross.to_pandas(),
        x="cat_short",
        y="primary_area",
        z="pct",
        histfunc="sum",
        title="Primary area distribution within each LLM category",
        color_continuous_scale="YlOrRd",
        text_auto=".1f",
    )

    fig_cross.update_layout(
        height=700,
        xaxis_tickangle=-30,
        xaxis_title="LLM Category",
        yaxis_title="Primary Area",
        coloraxis_colorbar=dict(title="% within category"),
        yaxis=dict(categoryorder="total ascending"),
    )

    fig_cross.update_traces(
        hovertemplate=(
            "LLM category: %{x}<br>"
            "Primary area: %{y}<br>"
            "Share within category: %{z:.2f}%<extra></extra>"
        )
    )

    fig_cross.write_html(str(FIGURES / "llm_vs_primary_area.html"))
    fig_cross
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UMAP: Papers colored by LLM category

    Each paper gets its single LLM-assigned category. UNCLASSIFIED papers
    (inference failures) are de-emphasized in gray.
    """)
    return


@app.cell
def _(COLOR_MAP, SHORT_NAMES, df_all):
    umap_df = df_all.select(
        "umap_x",
        "umap_y",
        "title",
        "rating_mean",
        "status",
        "display_category",
        "llm_confidence",
    ).with_columns(
        pl.when(pl.col("display_category") == "UNCLASSIFIED")
        .then(pl.lit("Unclassified"))
        .otherwise(pl.col("display_category").replace(SHORT_NAMES))
        .alias("cat_label")
    )

    _cat_order = [
        "SWE",
        "Inference",
        "Infra",
        "Science",
        "Robotics",
        "Data & Bench",
        "Other",
        "Unclassified",
    ]

    fig_umap = px.scatter(
        umap_df.to_pandas(),
        x="umap_x",
        y="umap_y",
        color="cat_label",
        color_discrete_map=COLOR_MAP,
        category_orders={"cat_label": _cat_order},
        hover_name="title",
        hover_data=["rating_mean", "status", "llm_confidence"],
        opacity=0.6,
        title="UMAP: Papers Colored by LLM Category",
    )
    for trace in fig_umap.data:
        if trace.name in ("Other", "Unclassified"):
            trace.marker.opacity = 1
            trace.marker.size = 4
            trace.marker.color = COLOR_MAP.get(trace.name, "#a9adb7")
    fig_umap.update_layout(height=700, width=1100)
    fig_umap.write_html(str(FIGURES / "llm_umap_by_category.html"))
    fig_umap
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bridge papers by LLM category

    Bridge papers sit semantically equidistant between Ward clusters (low
    `bridge_ratio`). Even though each has a single LLM category, their embedding
    neighborhood spans multiple research communities.
    """)
    return


@app.cell
def _(CATEGORIES: dict, df, format_paper_cat):
    bridge_threshold = df.filter(pl.col("bridge_ratio").is_not_null())[
        "bridge_ratio"
    ].quantile(0.10)
    bridge_papers = df.filter(
        pl.col("bridge_ratio").is_not_null()
        & (pl.col("bridge_ratio") <= bridge_threshold)
        & (pl.col("llm_category") != "UNCLASSIFIED")
    ).sort("bridge_ratio")

    bridge_by_cat = (
        bridge_papers.group_by("display_category")
        .agg(bridge_count=pl.len())
        .join(
            pl.DataFrame(
                [
                    {"display_category": k, "total": v["df"].shape[0]}
                    for k, v in CATEGORIES.items()
                ]
            ),
            on="display_category",
        )
        .with_columns(
            bridge_pct=(pl.col("bridge_count") / pl.col("total") * 100).round(1)
        )
        .sort("bridge_pct", descending=True)
    )
    print("Bridge papers (top 10% bridge_ratio) per category:")
    print(bridge_by_cat)

    print("\nTop 15 bridge papers (lowest bridge_ratio = most bridging):")
    for _i, _row in enumerate(bridge_papers.head(15).iter_rows(named=True), 1):
        br = _row["bridge_ratio"]
        cat = _row["display_category"]
        print(f"\n{_i}. {format_paper_cat(_row, f'bridge_ratio={br:.3f}, cat={cat}')}")
    return


@app.cell
def _(COLOR_MAP, SHORT_NAMES, df):
    bridge_plot = df.filter(
        pl.col("bridge_ratio").is_not_null()
        & (pl.col("llm_category") != "UNCLASSIFIED")
    ).with_columns(pl.col("display_category").replace(SHORT_NAMES).alias("cat_short"))

    fig_bridge = px.scatter(
        bridge_plot.to_pandas(),
        x="bridge_ratio",
        y="rating_mean",
        color="cat_short",
        color_discrete_map=COLOR_MAP,
        hover_name="title",
        hover_data=["status", "primary_area"],
        opacity=0.4,
        title="Semantic Bridge Ratio vs Rating by LLM Category",
    )
    fig_bridge.update_layout(
        height=500,
        width=900,
        xaxis_title="Bridge Ratio (lower = more bridging)",
        yaxis_title="Mean Rating",
    )
    fig_bridge.write_html(str(FIGURES / "llm_bridge_by_category.html"))
    fig_bridge
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Export per-category CSVs

    Export all papers per category, sorted by `local_top_overall` descending,
    into `exports/<category>.csv`.
    """)
    return


@app.cell
def _(CATEGORIES: dict, compute_local_scores):
    EXPORT_COLS = [
        "site",
        "title",
        "abstract",
        "status",
        "primary_area",
        "keywords",
        "rating",
        "presentation",
        "confidence",
        "soundness",
        "contribution",
        "pdf_link",
        "score_top_overall",
        "llm_category",
        "llm_confidence",
        "llm_reasoning",
    ]

    def export_category_csvs(
        categories: dict,
        out_dir: str = "exports",
    ) -> Path:
        """Export one CSV per category sorted by local_top_overall."""
        out = Path(out_dir)
        out.mkdir(exist_ok=True)

        for cat_name, cat_data in categories.items():
            subset = cat_data["df"]
            if subset.shape[0] == 0:
                continue

            scored = compute_local_scores(subset)

            export_df = (
                scored.sort("local_top_overall", descending=True)
                .with_columns(
                    pl.col("keywords").list.join(", ").alias("keywords"),
                    pl.col("rating").list.eval(pl.element().cast(pl.String)).list.join(", ").alias("rating"),
                    pl.col("presentation").list.eval(pl.element().cast(pl.String)).list.join(", ").alias("presentation"),
                    pl.col("confidence").list.eval(pl.element().cast(pl.String)).list.join(", ").alias("confidence"),
                    pl.col("soundness").list.eval(pl.element().cast(pl.String)).list.join(", ").alias("soundness"),
                    pl.col("contribution").list.eval(pl.element().cast(pl.String)).list.join(", ").alias("contribution"),
                )
                .select(EXPORT_COLS + ["local_top_overall"])
            )

            slug = cat_name.lower().replace(" ", "_")
            export_df.write_csv(out / f"{slug}.csv")
            print(f"  {cat_name}: {export_df.shape[0]} rows -> {out / f'{slug}.csv'}")

        print(f"\nExported {len(categories)} category CSVs to {out}/")
        return out

    export_dir = export_category_csvs(CATEGORIES)
    return (export_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(CATEGORIES: dict, all_category_picks: dict[str, list[dict]], df, df_all):
    n_classified = df.filter(
        (pl.col("llm_category") != "UNCLASSIFIED")
        & pl.col("llm_category").is_not_null()
    ).shape[0]
    n_unclassified = df_all.filter(pl.col("llm_category") == "UNCLASSIFIED").shape[0]

    total_picks = sum(len(_v) for _v in all_category_picks.values())
    unique_ids = set()
    for _picks in all_category_picks.values():
        for _p in _picks:
            unique_ids.add(_p["openreview_id"])

    print(f"Categories: {len(CATEGORIES)}")
    print(f"Classified papers (excl. benchmarks): {n_classified}")
    print(f"UNCLASSIFIED (inference failures): {n_unclassified}")
    print()
    for _cat_name, _cat in CATEGORIES.items():
        print(f"  {_cat['short']:>10s}: {_cat['df'].shape[0]:4d}")
    print()
    print(f"Total per-category picks: {total_picks}")
    print(f"Unique picks across all categories: {len(unique_ids)}")
    print(f"Figures saved to: {FIGURES}/")
    return


if __name__ == "__main__":
    app.run()
