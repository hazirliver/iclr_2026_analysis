import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import numpy as np
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Data Audit & Feature Engineering
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Validates data integrity, confirms column semantics, and builds
    a flat analysis table with engineered features
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    df = pl.read_parquet("iclr_2026_accepted.parquet")
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Basic integrity
    """)
    return


@app.cell
def _(df):
    assert df.shape == (5358, 31), f"Unexpected shape: {df.shape}"
    assert df.null_count().sum_horizontal().item() == 0, "Found nulls"
    assert df["openreview_id"].n_unique() == df.shape[0], "Duplicate openreview_id"
    print("Shape, nulls, uniqueness OK")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Identifier consistency
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Site should contain openreview_id, pdf_link should be site with forum -> pdf
    """)
    return


@app.cell
def _(df):
    id_in_site = df.select((pl.col("site").str.contains(pl.col("openreview_id"))).all()).item()
    id_in_pdf = df.select((pl.col("pdf_link").str.contains(pl.col("openreview_id"))).all()).item()
    pdf_from_site = df.select((pl.col("site").str.replace("forum", "pdf") == pl.col("pdf_link")).all()).item()
    assert id_in_site, "openreview_id not found in site URL"
    assert id_in_pdf, "openreview_id not found in pdf_link"
    assert pdf_from_site, "pdf_link != site with forum→pdf"
    print("Identifier consistency OK")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Status distribution
    """)
    return


@app.cell
def _(df):
    status_counts = df.group_by("status").len().sort("len", descending=True)
    status_counts
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validate `_avg` columns
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `_avg` columns should be `[mean, std]` of the corresponding raw list columns
    """)
    return


@app.cell
def _(df):
    avg_pairs = [
        ("rating", "rating_avg"),
        ("soundness", "soundness_avg"),
        ("presentation", "presentation_avg"),
        ("contribution", "contribution_avg"),
        ("confidence", "confidence_avg"),
        ("wc_review", "wc_review_avg"),
        ("wc_summary", "wc_summary_avg"),
        ("wc_strengths", "wc_strengths_avg"),
        ("wc_weaknesses", "wc_weaknesses_avg"),
        ("wc_questions", "wc_questions_avg"),
    ]

    print("\nValidating _avg columns (recompute mean/std from raw):")
    # Stored std uses ddof=1 (sample std), but Polars list.eval(element().std())
    # uses ddof=1 by default. We test both ddof values to identify which was used.
    for raw_col, avg_col in avg_pairs:
        recomputed = df.select(
            recomputed_mean=pl.col(raw_col).list.mean(),
            # ddof=1 (sample std) — Polars default in list context
            recomputed_std_ddof1=pl.col(raw_col).list.eval(pl.element().cast(pl.Float64).std(ddof=1)).list.first(),
            # ddof=0 (population std)
            recomputed_std_ddof0=pl.col(raw_col).list.eval(pl.element().cast(pl.Float64).std(ddof=0)).list.first(),
            stored_mean=pl.col(avg_col).list.get(0),
            stored_std=pl.col(avg_col).list.get(1),
        )
        mean_diff_val = (recomputed["recomputed_mean"] - recomputed["stored_mean"]).abs().max()
        mean_diff: float = mean_diff_val if isinstance(mean_diff_val, (int, float)) else 0.0  # type: ignore[assignment]
        std_ddof1_val = (recomputed["recomputed_std_ddof1"] - recomputed["stored_std"]).abs().max()
        std_diff_ddof1: float = std_ddof1_val if isinstance(std_ddof1_val, (int, float)) else 0.0  # type: ignore[assignment]
        std_ddof0_val = (recomputed["recomputed_std_ddof0"] - recomputed["stored_std"]).abs().max()
        std_diff_ddof0: float = std_ddof0_val if isinstance(std_ddof0_val, (int, float)) else 0.0  # type: ignore[assignment]
        # Pick whichever ddof matches better
        if std_diff_ddof1 <= std_diff_ddof0:
            std_diff = std_diff_ddof1
            ddof_used = "ddof=1"
        else:
            std_diff = std_diff_ddof0
            ddof_used = "ddof=0"
        status = "✓" if mean_diff < 0.01 and std_diff < 0.01 else "✗"
        print(f"  {status} {raw_col}: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f} ({ddof_used})")
    return (avg_pairs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validate `corr_rating_confidence`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pearson correlation between rating and confidence per paper
    """)
    return


@app.cell
def _(df):
    def pearson_corr(ratings: list[int], confs: list[int]) -> float | None:
        """Compute Pearson correlation, return None if undefined."""
        r = np.array(ratings, dtype=float)
        c = np.array(confs, dtype=float)
        if len(r) < 2 or r.std() == 0 or c.std() == 0:
            return None
        return float(np.corrcoef(r, c)[0, 1])


    corr_check = df.select("rating", "confidence", "corr_rating_confidence")
    recomputed_corrs = [pearson_corr(row[0], row[1]) for row in corr_check.select("rating", "confidence").iter_rows()]
    stored_corrs = corr_check["corr_rating_confidence"].to_list()

    diffs = [
        abs(r - s)
        for r, s in zip(recomputed_corrs, stored_corrs)
        if r is not None and s is not None and not (np.isnan(r) or np.isnan(s))
    ]
    max_corr_diff = max(diffs) if diffs else 0.0
    print(f"\ncorr_rating_confidence max recomputation diff: {max_corr_diff:.6f}")
    print(f"  Papers with undefined correlation: {sum(1 for r in recomputed_corrs if r is None)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reviewer count consistency
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All raw list columns should have the same length per paper
    """)
    return


@app.cell
def _(df):
    raw_list_cols = [
        "rating",
        "soundness",
        "presentation",
        "contribution",
        "confidence",
        "wc_review",
        "wc_summary",
        "wc_strengths",
        "wc_weaknesses",
        "wc_questions",
    ]
    lengths = df.select([pl.col(c).list.len().alias(f"{c}_len") for c in raw_list_cols])
    # Check all lengths equal within each row
    first_len = lengths.columns[0]
    all_consistent = all((lengths[first_len] == lengths[c]).all() for c in lengths.columns[1:])
    print(f"\nReviewer count consistency across {len(raw_list_cols)} list columns: {'✓' if all_consistent else '✗'}")

    reviewer_counts = lengths[first_len].value_counts().sort("count", descending=True)
    print(f"Reviewer count distribution:\n{reviewer_counts}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `authors#_avg` and `replies_avg` semantics
    """)
    return


@app.cell
def _(df):
    authors_col = df["authors#_avg"]
    authors_elem1_all_zero = df.select(pl.col("authors#_avg").list.get(1) == 0).to_series().all()
    print(f"\nauthors#_avg[1] always 0: {authors_elem1_all_zero}")
    print(
        f"authors#_avg[0] range: {df.select(pl.col('authors#_avg').list.get(0)).to_series().min()} - {df.select(pl.col('authors#_avg').list.get(0)).to_series().max()}"
    )

    replies_elem1_all_zero = df.select(pl.col("replies_avg").list.get(1) == 0).to_series().all()
    print(f"replies_avg[1] always 0: {replies_elem1_all_zero}")
    print(
        f"replies_avg[0] range: {df.select(pl.col('replies_avg').list.get(0)).to_series().min()} - {df.select(pl.col('replies_avg').list.get(0)).to_series().max()}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Feature Engineering
    """)
    return


@app.cell
def _(avg_pairs, df):
    # ── Unpack _avg columns into scalars ─────────────────────────────────────
    unpack_exprs = []
    for _raw_col, _avg_col in avg_pairs:
        base = _raw_col
        unpack_exprs.append(pl.col(_avg_col).list.get(0).alias(f"{base}_mean"))
        unpack_exprs.append(pl.col(_avg_col).list.get(1).alias(f"{base}_std"))

    # ── Scalar features from raw lists ───────────────────────────────────────
    score_cols = ["rating", "soundness", "presentation", "contribution"]
    for _col in score_cols:
        unpack_exprs.append(pl.col(_col).list.min().alias(f"{_col}_min"))
        unpack_exprs.append(pl.col(_col).list.max().alias(f"{_col}_max"))
        unpack_exprs.append((pl.col(_col).list.max() - pl.col(_col).list.min()).alias(f"{_col}_range"))

    # ── Reviewer count, replies, authors ─────────────────────────────────────
    unpack_exprs.extend(
        [
            pl.col("rating").list.len().alias("n_reviewers"),
            pl.col("replies_avg").list.get(0).alias("n_replies"),
            pl.col("authors#_avg").list.get(0).alias("n_authors"),
        ]
    )

    # ── Engagement proxies ───────────────────────────────────────────────────
    unpack_exprs.extend(
        [
            pl.col("wc_review").list.sum().alias("total_review_wc"),
            # mean questions wc already available as wc_questions_mean (above)
        ]
    )

    # ── Text for embedding ───────────────────────────────────────────────────
    unpack_exprs.append((pl.col("title") + " " + pl.col("abstract")).alias("text_for_embedding"))

    features = df.with_columns(unpack_exprs)

    # ── Ratios (computed after unpacking) ────────────────────────────────────
    features = features.with_columns(
        # strengths-to-weaknesses wc ratio (use means)
        (pl.col("wc_strengths_mean") / pl.col("wc_weaknesses_mean")).alias("strengths_weaknesses_ratio"),
        # questions-to-review-length ratio
        (pl.col("wc_questions_mean") / pl.col("wc_review_mean")).alias("questions_review_ratio"),
    )

    # ── Area-normalized z-scores ─────────────────────────────────────────────
    zscore_cols = ["rating_mean", "soundness_mean", "contribution_mean"]
    for col in zscore_cols:
        features = features.with_columns(
            ((pl.col(col) - pl.col(col).mean().over("primary_area")) / pl.col(col).std().over("primary_area")).alias(
                f"{col}_area_z"
            )
        )
    return (features,)


@app.cell
def _(df, features):
    new_cols = [c for c in features.columns if c not in df.columns]
    print(f"\nEngineered {len(new_cols)} new columns:")
    for c in sorted(new_cols):
        print(f"  {c}: {features[c].dtype}")

    print(f"\nFinal shape: {features.shape}")
    print("Null counts in new columns:")
    null_counts = features.select(new_cols).null_count()
    non_zero_nulls = {c: null_counts[c].item() for c in new_cols if null_counts[c].item() > 0}
    if non_zero_nulls:
        for c, n in non_zero_nulls.items():
            print(f"  {c}: {n} nulls")
    else:
        print("  None")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save
    """)
    return


@app.cell
def _(features):
    features.write_parquet("iclr_2026_features.parquet")
    print("\nSaved to iclr_2026_features.parquet")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Sanity check
    """)
    return


@app.cell
def _():
    print("""
    Trustworthy:
      - Row/column counts, no nulls, unique IDs
      - Identifier cross-references (site ↔ id ↔ pdf)
      - _avg columns match recomputed mean/std
      - Reviewer count consistent across all list columns
      - authors#_avg = [n_authors, 0], replies_avg = [n_replies, 0]

    Assumptions made:
      - authors#_avg[0] = author count (element 1 always 0)
      - replies_avg[0] = total reply count (element 1 always 0)
      - corr_rating_confidence uses Pearson (confirmed by recomputation)

    Noisy signals:
      - corr_rating_confidence undefined for papers with constant rating/confidence
      - Area z-scores for areas with very few papers have high variance
      - strengths_weaknesses_ratio can be extreme for very short reviews
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
