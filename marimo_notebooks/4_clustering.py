import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    from pathlib import Path
    from collections import Counter

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import ward as ward_linkage
    from scipy.spatial.distance import cdist
    import hdbscan
    import marimo as mo

    SEED = 42
    np.random.seed(SEED)
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
    df = pl.read_parquet("iclr_2026_embeddings.parquet")
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return (df,)


@app.cell
def _(df):
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    print(f"PCA columns: {len(pca_cols)}")

    X = df.select(pca_cols).to_numpy()
    print(f"Clustering on PCA space: {X.shape}")
    return (X,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. HDBSCAN
    """)
    return


@app.cell
def _(X):
    best_score = -1
    best_min_size = None
    best_labels = None
    hdbscan_sweep: list[dict] = []

    for min_size in [15, 25, 50, 75]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = (labels == -1).sum() / len(labels)

        # Silhouette on non-noise points only
        non_noise = labels != -1
        if non_noise.sum() > 1 and n_clusters > 1:
            score = silhouette_score(X[non_noise], labels[non_noise])
        else:
            score = -1

        hdbscan_sweep.append(
            {
                "min_cluster_size": min_size,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_frac * 100, 1),
                "silhouette": score if score > -1 else float("nan"),
            }
        )
        print(
            f"  min_cluster_size={min_size:3d}: {n_clusters:3d} clusters, noise={noise_frac:.1%}, silhouette={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_min_size = min_size
            best_labels = labels

    print(
        f"\nBest HDBSCAN: min_cluster_size={best_min_size}, silhouette={best_score:.3f}"
    )
    return best_labels, best_min_size, best_score, hdbscan_sweep, score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.1 Sweep diagnostics
    """)
    return


@app.cell
def _(best_min_size, hdbscan_sweep: list[dict]):
    # Three metrics in one view: silhouette (quality), noise % (coverage loss),
    # n_clusters (granularity). Helps pick min_cluster_size sensibly.
    _sweep = pl.DataFrame(hdbscan_sweep)
    fig1 = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Silhouette Score", "Noise %", "N Clusters"],
    )
    _x = _sweep["min_cluster_size"].to_list()
    fig1.add_trace(
        go.Scatter(
            x=_x,
            y=_sweep["silhouette"].to_list(),
            mode="lines+markers",
            name="silhouette",
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=_x, y=_sweep["noise_pct"].to_list(), mode="lines+markers", name="noise %"
        ),
        row=1,
        col=2,
    )
    fig1.add_trace(
        go.Scatter(
            x=_x,
            y=_sweep["n_clusters"].to_list(),
            mode="lines+markers",
            name="n_clusters",
        ),
        row=1,
        col=3,
    )
    if best_min_size is not None:
        for col_i in [1, 2, 3]:
            fig1.add_vline(
                x=best_min_size,
                line_dash="dash",
                line_color="green",
                annotation_text="best",
                row=1,
                col=col_i,
            )
    fig1.update_xaxes(title_text="min_cluster_size")
    fig1.update_layout(title="HDBSCAN Parameter Sweep", height=380, showlegend=False)
    fig1
    return


@app.cell
def _(best_labels):
    assert best_labels is not None, "HDBSCAN produced no valid clustering"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. KMeans
    """)
    return


@app.cell
def _(X):
    best_k_score = -1
    best_k = None
    kmeans_sweep: list[dict] = []

    for k in [10, 20, 30, 40, 50]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km_labels = km.fit_predict(X)
        score_ = silhouette_score(X, km_labels)
        kmeans_sweep.append({"k": k, "silhouette": score_})
        print(f"  k={k:3d}: silhouette={score_:.3f}")
        if score_ > best_k_score:
            best_k_score = score_
            best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    kmeans_labels = km_final.fit_predict(X)
    print(f"\nBest KMeans: k={best_k}, silhouette={best_k_score:.3f}")
    return best_k, kmeans_labels, kmeans_sweep


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 KMeans silhouette vs k
    """)
    return


@app.cell
def _(best_k, best_score, kmeans_sweep: list[dict]):
    _km_sweep = pl.DataFrame(kmeans_sweep)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=_km_sweep["k"].to_list(),
            y=_km_sweep["silhouette"].to_list(),
            mode="lines+markers",
            name="KMeans silhouette",
            line=dict(color="royalblue"),
        )
    )
    if best_score > -1:
        # Overlay best HDBSCAN silhouette as a horizontal reference
        fig2.add_hline(
            y=best_score,
            line_dash="dash",
            line_color="green",
            annotation_text=f"HDBSCAN best ({best_score:.3f})",
        )
    if best_k is not None:
        fig2.add_vline(
            x=best_k, line_dash="dash", line_color="royalblue", annotation_text="best k"
        )
    fig2.update_layout(
        title="KMeans Silhouette vs k (green = best HDBSCAN)",
        xaxis_title="k",
        yaxis_title="silhouette score",
        height=380,
    )
    fig2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Ward
    """)
    return


@app.cell
def _(X):
    Z = ward_linkage(X)
    return (Z,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.1 Dendrogram
    """)
    return


@app.cell
def _(Z):
    ddata = dendrogram(Z, truncate_mode="lastp", p=100, no_plot=True)
    fig3 = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig3.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="steelblue", width=1),
                showlegend=False,
            )
        )
    fig3.update_layout(
        title="Ward Dendrogram — last 100 merges (look for large vertical jumps to pick k)",
        xaxis_title="sample index (or cluster size in parentheses)",
        yaxis_title="Ward distance (merge cost)",
        height=450,
        width=1000,
    )
    fig3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.2 Sweep
    """)
    return


@app.cell
def _(X, score):
    WARD_K_RANGE = [20, 30, 40, 50]

    print("\n=== Ward Sweep ===")
    best_ward_score = -1
    best_ward_k = None
    best_ward_labels = None
    ward_sweep: list[dict] = []

    for _k in WARD_K_RANGE:
        ward = AgglomerativeClustering(n_clusters=_k, linkage="ward")
        _labels = ward.fit_predict(X)
        _score = silhouette_score(X, _labels, sample_size=2000, random_state=SEED)
        ward_sweep.append({"k": _k, "silhouette": _score})
        print(f"  k={_k:3d}: silhouette={_score:.4f}")
        if score > best_ward_score:
            best_ward_score = _score
            best_ward_k = _k
            best_ward_labels = _labels

    print(f"\nBest Ward: k={best_ward_k}, silhouette={best_ward_score:.4f}")
    assert best_ward_labels is not None, "Ward sweep produced no valid clustering"
    ward_labels = best_ward_labels
    return best_ward_k, ward_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Cluster Labeling
    """)
    return


@app.cell
def _(X, best_ward_k, df, kmeans_labels, ward_labels):
    n_ward = best_ward_k
    print(f"\n=== Cluster Analysis ({n_ward} Ward clusters) ===")

    df_clustered = df.with_columns(
        pl.Series("cluster_ward", ward_labels),
        pl.Series("cluster_kmeans", kmeans_labels),
    )

    cluster_ids = sorted(set(ward_labels))
    cluster_summaries = []

    for cid in cluster_ids:
        mask = ward_labels == cid
        cluster_df = df_clustered.filter(pl.Series(mask))
        n = cluster_df.shape[0]

        # Top keywords
        kw_str = cluster_df.select("keywords").explode("keywords").to_series().to_list()
        kw_counts = Counter(k.lower().strip() for k in kw_str if k)
        top_kw = [k for k, _ in kw_counts.most_common(5)]

        # Dominant areas
        top_areas = (
            cluster_df.group_by("primary_area")
            .len()
            .sort("len", descending=True)["primary_area"]
            .head(3)
            .to_list()
        )

        # Representative papers (nearest to centroid in feature space)
        cluster_X = X[mask]
        centroid = cluster_X.mean(axis=0)
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        rep_papers = cluster_df[np.argsort(dists)[:5].tolist()]

        mean_rating_val = cluster_df["rating_mean"].mean()
        mean_rating: float = (
            mean_rating_val if isinstance(mean_rating_val, (int, float)) else 0.0
        )  # type: ignore[assignment]

        cluster_summaries.append(
            {
                "cluster_id": cid,
                "label": f"Cluster {cid}",
                "size": n,
                "mean_rating": round(mean_rating, 2),
                "top_keywords": ", ".join(top_kw),
                "top_areas": ", ".join(top_areas),
            }
        )

        # Print details for small clusters (outlier themes) and the largest ones
        if (
            n <= 30
            or cid in sorted(cluster_ids, key=lambda c: (ward_labels == c).sum())[-5:]
        ):
            print(f"\n--- Cluster {cid} (n={n}, mean_rating={mean_rating:.2f}) ---")
            print(f"  Keywords: {', '.join(top_kw)}")
            print(f"  Areas: {', '.join(top_areas)}")
            print("  Representative papers:")
            for row in rep_papers.iter_rows(named=True):
                print(f"    - {row['title'][:80]}  (rating={row['rating_mean']:.1f})")

    summary_df = pl.DataFrame(cluster_summaries)
    print("\n=== Cluster Size Distribution (top 20) ===")
    print(summary_df.sort("size", descending=True).head(20))
    return cluster_ids, df_clustered, n_ward, summary_df


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4.1 Cluster size bar chart
    """)
    return


@app.cell
def _(n_ward, summary_df):
    _summary_sorted = summary_df.sort("size", descending=True).to_pandas()
    fig4 = px.bar(
        _summary_sorted,
        x="label",
        y="size",
        color="mean_rating",
        color_continuous_scale="RdYlGn",
        title=f"Ward Cluster Sizes (k={n_ward}, color = mean rating)",
    )
    fig4.update_layout(
        xaxis_tickangle=-60,
        height=450,
        coloraxis_colorbar_title="mean rating",
        xaxis={"tickfont": {"size": 9}},
    )
    fig4
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4.2 UMAP colored by Ward cluster
    """)
    return


@app.cell
def _(df_clustered, n_ward):
    _umap_df = df_clustered.select(
        "umap_x",
        "umap_y",
        "cluster_ward",
        "title",
        "rating_mean",
        "primary_area",
    ).with_columns(pl.col("cluster_ward").cast(pl.String).alias("cluster_label"))
    fig5 = px.scatter(
        _umap_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        hover_name="title",
        hover_data=["rating_mean", "primary_area"],
        opacity=0.55,
        title=f"UMAP: Colored by Ward Cluster (k={n_ward})",
    )
    fig5.update_layout(height=650, width=900)
    fig5
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4.3 Rating distributions per cluster
    """)
    return


@app.cell
def _(df_clustered):
    _rating_df = df_clustered.select(
        pl.col("cluster_ward").cast(pl.String).alias("cluster"),
        "rating_mean",
    ).to_pandas()
    _median_ser = _rating_df.groupby("cluster")["rating_mean"].median()
    _order_by_median: list[str] = sorted(
        _median_ser.index.tolist(), key=lambda c: -float(_median_ser[c])
    )
    fig6 = px.box(
        _rating_df,
        x="cluster",
        y="rating_mean",
        category_orders={"cluster": _order_by_median},
        title="Rating Distribution per Cluster (sorted by median rating)",
        points=False,  # suppress individual points — too many with k=50+
    )
    fig6.update_layout(
        xaxis_tickangle=-60,
        height=450,
        xaxis={"tickfont": {"size": 8}},
    )
    fig6
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4.4 Cross-Cluster Structure
    """)
    return


@app.cell
def _(X, cluster_ids, df, df_clustered, ward_labels):
    centroids = {cid: X[ward_labels == cid].mean(axis=0) for cid in cluster_ids}
    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])

    all_dists = cdist(X, centroid_matrix, metric="euclidean")

    # Bridge papers: ratio of 2nd-closest to closest centroid distance.
    # Ratio ≈ 1.0 → paper sits equidistant between two clusters.
    sorted_dists = np.sort(all_dists, axis=1)
    bridge_ratio = sorted_dists[:, 1] / (sorted_dists[:, 0] + 1e-10)

    bridge_idx = np.argsort(bridge_ratio)[:20]
    print("\n=== Top 20 Bridge Papers (close to 2+ clusters) ===")
    for idx in bridge_idx:
        c1, c2 = (
            cluster_ids[np.argsort(all_dists[idx])[0]],
            cluster_ids[np.argsort(all_dists[idx])[1]],
        )
        print(
            f"  ratio={bridge_ratio[idx]:.3f}: clusters [{c1}, {c2}] '{df['title'][int(idx)][:70]}'"
        )

    df_clustered2 = df_clustered.with_columns(
        pl.Series("bridge_ratio", bridge_ratio),
        pl.Series("dist_to_nearest_centroid", sorted_dists[:, 0]),
    )
    return bridge_ratio, df_clustered2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.5.1 Area composition per cluster
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Which research themes live in each cluster?
    """)
    return


@app.cell
def _(df_clustered2, n_ward):
    _area_comp = (
        df_clustered2.group_by("cluster_ward", "primary_area")
        .len()
        .with_columns(
            pl.col("cluster_ward").cast(pl.String).alias("cluster"),
            (pl.col("len") / pl.col("len").sum().over("cluster_ward")).alias(
                "fraction"
            ),
            pl.col("primary_area").str.slice(0, 35).alias("area_short"),
        )
    )
    # Order clusters by their top area (groups thematically similar clusters)
    _dominant_area = (
        _area_comp.sort("len", descending=True)
        .group_by("cluster")
        .first()
        .select("cluster", "area_short")
        .sort("area_short")
    )
    _cluster_order = _dominant_area["cluster"].to_list()

    fig7 = px.bar(
        _area_comp.to_pandas(),
        x="cluster",
        y="fraction",
        color="area_short",
        barmode="stack",
        category_orders={"cluster": _cluster_order},
        title=f"Primary Area Composition per Ward Cluster (k={n_ward}, sorted by dominant area)",
        labels={"fraction": "fraction of cluster", "area_short": "area"},
    )
    fig7.update_layout(
        height=550,
        xaxis_tickangle=-60,
        legend_title="area",
        xaxis={"tickfont": {"size": 8}},
    )
    fig7
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.5.2 Bridge ratio distribution
    """)
    return


@app.cell
def _(bridge_ratio):
    fig8 = go.Figure()
    fig8.add_trace(go.Histogram(x=bridge_ratio, nbinsx=60, name="bridge ratio"))
    fig8.add_vline(
        x=float(np.percentile(bridge_ratio, 5)),
        line_dash="dash",
        line_color="red",
        annotation_text="p5 (bridge candidates)",
    )
    fig8.update_layout(
        title="Bridge Ratio Distribution (≈1.0 → equidistant from 2 clusters)",
        xaxis_title="dist_2nd_nearest / dist_nearest",
        yaxis_title="count",
        height=380,
    )
    fig8
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Save
    """)
    return


@app.cell
def _(df_clustered2, n_ward, summary_df):
    df_clustered2.write_parquet("iclr_2026_clustered.parquet")
    summary_df.write_parquet("cluster_summary.parquet")
    print(
        f"\nSaved {df_clustered2.shape[0]} rows × {df_clustered2.shape[1]} columns to iclr_2026_clustered.parquet"
    )
    print(f"Saved cluster summary ({n_ward} clusters) to cluster_summary.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
