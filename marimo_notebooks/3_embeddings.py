import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import umap
    import marimo as mo

    EMBEDDINGS_FILE = Path("embeddings.parquet")
    OUTPUT_FILE = "iclr_2026_embeddings.parquet"
    SEED = 42
    np.random.seed(SEED)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _():
    features = pl.read_parquet("iclr_2026_features.parquet")
    print(f"Loaded features: {features.shape[0]} rows × {features.shape[1]} columns")
    return (features,)


@app.cell
def _(features):
    texts_df = features.select("openreview_id", "text_for_embedding")
    texts_df.write_parquet("texts_for_embedding.parquet")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Generate embeddings
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## vLLM Serve
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```sh
    vllm serve Qwen/Qwen3-Embedding-8B \
      --runner pooling \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype auto \
      --max-model-len 32768 \
      --max-num-seqs 256 \
      --gpu-memory-utilization 0.95 \
      --enforce-eager \
      --data-parallel-size 8 \
      --disable-log-requests
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run python script to generate embedings
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `uv run python embed-script.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Analyze results
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 Load and validate embeddings
    """)
    return


@app.cell
def _():
    emb_df = pl.read_parquet(EMBEDDINGS_FILE)
    print(f"\nLoaded embeddings: {emb_df.shape[0]} rows, columns={emb_df.columns}")
    assert "openreview_id" in emb_df.columns, "embeddings.parquet must have openreview_id"
    assert "embedding" in emb_df.columns, "embeddings.parquet must have embedding"
    return (emb_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Validate coverage before joining
    """)
    return


@app.cell
def _(emb_df, features):
    missing = features.filter(~pl.col("openreview_id").is_in(emb_df["openreview_id"])).shape[0]
    print(f"Papers without embedding: {missing}")
    assert missing == 0, f"{missing} papers have no embedding — recheck inference output"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Join embeddings onto featuresdf = features.join(
        emb_df.select("openreview_id", "embedding"), on="openreview_id", how="left"
    )
    print(f"After join: {df.shape[0]} rows × {df.shape[1]} columns")
    """)
    return


@app.cell
def _(emb_df, features):
    df = features.join(emb_df.select("openreview_id", "embedding"), on="openreview_id", how="left")
    print(f"After join: {df.shape[0]} rows × {df.shape[1]} columns")
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Extract numpy array# list[f32] column → (N, D) float32 array
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    print(f"Embedding matrix: shape={embeddings.shape}, dtype={embeddings.dtype}")
    """)
    return


@app.cell
def _(df):
    # list[f32] column → (N, D) float32 array
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    print(f"Embedding matrix: shape={embeddings.shape}, dtype={embeddings.dtype}")
    return (embeddings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.2 Check normalization
    """)
    return


@app.cell
def _(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.3 PCA for denoising
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First fit a full PCA to inspect the scree plot, then keep 90% variance
    """)
    return


@app.cell
def _(embeddings):
    pca_full = PCA(n_components=embeddings.shape[1], random_state=SEED)
    pca_full.fit(embeddings)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"\nPCA variance thresholds: 90%→{n90} components, 95%→{n95}, 99%→{n99}")
    return (cumvar,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA scree / cumulative variance
    """)
    return


@app.cell
def _(cumvar):
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            y=cumvar[:1_000],
            mode="lines+markers",
            name="Cumulative variance",
            marker=dict(size=3),
        )
    )
    for thresh, label, color in [
        (0.90, "90%", "red"),
        (0.95, "95%", "orange"),
        (0.99, "99%", "green"),
    ]:
        fig1.add_hline(y=thresh, line_dash="dash", line_color=color, annotation_text=label)
    fig1.update_layout(
        title="PCA Cumulative Explained Variance (first 1000 components)",
        xaxis_title="n_components",
        yaxis_title="cumulative variance",
        height=400,
    )
    fig1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit final PCA at 90% thresholdpca = PCA(n_components=0.90, random_state=SEED)
    embeddings_pca = pca.fit_transform(embeddings)
    print(
        f"Final PCA: {embeddings.shape[1]}D → {embeddings_pca.shape[1]}D "
        f"({pca.explained_variance_ratio_.sum():.4f} variance)"
    )
    """)
    return


@app.cell
def _(embeddings):
    pca = PCA(n_components=0.90, random_state=SEED)
    embeddings_pca = pca.fit_transform(embeddings)
    print(
        f"Final PCA: {embeddings.shape[1]}D → {embeddings_pca.shape[1]}D "
        f"({pca.explained_variance_ratio_.sum():.4f} variance)"
    )
    return (embeddings_pca,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.4 UMAP for visualization
    """)
    return


@app.cell
def _(embeddings_pca):
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
    umap_coords = reducer.fit_transform(embeddings_pca)
    print(f"UMAP: {embeddings_pca.shape[1]} → 2D")
    return (umap_coords,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2.4.1 UMAP colored by primary area
    """)
    return


@app.cell
def _(df, umap_coords):
    plot_df = pl.DataFrame(
        {
            "umap_x": umap_coords[:, 0],
            "umap_y": umap_coords[:, 1],
            "primary_area": df["primary_area"].to_list(),
            "title": df["title"].to_list(),
            "rating_mean": df["rating_mean"].to_list(),
            "status": df["status"].to_list(),
        }
    )
    fig2 = px.scatter(
        plot_df.to_pandas(),
        x="umap_x",
        y="umap_y",
        color="primary_area",
        hover_name="title",
        hover_data=["rating_mean", "status"],
        opacity=0.5,
        title="UMAP: Colored by Primary Area (sanity check)",
    )
    fig2.update_layout(height=700, width=1200)
    return (plot_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2.4.2 UMAP colored by rating mean
    """)
    return


@app.cell
def _(plot_df):
    fig3 = px.scatter(
        plot_df.to_pandas(),
        x="umap_x",
        y="umap_y",
        color="rating_mean",
        hover_name="title",
        color_continuous_scale="RdYlGn",
        opacity=0.6,
        title="UMAP: Colored by Rating Mean",
    )
    fig3.update_layout(height=600, width=800)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.5 Nearest neighbors & local density
    """)
    return


@app.cell
def _(embeddings):
    K = 10
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return K, distances


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exclude self (index 0), compute mean distance to K nearest
    """)
    return


@app.cell
def _(K, distances):
    mean_knn_dist = distances[:, 1:].mean(axis=1)
    print(f"Local density (mean {K}-NN cosine dist):")
    print(f"  mean={mean_knn_dist.mean():.4f}, std={mean_knn_dist.std():.4f}")
    print(f"  min={mean_knn_dist.min():.4f} (densest), max={mean_knn_dist.max():.4f} (most isolated)")
    return (mean_knn_dist,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2.5.1 kNN distance distribution
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Right tail = semantic outliers; left = papers in crowded topics
    """)
    return


@app.cell
def _(K, mean_knn_dist):
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=mean_knn_dist, nbinsx=80, name="mean kNN cosine dist"))
    fig4.update_layout(
        title=f"Local Density Distribution (mean {K}-NN cosine distance)",
        xaxis_title="mean kNN distance (higher = more isolated)",
        yaxis_title="count",
        height=400,
    )
    fig4
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Semantic outliers
    """)
    return


@app.cell
def _(mean_knn_dist):
    mean_knn_dist.shape
    return


@app.cell
def _(df, mean_knn_dist):
    outlier_idx = np.argsort(mean_knn_dist)[-20:][::-1]
    titles = df["title"].to_list()

    print("\n=== Top 20 Semantic Outliers (most isolated) ===")
    for idx in outlier_idx:
        i = int(idx)
        print(f"  dist={mean_knn_dist[i]:.4f}: {titles[i][:80]}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save output
    """)
    return


@app.cell
def _(df, embeddings_pca, mean_knn_dist, umap_coords):
    # Store: UMAP coords, kNN density, all PCA components
    # Drop the raw `embedding` list column — it's large and downstream scripts
    # use PCA components for clustering
    pca_series = [pl.Series(f"pca_{i}", embeddings_pca[:, i]) for i in range(embeddings_pca.shape[1])]
    result = df.drop("embedding").with_columns(  # raw vectors not needed downstream
        pl.Series("umap_x", umap_coords[:, 0]),
        pl.Series("umap_y", umap_coords[:, 1]),
        pl.Series("mean_knn_distance", mean_knn_dist),
        *pca_series,
    )

    result.write_parquet(OUTPUT_FILE)
    print(f"\nSaved {result.shape[0]} rows × {result.shape[1]} columns to {OUTPUT_FILE}")
    print(f"  New columns: umap_x, umap_y, mean_knn_distance, pca_0…pca_{embeddings_pca.shape[1] - 1}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
