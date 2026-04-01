import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    from collections import Counter

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from sklearn.cluster import AgglomerativeClustering
    import marimo as mo

    SEED = 42
    np.random.seed(SEED)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1.5 Keyword Canonicalization

    Author-provided keywords have ~9K unique values with heavy fragmentation
    (e.g. `llm` / `llms` / `large language model` / `large language models`).

    Approach: embed keywords with Qwen3-Embedding-8B → agglomerative clustering
    on cosine distance → pick most frequent member as canonical label.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load keyword embeddings
    """)
    return


@app.cell
def _():
    kw_emb = pl.read_parquet("keyword_embeddings.parquet")
    print(f"Loaded {kw_emb.shape[0]} keyword embeddings")
    return (kw_emb,)


@app.cell
def _(kw_emb):
    embeddings = np.array(kw_emb["embedding"].to_list(), dtype=np.float32)
    keywords = kw_emb["keyword"].to_list()
    print(f"Embedding matrix: {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norms: mean={norms.mean():.4f}, std={norms.std():.6f}")
    if norms.std() > 0.01:
        embeddings = embeddings / norms[:, np.newaxis]
        print("Normalized to unit length")
    return embeddings, keywords


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build keyword frequency from papers
    """)
    return


@app.cell
def _():
    features = pl.read_parquet("iclr_2026_features.parquet")
    kw_freq = (
        features.select("keywords")
        .explode("keywords")
        .with_columns(pl.col("keywords").str.to_lowercase().str.strip_chars())
        .filter(pl.col("keywords") != "")
        .group_by("keywords")
        .len()
        .sort("len", descending=True)
    )
    freq_map = dict(zip(kw_freq["keywords"].to_list(), kw_freq["len"].to_list()))
    print(f"Keyword frequencies: {len(freq_map)} unique, total occurrences={sum(freq_map.values())}")
    print(f"Appearing once: {sum(1 for v in freq_map.values() if v == 1)}")
    return features, freq_map


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sweep distance threshold

    Agglomerative clustering with `metric=cosine`, `linkage=average`.
    Lower threshold = more clusters (finer), higher = fewer (coarser merging).
    """)
    return


@app.cell
def _(embeddings):
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sweep_results = []
    for t in thresholds:
        clust = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=t,
            metric="cosine",
            linkage="average",
        )
        labels = clust.fit_predict(embeddings)
        n_clusters = len(set(labels))
        sweep_results.append({"threshold": t, "n_clusters": n_clusters})
        print(f"  threshold={t:.2f}: {n_clusters} clusters")
    return (sweep_results,)


@app.cell
def _(sweep_results):
    fig_sweep = px.line(
        sweep_results,
        x="threshold",
        y="n_clusters",
        markers=True,
        title="Keyword Clusters vs Cosine Distance Threshold",
        labels={"threshold": "distance threshold", "n_clusters": "number of clusters"},
    )
    fig_sweep.update_layout(height=400)
    fig_sweep
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cluster at chosen threshold
    """)
    return


@app.cell
def _(embeddings, freq_map, keywords):
    THRESHOLD = 0.20

    clust = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    labels = clust.fit_predict(embeddings)
    n_clusters = len(set(labels))
    print(f"Threshold={THRESHOLD}: {n_clusters} clusters from {len(keywords)} keywords")

    # For each cluster, pick the most frequent keyword as canonical label
    cluster_members: dict[int, list[str]] = {}
    for kw, label in zip(keywords, labels):
        cluster_members.setdefault(int(label), []).append(kw)

    canonical_map: dict[str, str] = {}
    cluster_info = []
    for cid, members in cluster_members.items():
        # Sort by frequency (descending), then alphabetically for ties
        members_sorted = sorted(members, key=lambda k: (-freq_map.get(k, 0), k))
        canonical = members_sorted[0]
        total_freq = sum(freq_map.get(m, 0) for m in members)
        for m in members:
            canonical_map[m] = canonical
        cluster_info.append({
            "cluster_id": cid,
            "canonical": canonical,
            "size": len(members),
            "total_freq": total_freq,
            "members": ", ".join(members_sorted[:10]),
        })

    cluster_df = pl.DataFrame(cluster_info).sort("total_freq", descending=True)
    print(f"Singletons: {cluster_df.filter(pl.col('size') == 1).shape[0]}")
    print(f"Clusters with 2+ members: {cluster_df.filter(pl.col('size') > 1).shape[0]}")
    return canonical_map, cluster_df, labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cluster size distribution
    """)
    return


@app.cell
def _(cluster_df):
    fig_sizes = px.histogram(
        cluster_df.to_pandas(),
        x="size",
        nbins=50,
        title="Keyword Cluster Size Distribution",
        labels={"size": "keywords per cluster"},
    )
    fig_sizes.update_layout(height=400)
    fig_sizes
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Top 30 canonical keywords (post-canonicalization)
    """)
    return


@app.cell
def _(cluster_df):
    top30 = cluster_df.head(30)
    fig_top = px.bar(
        top30.to_pandas(),
        x="total_freq",
        y="canonical",
        orientation="h",
        title="Top 30 Canonical Keywords (after merging synonyms)",
        hover_data=["size", "members"],
    )
    fig_top.update_layout(height=700, yaxis={"categoryorder": "total ascending"})
    fig_top
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Largest clusters (inspect merge quality)
    """)
    return


@app.cell
def _(cluster_df):
    largest = cluster_df.filter(pl.col("size") > 1).head(20)
    for row in largest.iter_rows(named=True):
        print(f"\n[{row['canonical']}] ({row['size']} members, freq={row['total_freq']})")
        print(f"  {row['members']}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save mapping and update features
    """)
    return


@app.cell
def _(canonical_map, features, labels, keywords):
    # Save keyword → canonical mapping
    mapping_df = pl.DataFrame({
        "keyword": list(canonical_map.keys()),
        "canonical_keyword": list(canonical_map.values()),
        "cluster_id": [int(labels[keywords.index(k)]) for k in canonical_map],
    })
    mapping_df.write_parquet("keyword_mapping.parquet")
    print(f"Saved keyword_mapping.parquet: {mapping_df.shape[0]} rows")

    # Add canonical_keywords column to features
    def map_keywords(kw_list: list[str]) -> list[str]:
        seen = set()
        result = []
        for k in kw_list:
            canon = canonical_map.get(k.lower().strip(), k.lower().strip())
            if canon not in seen:
                seen.add(canon)
                result.append(canon)
        return result

    canonical_lists = [
        map_keywords(row) for row in features["keywords"].to_list()
    ]
    updated = features.with_columns(
        pl.Series("canonical_keywords", canonical_lists)
    )
    updated.write_parquet("iclr_2026_features.parquet")
    print(f"Updated iclr_2026_features.parquet with canonical_keywords column ({updated.shape[1]} cols)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
