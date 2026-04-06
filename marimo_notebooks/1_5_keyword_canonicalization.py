import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import plotly.express as px
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

    Approach: manual pre-merge for known abbreviations → embed keywords with
    Qwen3-Embedding-8B → agglomerative clustering on cosine distance → pick
    most frequent member as canonical label.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load keyword embeddings and paper frequencies
    """)
    return


@app.cell
def _():
    kw_emb = pl.read_parquet("keyword_embeddings.parquet")
    print(f"Loaded {kw_emb.shape[0]} keyword embeddings")
    return (kw_emb,)


@app.cell
def _(kw_emb):
    kw_embeddings = np.array(kw_emb["embedding"].to_list(), dtype=np.float32)
    kw_list = kw_emb["keyword"].to_list()
    print(f"Embedding matrix: {kw_embeddings.shape}")

    kw_norms = np.linalg.norm(kw_embeddings, axis=1)
    print(f"Norms: mean={kw_norms.mean():.4f}, std={kw_norms.std():.6f}")
    if kw_norms.std() > 0.01:
        kw_embeddings = kw_embeddings / kw_norms[:, np.newaxis]
        print("Normalized to unit length")
    return kw_embeddings, kw_list


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
    print(
        f"Keyword frequencies: {len(freq_map)} unique, total occurrences={sum(freq_map.values())}"
    )
    print(f"Appearing once: {sum(1 for v in freq_map.values() if v == 1)}")
    return features, freq_map


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Manual pre-merge: abbreviations → full forms

    Short abbreviations (2-4 chars) embed far from their full forms because
    the embedding model treats them as generic tokens. These must be mapped
    explicitly before clustering.
    """)
    return


@app.cell
def _():
    MANUAL_MERGES: dict[str, str] = {
        # Common ML abbreviations
        "rl": "reinforcement learning",
        "llm": "large language model",
        "llms": "large language model",
        "lmm": "large multimodal model",
        "lmms": "large multimodal model",
        "mllm": "multimodal large language model",
        "mllms": "multimodal large language model",
        "vlm": "vision-language model",
        "vlms": "vision-language model",
        "gnn": "graph neural network",
        "gnns": "graph neural network",
        "cnn": "convolutional neural network",
        "cnns": "convolutional neural network",
        "vae": "variational autoencoder",
        "vaes": "variational autoencoder",
        "gan": "generative adversarial network",
        "gans": "generative adversarial network",
        "dpo": "direct preference optimization",
        "rlhf": "reinforcement learning from human feedback",
        "ppo": "proximal policy optimization",
        "sft": "supervised fine-tuning",
        "rag": "retrieval-augmented generation",
        "icl": "in-context learning",
        "cot": "chain-of-thought",
        "chain of thought": "chain-of-thought",
        "moe": "mixture of experts",
        "nerf": "neural radiance fields",
        "nerfs": "neural radiance fields",
        "snn": "spiking neural network",
        "snns": "spiking neural network",
        "ssl": "self-supervised learning",
        "cl": "continual learning",
        "tts": "text-to-speech",
        "asr": "automatic speech recognition",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "od": "object detection",
        "nas": "neural architecture search",
        "gflownet": "gflownets",
        "pde": "partial differential equations",
        "pdes": "partial differential equations",
        "ode": "ordinary differential equations",
        "odes": "ordinary differential equations",
        "dp": "differential privacy",
        "fl": "federated learning",
        "marl": "multi-agent reinforcement learning",
        "lora": "low-rank adaptation",
        "peft": "parameter-efficient fine-tuning",
        "3dgs": "3d gaussian splatting",
        "kd": "knowledge distillation",
    }
    print(f"Manual merges defined: {len(MANUAL_MERGES)} abbreviation → full form")
    return (MANUAL_MERGES,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Apply manual merges to embedding index

    Replace each abbreviation's embedding with its full form's embedding,
    so clustering groups them together.
    """)
    return


@app.cell
def _(MANUAL_MERGES: dict[str, str], kw_embeddings, kw_list):
    kw_to_idx = {kw: i for i, kw in enumerate(kw_list)}

    merged_embeddings = kw_embeddings.copy()
    merges_applied = 0
    for abbrev, full_form in MANUAL_MERGES.items():
        if abbrev in kw_to_idx and full_form in kw_to_idx:
            merged_embeddings[kw_to_idx[abbrev]] = kw_embeddings[kw_to_idx[full_form]]
            merges_applied += 1

    print(
        f"Applied {merges_applied} embedding replacements (abbrev → full form vector)"
    )
    return (merged_embeddings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sweep distance threshold
    """)
    return


@app.cell
def _(merged_embeddings):
    thresholds = [0.05, 0.20, 0.30, 0.40, 0.45, 0.50]
    sweep_results = []
    for t in thresholds:
        sweep_clust = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=t,
            metric="cosine",
            linkage="average",
        )
        sweep_labels = sweep_clust.fit_predict(merged_embeddings)
        n = len(set(sweep_labels))
        sweep_results.append({"threshold": t, "n_clusters": n})
        print(f"  threshold={t:.2f}: {n} clusters")
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
def _(MANUAL_MERGES: dict[str, str], freq_map, kw_list, merged_embeddings):
    THRESHOLD = 0.40

    final_clust = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    final_labels = final_clust.fit_predict(merged_embeddings)
    print(
        f"Threshold={THRESHOLD}: {len(set(final_labels))} clusters from {len(kw_list)} keywords"
    )

    # Build cluster members
    cluster_members: dict[int, list[str]] = {}
    for kw, lbl in zip(kw_list, final_labels):
        cluster_members.setdefault(int(lbl), []).append(kw)

    # Build canonical map: most frequent keyword per cluster as canonical,
    # with manual merge targets taking priority
    manual_targets = set(MANUAL_MERGES.values())
    canonical_map: dict[str, str] = {}
    cluster_info = []
    for cid, members in cluster_members.items():
        members_sorted = sorted(members, key=lambda k: (-freq_map.get(k, 0), k))
        # Prefer manual merge targets as canonical if present in the cluster
        canonical = next(
            (m for m in members_sorted if m in manual_targets), members_sorted[0]
        )
        total_freq = sum(freq_map.get(m, 0) for m in members)
        for m in members:
            canonical_map[m] = canonical
        # Also map manual abbreviations that may not be in this cluster
        for _abbrev, target in MANUAL_MERGES.items():
            if target == canonical and _abbrev in canonical_map:
                canonical_map[_abbrev] = canonical
        cluster_info.append(
            {
                "cluster_id": cid,
                "canonical": canonical,
                "size": len(members),
                "total_freq": total_freq,
                "members": ", ".join(members_sorted),
            }
        )

    cluster_df = pl.DataFrame(cluster_info).sort("total_freq", descending=True)
    print(f"Singletons: {cluster_df.filter(pl.col('size') == 1).shape[0]}")
    print(f"Clusters with 2+ members: {cluster_df.filter(pl.col('size') > 1).shape[0]}")
    return canonical_map, cluster_df, final_labels


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
    cluster_df
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
    largest_clusters = cluster_df.filter(pl.col("size") > 1).head(20)
    for row in largest_clusters.iter_rows(named=True):
        print(
            f"\n[{row['canonical']}] ({row['size']} members, freq={row['total_freq']})"
        )
        print(f"  {row['members']}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save mapping and update features
    """)
    return


@app.cell
def _(canonical_map: dict[str, str], features, final_labels, kw_list):
    kw_to_label = {kw: int(lbl) for kw, lbl in zip(kw_list, final_labels)}
    mapping_df = pl.DataFrame(
        {
            "keyword": list(canonical_map.keys()),
            "canonical_keyword": list(canonical_map.values()),
            "cluster_id": [kw_to_label.get(k, -1) for k in canonical_map],
        }
    )
    mapping_df.write_parquet("keyword_mapping.parquet")
    print(f"Saved keyword_mapping.parquet: {mapping_df.shape[0]} rows")

    def map_keywords(kw_row: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for k in kw_row:
            canon = canonical_map.get(k.lower().strip(), k.lower().strip())
            if canon not in seen:
                seen.add(canon)
                result.append(canon)
        return result

    canonical_lists = [map_keywords(row) for row in features["keywords"].to_list()]
    updated_features = features.with_columns(
        pl.Series("canonical_keywords", canonical_lists)
    )
    updated_features.write_parquet("iclr_2026_features.parquet")
    print(
        f"Updated iclr_2026_features.parquet with canonical_keywords column ({updated_features.shape[1]} cols)"
    )
    return


@app.cell
def _(cluster_df):
    cluster_df
    return


@app.cell
def _(cluster_df):
    cluster_df.write_csv("keywords_clusters.tsv", separator="\t", include_header=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
