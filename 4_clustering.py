"""Phase 4: Clustering & Theme Discovery.

HDBSCAN (primary) + KMeans (baseline) on embedding/PCA space if available,
otherwise falls back to keyword TF-IDF + numeric features.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

SEED = 42
np.random.seed(SEED)
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
# Inline progress plots use .show() — not saved to figures/

# ── Load data ────────────────────────────────────────────────────────────
INPUT_FILE = "iclr_2026_embeddings.parquet"
if not Path(INPUT_FILE).exists():
    INPUT_FILE = "iclr_2026_features.parquet"

df = pl.read_parquet(INPUT_FILE)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {INPUT_FILE}")

# ── Determine if we have embeddings ──────────────────────────────────────
pca_cols = [c for c in df.columns if c.startswith("pca_")]
has_embeddings = len(pca_cols) > 0
has_umap = "umap_x" in df.columns
print(
    f"Embedding PCA columns found: {len(pca_cols)} → {'full mode' if has_embeddings else 'fallback mode'}"
)

# ── Build clustering feature matrix ─────────────────────────────────────
if has_embeddings:
    X = df.select(pca_cols).to_numpy()
    print(f"Clustering on PCA space: {X.shape}")
else:
    # Fallback: TF-IDF on keywords + scaled numeric features
    print("Building fallback feature matrix (TF-IDF keywords + numeric)")

    # TF-IDF on keywords (join per-paper keywords into a single string)
    keyword_texts = df.select(pl.col("keywords").list.join(" ")).to_series().to_list()

    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(keyword_texts).toarray()

    # Numeric features
    numeric_cols = [
        "rating_mean",
        "soundness_mean",
        "presentation_mean",
        "contribution_mean",
        "confidence_mean",
        "rating_std",
        "n_reviewers",
        "n_replies",
        "wc_review_mean",
        "strengths_weaknesses_ratio",
    ]
    numeric_matrix = df.select(numeric_cols).to_numpy()
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_matrix)

    X = np.hstack([tfidf_matrix, numeric_scaled])
    print(
        f"Fallback feature matrix: {X.shape} (TF-IDF: {tfidf_matrix.shape[1]}, numeric: {numeric_scaled.shape[1]})"
    )

# ══════════════════════════════════════════════════════════════════════════
# HDBSCAN (primary)
# ══════════════════════════════════════════════════════════════════════════

print("\n=== HDBSCAN Sweep ===")
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
        f"  min_cluster_size={min_size:3d}: {n_clusters:3d} clusters, "
        f"noise={noise_frac:.1%}, silhouette={score:.3f}"
    )

    if score > best_score:
        best_score = score
        best_min_size = min_size
        best_labels = labels

print(f"\nBest HDBSCAN: min_cluster_size={best_min_size}, silhouette={best_score:.3f}")

# ── PLOT 1: HDBSCAN sweep diagnostics ────────────────────────────────────
# Three metrics in one view: silhouette (quality), noise % (coverage loss),
# n_clusters (granularity). Helps pick min_cluster_size sensibly.
_sweep = pl.DataFrame(hdbscan_sweep)
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Silhouette Score", "Noise %", "N Clusters"],
)
_x = _sweep["min_cluster_size"].to_list()
fig.add_trace(
    go.Scatter(
        x=_x, y=_sweep["silhouette"].to_list(), mode="lines+markers", name="silhouette"
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=_x, y=_sweep["noise_pct"].to_list(), mode="lines+markers", name="noise %"
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=_x, y=_sweep["n_clusters"].to_list(), mode="lines+markers", name="n_clusters"
    ),
    row=1,
    col=3,
)
if best_min_size is not None:
    for col_i in [1, 2, 3]:
        fig.add_vline(
            x=best_min_size,
            line_dash="dash",
            line_color="green",
            annotation_text="best",
            row=1,
            col=col_i,
        )
fig.update_xaxes(title_text="min_cluster_size")
fig.update_layout(title="HDBSCAN Parameter Sweep", height=380, showlegend=False)
fig.show()
assert best_labels is not None, "HDBSCAN produced no valid clustering"
hdbscan_labels = best_labels
n_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

# ══════════════════════════════════════════════════════════════════════════
# KMeans (baseline comparison)
# ══════════════════════════════════════════════════════════════════════════

print("\n=== KMeans Sweep ===")
best_k_score = -1
best_k = None
kmeans_sweep: list[dict] = []

for k in [10, 20, 30, 40, 50]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(X)
    score = silhouette_score(X, km_labels)
    kmeans_sweep.append({"k": k, "silhouette": score})
    print(f"  k={k:3d}: silhouette={score:.3f}")
    if score > best_k_score:
        best_k_score = score
        best_k = k

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
kmeans_labels = km_final.fit_predict(X)
kmeans_centroids = km_final.cluster_centers_
print(f"\nBest KMeans: k={best_k}, silhouette={best_k_score:.3f}")

# ── PLOT 2: KMeans silhouette vs k ────────────────────────────────────────
# HDBSCAN and KMeans silhouette on the same scale for direct comparison
_km_sweep = pl.DataFrame(kmeans_sweep)
fig = go.Figure()
fig.add_trace(
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
    fig.add_hline(
        y=best_score,
        line_dash="dash",
        line_color="green",
        annotation_text=f"HDBSCAN best ({best_score:.3f})",
    )
if best_k is not None:
    fig.add_vline(
        x=best_k, line_dash="dash", line_color="royalblue", annotation_text="best k"
    )
fig.update_layout(
    title="KMeans Silhouette vs k (green = best HDBSCAN)",
    xaxis_title="k",
    yaxis_title="silhouette score",
    height=380,
)
fig.show()

# Use HDBSCAN as primary assignment
cluster_labels = hdbscan_labels
cluster_source = "hdbscan"

# ══════════════════════════════════════════════════════════════════════════
# CLUSTER LABELING
# ══════════════════════════════════════════════════════════════════════════

print(f"\n=== Cluster Analysis ({n_hdbscan} HDBSCAN clusters) ===")

df_clustered = df.with_columns(
    pl.Series("cluster_hdbscan", hdbscan_labels),
    pl.Series("cluster_kmeans", kmeans_labels),
)

# Per-cluster summary
cluster_ids = sorted(set(hdbscan_labels))
cluster_summaries = []

for cid in cluster_ids:
    mask = hdbscan_labels == cid
    cluster_df = df_clustered.filter(pl.Series(mask))
    n = cluster_df.shape[0]

    # Top keywords
    all_kw = cluster_df.select("keywords").explode("keywords")
    kw_str = all_kw.to_series().to_list()
    kw_counts = Counter(k.lower().strip() for k in kw_str if k)
    top_kw = [k for k, _ in kw_counts.most_common(5)]

    # Dominant areas
    area_counts = cluster_df.group_by("primary_area").len().sort("len", descending=True)
    top_areas = area_counts["primary_area"].head(3).to_list()

    # Representative papers (nearest to cluster centroid in feature space)
    cluster_X = X[mask]
    centroid = cluster_X.mean(axis=0)
    dists = np.linalg.norm(cluster_X - centroid, axis=1)
    rep_indices = np.argsort(dists)[:5]
    rep_papers = cluster_df[rep_indices.tolist()]

    # Mean rating
    mean_rating_val = cluster_df["rating_mean"].mean()
    mean_rating: float = (
        mean_rating_val if isinstance(mean_rating_val, (int, float)) else 0.0
    )  # type: ignore[assignment]

    label_name = "NOISE" if cid == -1 else f"Cluster {cid}"
    cluster_summaries.append(
        {
            "cluster_id": cid,
            "label": label_name,
            "size": n,
            "mean_rating": round(mean_rating, 2),
            "top_keywords": ", ".join(top_kw),
            "top_areas": ", ".join(top_areas),
        }
    )

    if cid != -1 or n < 200:
        print(f"\n--- {label_name} (n={n}, mean_rating={mean_rating:.2f}) ---")
        print(f"  Keywords: {', '.join(top_kw)}")
        print(f"  Areas: {', '.join(top_areas)}")
        print("  Representative papers:")
        for row in rep_papers.iter_rows(named=True):
            print(f"    - {row['title'][:80]}  (rating={row['rating_mean']:.1f})")

summary_df = pl.DataFrame(cluster_summaries)
print("\n=== Cluster Size Distribution ===")
print(summary_df.sort("size", descending=True))

# ── PLOT 3: Cluster size bar chart ────────────────────────────────────────
# Color-coded by mean rating — size alone doesn't indicate quality.
# NOISE is shown separately so it doesn't visually dominate.
_summary_sorted = summary_df.sort("size", descending=True).to_pandas()
fig = px.bar(
    _summary_sorted,
    x="label",
    y="size",
    color="mean_rating",
    color_continuous_scale="RdYlGn",
    text="size",
    title="Cluster Size (color = mean rating)",
)
fig.update_traces(textposition="outside")
fig.update_layout(
    xaxis_tickangle=-30, height=420, coloraxis_colorbar_title="mean rating"
)
fig.show()

# ── PLOT 4: UMAP colored by HDBSCAN cluster ──────────────────────────────
# Only meaningful when embeddings were computed; skip gracefully otherwise.
if has_umap:
    _umap_df = (
        df_clustered.select(
            "umap_x",
            "umap_y",
            "cluster_hdbscan",
            "title",
            "rating_mean",
            "primary_area",
        )
        .with_columns(pl.col("cluster_hdbscan").cast(pl.String).alias("cluster_label"))
        .to_pandas()
    )
    fig = px.scatter(
        _umap_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        hover_name="title",
        hover_data=["rating_mean", "primary_area"],
        opacity=0.55,
        title="UMAP: Colored by HDBSCAN Cluster (-1 = noise)",
    )
    fig.update_layout(height=650, width=900)
    fig.show()
else:
    print("⚠ UMAP plot skipped (no embeddings)")

# ── PLOT 5: Rating distributions per cluster ─────────────────────────────
# Are clusters differentiated by review quality, or just by topic?
# Box plots keep it readable even with many clusters.
_rating_df = df_clustered.select(
    pl.col("cluster_hdbscan").cast(pl.String).alias("cluster"),
    "rating_mean",
).to_pandas()
_order = (
    summary_df.sort("size", descending=True)["cluster_id"].cast(pl.String).to_list()
)
fig = px.box(
    _rating_df,
    x="cluster",
    y="rating_mean",
    category_orders={"cluster": _order},
    title="Rating Distribution per Cluster (sorted by cluster size)",
    points="outliers",
)
fig.update_layout(xaxis_tickangle=-30, height=420)
fig.show()

# ══════════════════════════════════════════════════════════════════════════
# CROSS-CLUSTER STRUCTURE
# ══════════════════════════════════════════════════════════════════════════

# Compute cluster centroids for HDBSCAN (non-noise only)
centroids = {}
for cid in cluster_ids:
    if cid == -1:
        continue
    mask = hdbscan_labels == cid
    centroids[cid] = X[mask].mean(axis=0)

if centroids:
    centroid_matrix = np.array([centroids[cid] for cid in sorted(centroids.keys())])
    centroid_ids = sorted(centroids.keys())

    # Distance from each paper to each centroid
    from scipy.spatial.distance import cdist

    all_dists = cdist(X, centroid_matrix, metric="euclidean")

    # Bridge papers: close to 2+ centroids
    # For each paper, find the two closest centroids
    sorted_dists = np.sort(all_dists, axis=1)
    # Ratio of 2nd closest to closest — low ratio = bridge paper
    bridge_ratio = sorted_dists[:, 1] / (sorted_dists[:, 0] + 1e-10)

    bridge_idx = np.argsort(bridge_ratio)[:20]
    print("\n=== Top 20 Bridge Papers (close to 2+ clusters) ===")
    for idx in bridge_idx:
        closest = np.argsort(all_dists[idx])[:2]
        c1, c2 = centroid_ids[closest[0]], centroid_ids[closest[1]]
        print(
            f"  ratio={bridge_ratio[idx]:.3f}: clusters [{c1}, {c2}] "
            f"'{df['title'][int(idx)][:70]}'"
        )

    # Store bridge scores
    df_clustered = df_clustered.with_columns(
        pl.Series("bridge_ratio", bridge_ratio),
        pl.Series("dist_to_nearest_centroid", sorted_dists[:, 0]),
    )

    # ── PLOT 7: Bridge ratio distribution ────────────────────────────────
    # Left tail (ratio ≈ 1) = true bridge papers equidistant from two clusters.
    # Right tail = papers that clearly belong to one cluster.
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bridge_ratio, nbinsx=60, name="bridge ratio"))
    fig.add_vline(
        x=float(np.percentile(bridge_ratio, 5)),
        line_dash="dash",
        line_color="red",
        annotation_text="p5 (bridge candidates)",
    )
    fig.update_layout(
        title="Bridge Ratio Distribution (≈1.0 → equidistant from 2 clusters)",
        xaxis_title="dist_2nd_nearest / dist_nearest",
        yaxis_title="count",
        height=380,
    )
    fig.show()

# ── PLOT 6: Area composition per cluster ─────────────────────────────────
# The key EDA question: which research themes live in each cluster?
# Normalized stacked bar so small and large clusters are comparable.
_non_noise = df_clustered.filter(pl.col("cluster_hdbscan") != -1)
if _non_noise.shape[0] > 0:
    _area_comp = (
        _non_noise.group_by("cluster_hdbscan", "primary_area")
        .len()
        .with_columns(
            pl.col("cluster_hdbscan").cast(pl.String).alias("cluster"),
        )
    )
    # Normalise within each cluster to get fractional area breakdown
    _area_comp = _area_comp.with_columns(
        (pl.col("len") / pl.col("len").sum().over("cluster")).alias("fraction")
    )
    # Shorten long area labels for readability
    _area_comp = _area_comp.with_columns(
        pl.col("primary_area").str.slice(0, 40).alias("area_short")
    )
    fig = px.bar(
        _area_comp.to_pandas(),
        x="cluster",
        y="fraction",
        color="area_short",
        barmode="stack",
        title="Primary Area Composition per Cluster (normalised)",
        labels={"fraction": "fraction of cluster", "area_short": "area"},
    )
    fig.update_layout(height=500, xaxis_tickangle=-30, legend_title="area (truncated)")
    fig.show()

# ── Noise point analysis ────────────────────────────────────────────────
noise_mask = hdbscan_labels == -1
n_noise = int(noise_mask.sum())
if n_noise > 0:
    noise_df = df_clustered.filter(pl.Series(noise_mask))
    print(f"\n=== Noise Points: {n_noise} ({n_noise / len(df) * 100:.1f}%) ===")
    noise_areas = noise_df.group_by("primary_area").len().sort("len", descending=True)
    print("Top areas in noise:")
    print(noise_areas.head(5))
    print(
        f"Mean rating (noise): {noise_df['rating_mean'].mean():.2f} "
        f"vs overall: {df['rating_mean'].mean():.2f}"
    )

# ── Save ─────────────────────────────────────────────────────────────────
df_clustered.write_parquet("iclr_2026_clustered.parquet")
summary_df.write_parquet("cluster_summary.parquet")
print("\nSaved clustering results to iclr_2026_clustered.parquet")
print("Saved cluster summary to cluster_summary.parquet")
