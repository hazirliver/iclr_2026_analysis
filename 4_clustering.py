"""Phase 4: Clustering & Theme Discovery.

Ward hierarchical clustering (primary, no noise points) + KMeans (baseline)
on embedding/PCA space if available, otherwise falls back to keyword
TF-IDF + numeric features.

Why Ward over HDBSCAN: academic paper embeddings form a continuous manifold
rather than tight density cores, so HDBSCAN assigns most points as noise.
Ward's minimum-variance criterion handles this geometry better and always
produces a full partition.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward as ward_linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
    f"Embedding PCA columns found: {len(pca_cols)} → "
    f"{'full mode' if has_embeddings else 'fallback mode'}"
)

# ── Build clustering feature matrix ──────────────────────────────────────
if has_embeddings:
    X = df.select(pca_cols).to_numpy()
    print(f"Clustering on PCA space: {X.shape}")
else:
    # Fallback: TF-IDF on keywords + scaled numeric features
    print("Building fallback feature matrix (TF-IDF keywords + numeric)")

    keyword_texts = df.select(pl.col("keywords").list.join(" ")).to_series().to_list()
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(keyword_texts).toarray()

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
        f"Fallback feature matrix: {X.shape} "
        f"(TF-IDF: {tfidf_matrix.shape[1]}, numeric: {numeric_scaled.shape[1]})"
    )

# ══════════════════════════════════════════════════════════════════════════
# WARD DENDROGRAM (top merges only)
# ══════════════════════════════════════════════════════════════════════════

# Compute full Ward linkage once — reused for both the dendrogram plot and
# picking k. For n=5358 × ~300 PCA dims this takes ~30-60s.
print("\nComputing Ward linkage matrix (may take ~30–60s on PCA embeddings)...")
Z = ward_linkage(X)
print(f"Linkage matrix shape: {Z.shape}")

# ── PLOT 1: Dendrogram (last 60 merges) ──────────────────────────────────
# The y-axis is the Ward distance (inertia increase) at each merge.
# Large vertical jumps = natural cluster boundaries. Use this to pick k.
ddata = dendrogram(Z, truncate_mode="lastp", p=60, no_plot=True)
fig = go.Figure()
for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="steelblue", width=1),
            showlegend=False,
        )
    )
fig.update_layout(
    title="Ward Dendrogram — last 60 merges (look for large vertical jumps to pick k)",
    xaxis_title="sample index (or cluster size in parentheses)",
    yaxis_title="Ward distance (merge cost)",
    height=450,
    width=1000,
)
fig.show()

# ══════════════════════════════════════════════════════════════════════════
# WARD SWEEP
# ══════════════════════════════════════════════════════════════════════════

# Sweep a wide range — KMeans silhouette was still climbing at k=50,
# so Ward likely benefits from similar or higher granularity.
WARD_K_RANGE = [20, 30, 40, 50, 75, 100]

print("\n=== Ward Sweep ===")
best_ward_score = -1
best_ward_k = None
best_ward_labels = None
ward_sweep: list[dict] = []

for k in WARD_K_RANGE:
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = ward.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=2000, random_state=SEED)
    ward_sweep.append({"k": k, "silhouette": score})
    print(f"  k={k:3d}: silhouette={score:.4f}")
    if score > best_ward_score:
        best_ward_score = score
        best_ward_k = k
        best_ward_labels = labels

print(f"\nBest Ward: k={best_ward_k}, silhouette={best_ward_score:.4f}")
assert best_ward_labels is not None, "Ward sweep produced no valid clustering"
ward_labels = best_ward_labels

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
    score = silhouette_score(X, km_labels, sample_size=2000, random_state=SEED)
    kmeans_sweep.append({"k": k, "silhouette": score})
    print(f"  k={k:3d}: silhouette={score:.4f}")
    if score > best_k_score:
        best_k_score = score
        best_k = k

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
kmeans_labels = km_final.fit_predict(X)
print(f"\nBest KMeans: k={best_k}, silhouette={best_k_score:.4f}")

# ── PLOT 2: Ward vs KMeans silhouette sweep ───────────────────────────────
# Both on the same x-axis (k) so the curves are directly comparable.
# Ward has no noise points — any silhouette advantage is genuine.
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[d["k"] for d in ward_sweep],
        y=[d["silhouette"] for d in ward_sweep],
        mode="lines+markers",
        name="Ward",
        line=dict(color="darkorange"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[d["k"] for d in kmeans_sweep],
        y=[d["silhouette"] for d in kmeans_sweep],
        mode="lines+markers",
        name="KMeans",
        line=dict(color="royalblue"),
    )
)
if best_ward_k is not None:
    fig.add_vline(
        x=best_ward_k,
        line_dash="dash",
        line_color="darkorange",
        annotation_text=f"Ward best k={best_ward_k}",
    )
fig.update_layout(
    title="Silhouette Score: Ward vs KMeans",
    xaxis_title="k (number of clusters)",
    yaxis_title="silhouette score (sample_size=2000)",
    height=400,
)
fig.show()

# ══════════════════════════════════════════════════════════════════════════
# CLUSTER LABELING
# ══════════════════════════════════════════════════════════════════════════

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

# ── PLOT 3: Cluster size bar chart ────────────────────────────────────────
# Color = mean rating. With k=50+ there will be many bars — log scale helps.
_summary_sorted = summary_df.sort("size", descending=True).to_pandas()
fig = px.bar(
    _summary_sorted,
    x="label",
    y="size",
    color="mean_rating",
    color_continuous_scale="RdYlGn",
    title=f"Ward Cluster Sizes (k={n_ward}, color = mean rating)",
)
fig.update_layout(
    xaxis_tickangle=-60,
    height=450,
    coloraxis_colorbar_title="mean rating",
    xaxis={"tickfont": {"size": 9}},
)
fig.show()

# ── PLOT 4: UMAP colored by Ward cluster ─────────────────────────────────
if has_umap:
    _umap_df = (
        df_clustered.select(
            "umap_x",
            "umap_y",
            "cluster_ward",
            "title",
            "rating_mean",
            "primary_area",
        )
        .with_columns(pl.col("cluster_ward").cast(pl.String).alias("cluster_label"))
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
        title=f"UMAP: Colored by Ward Cluster (k={n_ward})",
    )
    fig.update_layout(height=650, width=900)
    fig.show()
else:
    print("⚠ UMAP plot skipped (no embeddings)")

# ── PLOT 5: Rating distributions per cluster ─────────────────────────────
# Sort clusters by median rating to see the quality gradient across themes.
_rating_df = df_clustered.select(
    pl.col("cluster_ward").cast(pl.String).alias("cluster"),
    "rating_mean",
).to_pandas()
_median_ser = _rating_df.groupby("cluster")["rating_mean"].median()
_order_by_median: list[str] = sorted(
    _median_ser.index.tolist(), key=lambda c: -float(_median_ser[c])
)
fig = px.box(
    _rating_df,
    x="cluster",
    y="rating_mean",
    category_orders={"cluster": _order_by_median},
    title="Rating Distribution per Cluster (sorted by median rating)",
    points=False,  # suppress individual points — too many with k=50+
)
fig.update_layout(
    xaxis_tickangle=-60,
    height=450,
    xaxis={"tickfont": {"size": 8}},
)
fig.show()

# ══════════════════════════════════════════════════════════════════════════
# CROSS-CLUSTER STRUCTURE
# ══════════════════════════════════════════════════════════════════════════

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
        f"  ratio={bridge_ratio[idx]:.3f}: clusters [{c1}, {c2}] "
        f"'{df['title'][int(idx)][:70]}'"
    )

df_clustered = df_clustered.with_columns(
    pl.Series("bridge_ratio", bridge_ratio),
    pl.Series("dist_to_nearest_centroid", sorted_dists[:, 0]),
)

# ── PLOT 6: Area composition per cluster ─────────────────────────────────
# Which research themes live in each cluster? Sort clusters by dominant area
# so thematically related clusters appear adjacent.
_area_comp = (
    df_clustered.group_by("cluster_ward", "primary_area")
    .len()
    .with_columns(
        pl.col("cluster_ward").cast(pl.String).alias("cluster"),
        (pl.col("len") / pl.col("len").sum().over("cluster_ward")).alias("fraction"),
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

fig = px.bar(
    _area_comp.to_pandas(),
    x="cluster",
    y="fraction",
    color="area_short",
    barmode="stack",
    category_orders={"cluster": _cluster_order},
    title=f"Primary Area Composition per Ward Cluster (k={n_ward}, sorted by dominant area)",
    labels={"fraction": "fraction of cluster", "area_short": "area"},
)
fig.update_layout(
    height=550,
    xaxis_tickangle=-60,
    legend_title="area",
    xaxis={"tickfont": {"size": 8}},
)
fig.show()

# ── PLOT 7: Bridge ratio distribution ────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════

df_clustered.write_parquet("iclr_2026_clustered.parquet")
summary_df.write_parquet("cluster_summary.parquet")
print(
    f"\nSaved {df_clustered.shape[0]} rows × {df_clustered.shape[1]} columns "
    f"to iclr_2026_clustered.parquet"
)
print(f"Saved cluster summary ({n_ward} clusters) to cluster_summary.parquet")
