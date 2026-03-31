"""Phase 4: Clustering & Theme Discovery.

HDBSCAN (primary) + KMeans (baseline) on embedding/PCA space if available,
otherwise falls back to keyword TF-IDF + numeric features.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

SEED = 42
np.random.seed(SEED)
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────
INPUT_FILE = "iclr_2026_embeddings.parquet"
if not Path(INPUT_FILE).exists():
    INPUT_FILE = "iclr_2026_features.parquet"

df = pl.read_parquet(INPUT_FILE)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {INPUT_FILE}")

# ── Determine if we have embeddings ──────────────────────────────────────
pca_cols = [c for c in df.columns if c.startswith("pca_")]
has_embeddings = len(pca_cols) > 0
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

    print(
        f"  min_cluster_size={min_size:3d}: {n_clusters:3d} clusters, "
        f"noise={noise_frac:.1%}, silhouette={score:.3f}"
    )

    if score > best_score:
        best_score = score
        best_min_size = min_size
        best_labels = labels

print(f"\nBest HDBSCAN: min_cluster_size={best_min_size}, silhouette={best_score:.3f}")
assert best_labels is not None, "HDBSCAN produced no valid clustering"
hdbscan_labels = best_labels
n_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

# ══════════════════════════════════════════════════════════════════════════
# KMeans (baseline comparison)
# ══════════════════════════════════════════════════════════════════════════

print("\n=== KMeans Sweep ===")
best_k_score = -1
best_k = None

for k in [10, 20, 30, 40, 50]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(X)
    score = silhouette_score(X, km_labels)
    print(f"  k={k:3d}: silhouette={score:.3f}")
    if score > best_k_score:
        best_k_score = score
        best_k = k

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
kmeans_labels = km_final.fit_predict(X)
kmeans_centroids = km_final.cluster_centers_
print(f"\nBest KMeans: k={best_k}, silhouette={best_k_score:.3f}")

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
