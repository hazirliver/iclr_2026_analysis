"""Phase 3: Embeddings & Semantic Structure.

Loads embeddings.parquet produced by the vllm/Qwen3-Embedding-8B inference
script (columns: openreview_id str, embedding list[f32]). Joins with
features on openreview_id, then runs PCA, UMAP, kNN density analysis,
and near-duplicate detection.

Inline progress-check plots use fig.show() (opens in browser).
Artifacts saved to downstream scripts via iclr_2026_embeddings.parquet.
"""

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap

EMBEDDINGS_FILE = Path("../embeddings.parquet")
OUTPUT_FILE = "../iclr_2026_embeddings.parquet"
SEED = 42
np.random.seed(SEED)

features = pl.read_parquet("../iclr_2026_features.parquet")
print(f"Loaded features: {features.shape[0]} rows × {features.shape[1]} columns")

# ── Export texts (idempotent — safe to re-run) ────────────────────────────
texts_df = features.select("openreview_id", "text_for_embedding")
texts_df.write_parquet("texts_for_embedding.parquet")
print(f"Saved {texts_df.shape[0]} texts to texts_for_embedding.parquet")

if not EMBEDDINGS_FILE.exists():
    print(f"\n{'=' * 60}")
    print("EMBEDDINGS NOT FOUND — run inference first")
    print(f"{'=' * 60}")
    print(f"""
Expected: embeddings.parquet with schema
  openreview_id: str
  embedding:     list[f32]  (shape: [{features.shape[0]}, embed_dim])

Reference inference script uses:
  vllm serve Qwen/Qwen3-Embedding-8B + AsyncOpenAI client
  result.write_parquet("embeddings.parquet")

Skipping semantic analysis — passing features through as-is.
""")
    features.write_parquet(OUTPUT_FILE)
    print(f"Saved features (no embeddings) to {OUTPUT_FILE}")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════
# LOAD & VALIDATE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════

emb_df = pl.read_parquet(EMBEDDINGS_FILE)
print(f"\nLoaded embeddings: {emb_df.shape[0]} rows, columns={emb_df.columns}")
assert "openreview_id" in emb_df.columns, "embeddings.parquet must have openreview_id"
assert "embedding" in emb_df.columns, "embeddings.parquet must have embedding"

# Validate coverage before joining
missing = features.filter(
    ~pl.col("openreview_id").is_in(emb_df["openreview_id"])
).shape[0]
print(f"Papers without embedding: {missing}")
assert missing == 0, f"{missing} papers have no embedding — recheck inference output"

# Join embeddings onto features (preserves feature row order)
df = features.join(
    emb_df.select("openreview_id", "embedding"), on="openreview_id", how="left"
)
print(f"After join: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Extract numpy array ──────────────────────────────────────────────────
# list[f32] column → (N, D) float32 array
embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
print(f"Embedding matrix: shape={embeddings.shape}, dtype={embeddings.dtype}")

# ── Check normalization ──────────────────────────────────────────────────
norms = np.linalg.norm(embeddings, axis=1)
print(
    f"Norms: mean={norms.mean():.4f}, std={norms.std():.6f}, "
    f"min={norms.min():.4f}, max={norms.max():.4f}"
)

# ── PLOT 1: Embedding norm distribution ──────────────────────────────────
# Quick sanity check: should be near-constant at 1.0 if already L2-normalized
fig = go.Figure()
fig.add_trace(go.Histogram(x=norms, nbinsx=50, name="L2 norm"))
fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="1.0")
fig.update_layout(
    title="Embedding L2 Norm Distribution (should peak at 1 if pre-normalized)",
    xaxis_title="L2 norm",
    yaxis_title="count",
    height=400,
)
fig.show()

if norms.std() > 0.01:
    print("Norms vary — normalizing embeddings to unit length")
    embeddings = embeddings / norms[:, np.newaxis]
else:
    print("Embeddings already unit-normalized ✓")

# ══════════════════════════════════════════════════════════════════════════
# PCA FOR DENOISING
# ══════════════════════════════════════════════════════════════════════════

# First fit a full PCA to inspect the scree plot, then keep 90% variance
pca_full = PCA(n_components=min(200, embeddings.shape[1]), random_state=SEED)
pca_full.fit(embeddings)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.searchsorted(cumvar, 0.90)) + 1
n95 = int(np.searchsorted(cumvar, 0.95)) + 1
n99 = int(np.searchsorted(cumvar, 0.99)) + 1
print(f"\nPCA variance thresholds: 90%→{n90} components, 95%→{n95}, 99%→{n99}")

# ── PLOT 2: PCA scree / cumulative variance ───────────────────────────────
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=cumvar[:100],
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
    fig.add_hline(y=thresh, line_dash="dash", line_color=color, annotation_text=label)
fig.update_layout(
    title="PCA Cumulative Explained Variance (first 100 components)",
    xaxis_title="n_components",
    yaxis_title="cumulative variance",
    height=400,
)
fig.show()

# Fit final PCA at 90% threshold
pca = PCA(n_components=0.90, random_state=SEED)
embeddings_pca = pca.fit_transform(embeddings)
print(
    f"Final PCA: {embeddings.shape[1]}D → {embeddings_pca.shape[1]}D "
    f"({pca.explained_variance_ratio_.sum():.4f} variance)"
)

# ══════════════════════════════════════════════════════════════════════════
# UMAP FOR VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

# Cluster in PCA space; UMAP is only for visual inspection
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
umap_coords = reducer.fit_transform(embeddings_pca)
print(f"UMAP: {embeddings_pca.shape[1]}D → 2D")

# ── PLOT 3: UMAP colored by primary area ─────────────────────────────────
# Quick check: do semantically related areas cluster together?
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
fig = px.scatter(
    plot_df.to_pandas(),
    x="umap_x",
    y="umap_y",
    color="primary_area",
    hover_name="title",
    hover_data=["rating_mean", "status"],
    opacity=0.5,
    title="UMAP: Colored by Primary Area (sanity check)",
)
fig.update_layout(height=700, width=1000)
fig.show()

# ── PLOT 4: UMAP colored by rating mean ──────────────────────────────────
fig = px.scatter(
    plot_df.to_pandas(),
    x="umap_x",
    y="umap_y",
    color="rating_mean",
    hover_name="title",
    color_continuous_scale="RdYlGn",
    opacity=0.6,
    title="UMAP: Colored by Rating Mean",
)
fig.update_layout(height=600, width=800)
fig.show()

# ══════════════════════════════════════════════════════════════════════════
# NEAREST NEIGHBORS & LOCAL DENSITY
# ══════════════════════════════════════════════════════════════════════════

K = 10
print(f"\nFitting {K}-NN index on normalized embeddings...")
nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
nn.fit(embeddings)
distances, indices = nn.kneighbors(embeddings)

# Exclude self (index 0), use cosine distances to k nearest neighbors
mean_knn_dist = distances[:, 1:].mean(axis=1)
print(f"Local density (mean {K}-NN cosine dist):")
print(f"  mean={mean_knn_dist.mean():.4f}, std={mean_knn_dist.std():.4f}")
print(
    f"  min={mean_knn_dist.min():.4f} (densest), max={mean_knn_dist.max():.4f} (most isolated)"
)

# ── PLOT 5: kNN distance distribution ────────────────────────────────────
# Right tail = semantic outliers; left = papers in crowded topics
fig = go.Figure()
fig.add_trace(go.Histogram(x=mean_knn_dist, nbinsx=60, name="mean kNN cosine dist"))
fig.update_layout(
    title=f"Local Density Distribution (mean {K}-NN cosine distance)",
    xaxis_title="mean kNN distance (higher = more isolated)",
    yaxis_title="count",
    height=400,
)
fig.show()

# ── Semantic outliers ────────────────────────────────────────────────────
outlier_idx = np.argsort(mean_knn_dist)[-20:][::-1]
print("\n=== Top 20 Semantic Outliers (most isolated) ===")
for idx in outlier_idx:
    print(f"  dist={mean_knn_dist[idx]:.4f}: {df['title'][idx][:80]}")

# ══════════════════════════════════════════════════════════════════════════
# NEAR-DUPLICATE DETECTION
# ══════════════════════════════════════════════════════════════════════════

# Pairs with cosine similarity > 0.95 (cosine distance < 0.05)
DUPLICATE_THRESHOLD = 0.05
near_dupes = []
for i in range(len(distances)):
    for j_idx in range(1, K + 1):
        if distances[i, j_idx] < DUPLICATE_THRESHOLD:
            j = int(indices[i, j_idx])
            if i < j:  # avoid counting pairs twice
                near_dupes.append((i, j, float(1 - distances[i, j_idx])))

print(f"\nNear-duplicate pairs (cosine sim > 0.95): {len(near_dupes)}")
if near_dupes:
    # Show up to 10 most similar pairs
    near_dupes.sort(key=lambda x: -x[2])
    for i, j, sim in near_dupes[:10]:
        print(f"  sim={sim:.4f}: '{df['title'][i][:65]}'")
        print(f"           vs: '{df['title'][j][:65]}'")

# ══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════

# Store: UMAP coords, kNN density, all PCA components
# Drop the raw `embedding` list column — it's large and downstream scripts
# use PCA components for clustering
pca_series = [
    pl.Series(f"pca_{i}", embeddings_pca[:, i]) for i in range(embeddings_pca.shape[1])
]
result = df.drop("embedding").with_columns(  # raw vectors not needed downstream
    pl.Series("umap_x", umap_coords[:, 0]),
    pl.Series("umap_y", umap_coords[:, 1]),
    pl.Series("mean_knn_distance", mean_knn_dist),
    *pca_series,
)

result.write_parquet(OUTPUT_FILE)
print(f"\nSaved {result.shape[0]} rows × {result.shape[1]} columns to {OUTPUT_FILE}")
print(
    f"  New columns: umap_x, umap_y, mean_knn_distance, "
    f"pca_0…pca_{embeddings_pca.shape[1] - 1}"
)
