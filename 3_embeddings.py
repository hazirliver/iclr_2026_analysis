"""Phase 3: Embeddings & Semantic Structure.

Exports texts for external GPU embedding, then (if embeddings exist)
runs PCA, UMAP, nearest neighbors, and duplicate detection.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap

EMBEDDINGS_FILE = Path("embeddings.npy")
OUTPUT_FILE = "iclr_2026_embeddings.parquet"

df = pl.read_parquet("iclr_2026_features.parquet")
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ── Export texts for external embedding ──────────────────────────────────
texts_df = df.select("openreview_id", "text_for_embedding")
texts_df.write_parquet("texts_for_embedding.parquet")
print(f"Saved {texts_df.shape[0]} texts to texts_for_embedding.parquet")
print("  Columns: openreview_id, text_for_embedding")

if not EMBEDDINGS_FILE.exists():
    print(f"\n{'=' * 60}")
    print("EMBEDDINGS NOT FOUND")
    print(f"{'=' * 60}")
    print(f"""
To compute embeddings:
  1. Copy texts_for_embedding.parquet to your GPU VM
  2. Run Qwen/Qwen3-Embedding-8B on the 'text_for_embedding' column
  3. Save the result as embeddings.npy (shape: [{df.shape[0]}, embed_dim])
     - Rows must align with texts_for_embedding.parquet row order
     - Normalize embeddings to unit length (L2 norm = 1)
  4. Copy embeddings.npy back here
  5. Re-run this script

Skipping semantic analysis for now.
""")
    # Pass through features as-is so downstream scripts can still run
    df.write_parquet(OUTPUT_FILE)
    print(f"Saved features (no embeddings) to {OUTPUT_FILE}")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════
# POST-EMBEDDING PROCESSING (runs only when embeddings.npy exists)
# ══════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

# ── Load and validate embeddings ─────────────────────────────────────────
embeddings = np.load(EMBEDDINGS_FILE)
assert embeddings.shape[0] == df.shape[0], (
    f"Row mismatch: embeddings {embeddings.shape[0]} vs data {df.shape[0]}"
)
print(f"\nLoaded embeddings: shape={embeddings.shape}")

# Check normalization
norms = np.linalg.norm(embeddings, axis=1)
print(
    f"Embedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
    f"min={norms.min():.4f}, max={norms.max():.4f}"
)

if norms.std() > 0.01:
    print("Normalizing embeddings to unit length")
    embeddings = embeddings / norms[:, np.newaxis]

# ── PCA for denoising ───────────────────────────────────────────────────
pca = PCA(n_components=0.9, random_state=SEED)
embeddings_pca = pca.fit_transform(embeddings)
print(
    f"\nPCA: {embeddings.shape[1]} → {embeddings_pca.shape[1]} components "
    f"(90% variance, explained={pca.explained_variance_ratio_.sum():.4f})"
)

# ── UMAP for visualization ──────────────────────────────────────────────
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
umap_coords = reducer.fit_transform(embeddings_pca)
print(f"UMAP: {embeddings_pca.shape[1]} → 2D")

# ── Nearest neighbors & local density ────────────────────────────────────
K = 10
nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
nn.fit(embeddings)
distances, indices = nn.kneighbors(embeddings)

# Exclude self (index 0), compute mean distance to K nearest
mean_knn_dist = distances[:, 1:].mean(axis=1)
print(f"\nLocal density (mean {K}-NN cosine distance):")
print(f"  mean={mean_knn_dist.mean():.4f}, std={mean_knn_dist.std():.4f}")
print(
    f"  min={mean_knn_dist.min():.4f} (densest), max={mean_knn_dist.max():.4f} (most isolated)"
)

# ── Near-duplicate detection ─────────────────────────────────────────────
# Pairs with cosine similarity > 0.95 (distance < 0.05)
DUPLICATE_THRESHOLD = 0.05
near_dupes = []
for i in range(len(distances)):
    for j_idx in range(1, K + 1):
        if distances[i, j_idx] < DUPLICATE_THRESHOLD:
            j = indices[i, j_idx]
            if i < j:  # avoid counting pairs twice
                near_dupes.append((i, j, 1 - distances[i, j_idx]))

print(f"\nNear-duplicate pairs (cosine sim > 0.95): {len(near_dupes)}")
if near_dupes:
    for i, j, sim in near_dupes[:10]:
        print(f"  sim={sim:.4f}: '{df['title'][i][:60]}...'")
        print(f"           vs '{df['title'][j][:60]}...'")

# ── Semantic outliers ────────────────────────────────────────────────────
outlier_idx = np.argsort(mean_knn_dist)[-20:][::-1]
print("\n=== Top 20 Semantic Outliers (most isolated) ===")
for idx in outlier_idx:
    print(f"  dist={mean_knn_dist[idx]:.4f}: {df['title'][idx][:80]}")

# ── Attach everything to dataframe ───────────────────────────────────────
# Store PCA and UMAP as separate columns (not list columns for easy use)
pca_cols = {f"pca_{i}": embeddings_pca[:, i] for i in range(embeddings_pca.shape[1])}
result = df.with_columns(
    pl.Series("umap_x", umap_coords[:, 0]),
    pl.Series("umap_y", umap_coords[:, 1]),
    pl.Series("mean_knn_distance", mean_knn_dist),
    *[pl.Series(name, values) for name, values in pca_cols.items()],
)

result.write_parquet(OUTPUT_FILE)
print(f"\nSaved {result.shape[0]} rows × {result.shape[1]} columns to {OUTPUT_FILE}")
print(f"  Includes: {embeddings_pca.shape[1]} PCA components, UMAP 2D, kNN distances")
