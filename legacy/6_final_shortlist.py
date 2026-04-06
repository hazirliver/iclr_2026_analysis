"""Phase 6: Final Shortlist.

Produces diversified reading lists using rank-then-diversify.
Enforces diversity: no more than N papers from any single cluster, area, or status.
"""

import polars as pl

df = pl.read_parquet("../iclr_2026_scored.parquet")
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

has_clusters = "cluster_ward" in df.columns
has_semantic = "score_semantic_novel" in df.columns
has_bridge = "score_bridge" in df.columns

DISPLAY_COLS = [
    "title",
    "openreview_id",
    "status",
    "primary_area",
    "rating_mean",
    "soundness_mean",
    "contribution_mean",
    "site",
]
if has_clusters:
    DISPLAY_COLS.append("cluster_ward")


def format_paper(row: dict, reason: str) -> str:
    """Format a paper for display."""
    cluster = f" cluster={row.get('cluster_ward', '?')}" if has_clusters else ""
    return (
        f"  [{row['status']:8s}] rating={row['rating_mean']:.1f} "
        f"sound={row['soundness_mean']:.1f} contrib={row['contribution_mean']:.1f}"
        f"{cluster}\n"
        f"    {row['title'][:80]}\n"
        f"    area={row['primary_area'][:50]}\n"
        f"    {row['site']}\n"
        f"    → {reason}"
    )


def diversified_top_n(
    scored_df: pl.DataFrame,
    score_col: str,
    n: int,
    max_per_area: int = 3,
    max_per_cluster: int | None = 3,
) -> pl.DataFrame:
    """Select top-N papers with diversity constraints."""
    ranked = scored_df.sort(score_col, descending=True)
    selected = []
    area_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}

    for row in ranked.iter_rows(named=True):
        if len(selected) >= n:
            break

        area = row["primary_area"]
        cluster = row.get("cluster_ward")

        if area_counts.get(area, 0) >= max_per_area:
            continue
        if (
            max_per_cluster
            and cluster is not None
            and cluster_counts.get(cluster, 0) >= max_per_cluster
        ):
            continue

        selected.append(row)
        area_counts[area] = area_counts.get(area, 0) + 1
        if cluster is not None:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    return pl.DataFrame(selected)


def print_list(title: str, papers: pl.DataFrame, score_col: str):
    """Print a shortlist."""
    print(f"\n{'=' * 70}")
    print(f"{title} ({papers.shape[0]} papers)")
    print(f"{'=' * 70}")
    for i, row in enumerate(papers.iter_rows(named=True), 1):
        print(f"\n{i}. {format_paper(row, f'{score_col}={row[score_col]:.3f}')}")


# ══════════════════════════════════════════════════════════════════════════
# SHORTLISTS
# ══════════════════════════════════════════════════════════════════════════

# 1. Top 20 overall
top_overall = diversified_top_n(df, "score_top_overall", 20)
print_list("TOP 20 OVERALL", top_overall, "score_top_overall")

# 2. Top 15 hidden gems (Poster only)
poster_df = df.filter(pl.col("status").str.contains("Poster"))
hidden_gems = diversified_top_n(poster_df, "score_hidden_gem", 15)
print_list("TOP 15 HIDDEN GEMS", hidden_gems, "score_hidden_gem")

# 3. Top 15 controversial
controversial = diversified_top_n(df, "score_controversial", 15, max_per_area=4)
print_list("TOP 15 CONTROVERSIAL", controversial, "score_controversial")

# 4. Top 15 semantically novel (if available)
if has_semantic:
    semantic = diversified_top_n(df, "score_semantic_novel", 15)
    print_list("TOP 15 SEMANTICALLY NOVEL", semantic, "score_semantic_novel")

# 5. Top 3-5 per major cluster
if has_clusters:
    cluster_col = "cluster_ward"
    clusters = df.group_by(cluster_col).len().sort("len", descending=True)
    major_clusters = clusters.filter(pl.col("len") >= 50)[cluster_col].to_list()

    print(f"\n{'=' * 70}")
    print(
        f"TOP PAPERS PER MAJOR CLUSTER ({len(major_clusters)} clusters with 50+ papers)"
    )
    print(f"{'=' * 70}")
    for cid in major_clusters:
        cluster_df = df.filter(pl.col(cluster_col) == cid)
        top = cluster_df.sort("score_top_overall", descending=True).head(5)
        print(f"\n--- Cluster {cid} (n={cluster_df.shape[0]}) ---")
        for row in top.iter_rows(named=True):
            print(f"  rating={row['rating_mean']:.1f}  {row['title'][:70]}")
            print(f"    {row['site']}")

# 6. Top 3-5 per primary area
areas = df.group_by("primary_area").len().sort("len", descending=True)

print(f"\n{'=' * 70}")
print(f"TOP PAPERS PER PRIMARY AREA ({areas.shape[0]} areas)")
print(f"{'=' * 70}")
for row in areas.iter_rows(named=True):
    area = row["primary_area"]
    area_df = df.filter(pl.col("primary_area") == area)
    n_show = 5 if row["len"] >= 50 else 3
    top = area_df.sort("score_area_leader", descending=True).head(n_show)
    print(f"\n--- {area} (n={row['len']}) ---")
    for paper in top.iter_rows(named=True):
        print(f"  rating={paper['rating_mean']:.1f}  {paper['title'][:70]}")
        print(f"    {paper['site']}")

# ══════════════════════════════════════════════════════════════════════════
# 7. FINAL DIVERSIFIED "START HERE" LIST
# Combine top papers from each archetype, then deduplicate + diversify
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("FINAL DIVERSIFIED 'START HERE' LIST")
print(f"{'=' * 70}")

# Collect candidates from each archetype with their reason
candidates: list[tuple[str, str, float]] = []  # (openreview_id, reason, priority)

for row in df.sort("score_top_overall", descending=True).head(30).iter_rows(named=True):
    candidates.append((row["openreview_id"], "top overall", row["score_top_overall"]))

for row in (
    poster_df.sort("score_hidden_gem", descending=True).head(20).iter_rows(named=True)
):
    candidates.append((row["openreview_id"], "hidden gem", row["score_hidden_gem"]))

for row in (
    df.sort("score_controversial", descending=True).head(15).iter_rows(named=True)
):
    candidates.append(
        (row["openreview_id"], "controversial", row["score_controversial"])
    )

for row in df.sort("score_consensus", descending=True).head(15).iter_rows(named=True):
    candidates.append(
        (row["openreview_id"], "consensus standout", row["score_consensus"])
    )

for row in (
    df.sort("score_high_engagement", descending=True).head(15).iter_rows(named=True)
):
    candidates.append(
        (row["openreview_id"], "high engagement", row["score_high_engagement"])
    )

if has_semantic:
    for row in (
        df.sort("score_semantic_novel", descending=True).head(15).iter_rows(named=True)
    ):
        candidates.append(
            (row["openreview_id"], "semantically novel", row["score_semantic_novel"])
        )

if has_bridge:
    for row in df.sort("score_bridge", descending=True).head(15).iter_rows(named=True):
        candidates.append((row["openreview_id"], "bridge paper", row["score_bridge"]))

# Deduplicate: keep first occurrence (highest priority archetype)
seen = set()
unique_candidates = []
for oid, reason, score in candidates:
    if oid not in seen:
        seen.add(oid)
        unique_candidates.append((oid, reason))

# Apply diversity constraints
final_list = []
area_counts: dict[str, int] = {}
cluster_counts: dict[int, int] = {}
status_counts: dict[str, int] = {}
# Adapt cluster constraint to number of clusters available
if has_clusters:
    n_clusters = df["cluster_ward"].n_unique()
else:
    n_clusters = 0

MAX_PER_AREA = 5
# Relax cluster cap when few clusters exist (fallback mode)
MAX_PER_CLUSTER = max(10, 30 // max(n_clusters, 1)) if n_clusters > 0 else None
MAX_PER_STATUS = 25
TARGET = 30

for oid, reason in unique_candidates:
    if len(final_list) >= TARGET:
        break

    paper = df.filter(pl.col("openreview_id") == oid).row(0, named=True)
    area = paper["primary_area"]
    status = paper["status"]
    cluster = paper.get("cluster_ward")

    if area_counts.get(area, 0) >= MAX_PER_AREA:
        continue
    if status_counts.get(status, 0) >= MAX_PER_STATUS:
        continue
    if (
        MAX_PER_CLUSTER is not None
        and cluster is not None
        and cluster_counts.get(cluster, 0) >= MAX_PER_CLUSTER
    ):
        continue

    final_list.append((paper, reason))
    area_counts[area] = area_counts.get(area, 0) + 1
    status_counts[status] = status_counts.get(status, 0) + 1
    if cluster is not None:
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

print(f"\nSelected {len(final_list)} papers from {len(unique_candidates)} candidates\n")
for i, (paper, reason) in enumerate(final_list, 1):
    print(f"{i:2d}. {format_paper(paper, reason)}\n")

# ── Summary stats ────────────────────────────────────────────────────────
final_areas = [p["primary_area"] for p, _ in final_list]
final_statuses = [p["status"] for p, _ in final_list]
print("\nDiversity check:")
print(f"  Unique areas: {len(set(final_areas))}")
print(f"  Unique statuses: {len(set(final_statuses))}")
if has_clusters:
    final_clusters = [p.get("cluster_ward") for p, _ in final_list]
    print(f"  Unique clusters: {len(set(c for c in final_clusters if c is not None))}")
