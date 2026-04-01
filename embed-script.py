import asyncio
import numpy as np
import polars as pl
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 64  # texts per API call
MAX_CONCURRENT = 32  # parallel in-flight requests (8 GPUs can handle more)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
sem = asyncio.Semaphore(MAX_CONCURRENT)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    async with sem:
        resp = await client.embeddings.create(model=MODEL, input=texts)
    # sort by index to preserve order
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


async def embed_all(df: pl.DataFrame) -> pl.DataFrame:
    texts = df["text_for_embedding"].to_list()
    ids = df["openreview_id"].to_list()

    # chunk into batches
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    # fire all batches concurrently (bounded by semaphore)
    tasks = [embed_batch(b) for b in batches]
    results = await tqdm.gather(*tasks, desc="Embedding batches", total=len(batches))

    # flatten
    embeddings = [emb for batch_embs in results for emb in batch_embs]

    print(f"Done: {len(embeddings)} embeddings, dim={len(embeddings[0])}")

    # build result dataframe
    emb_array = np.array(embeddings, dtype=np.float32)
    result = pl.DataFrame({
        "openreview_id": ids,
        "embedding": [row.tolist() for row in emb_array],
    })
    return result


# ── usage ────────────────────────────────────────────────
if __name__ == "__main__":
    # load your dataframe (adjust path)
    df = pl.read_parquet("iclr_2026_features.parquet")

    print(f"Embedding {len(df)} rows...")
    result = asyncio.run(embed_all(df))

    # join back or save standalone
    result.write_parquet("embeddings.parquet")
    print("Saved to embeddings.parquet")
