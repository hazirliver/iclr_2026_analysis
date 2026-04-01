"""Embed unique keywords via vLLM-served Qwen3-Embedding-8B.

Run on GPU VM with vLLM serving:
  vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling --host 0.0.0.0 --port 8000 \
    --dtype auto --max-model-len 32768 --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 --enforce-eager \
    --data-parallel-size 8 --disable-log-requests

Then:
  uv run python embed-keywords-script.py
"""

import asyncio

import numpy as np
import polars as pl
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 64
MAX_CONCURRENT = 32

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
sem = asyncio.Semaphore(MAX_CONCURRENT)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    async with sem:
        resp = await client.embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


async def embed_all(keywords: list[str]) -> pl.DataFrame:
    batches = [keywords[i : i + BATCH_SIZE] for i in range(0, len(keywords), BATCH_SIZE)]

    tasks = [embed_batch(b) for b in batches]
    results = await tqdm.gather(*tasks, desc="Embedding keyword batches", total=len(batches))

    embeddings = [emb for batch_embs in results for emb in batch_embs]
    print(f"Done: {len(embeddings)} embeddings, dim={len(embeddings[0])}")

    emb_array = np.array(embeddings, dtype=np.float32)
    return pl.DataFrame(
        {
            "keyword": keywords,
            "embedding": [row.tolist() for row in emb_array],
        }
    )


if __name__ == "__main__":
    df = pl.read_parquet("iclr_2026_features.parquet")

    # Explode, normalize, deduplicate
    unique_kw = (
        df.select("keywords")
        .explode("keywords")
        .with_columns(pl.col("keywords").str.to_lowercase().str.strip_chars())
        .filter(pl.col("keywords") != "")
        .select("keywords")
        .unique()
        .sort("keywords")
        .to_series()
        .to_list()
    )
    print(f"Unique keywords: {len(unique_kw)}")

    result = asyncio.run(embed_all(unique_kw))
    result.write_parquet("keyword_embeddings.parquet")
    print("Saved to keyword_embeddings.parquet")
