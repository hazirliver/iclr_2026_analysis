"""Classify ICLR 2026 papers via vLLM-served Kimi-K2.5.

Run on GPU VM with vLLM serving:
  vllm serve moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95 \
    --disable-log-requests \
    --port 8000

Then:
  uv run python classify-script.py
"""

import asyncio
import json

import polars as pl
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ── config ──────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "moonshotai/Kimi-K2.5"
MAX_CONCURRENT = 128
MAX_RETRIES = 3

INPUT_FILE = "iclr_2026_accepted.parquet"
OUTPUT_FILE = "classification_results.parquet"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
sem = asyncio.Semaphore(MAX_CONCURRENT)

# ── categories & schema ─────────────────────────────────
CATEGORIES = [
    "AI Agents",
    "RL",
    "Inference Optimisation",
    "Infrastructure",
    "AI Safety, Ethics and Societal Impact",
    "AI for Life Sciences",
    "Robotics",
    "Media",
]

CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reasoning": {"type": "string"},
            },
            "required": ["category", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """\
You are an expert research paper classifier.

Your task is to assign exactly ONE dominant category to a scientific paper based ONLY on its title and abstract.

You must choose the single most relevant category from the predefined list below. Even if multiple categories seem relevant, you must select the one that best represents the primary contribution or focus of the paper.

Categories:
1. AI Agents — autonomous systems, multi-agent systems, planning agents, tool-using agents, LLM agents, SWE agents
2. RL — reinforcement learning, policy optimization, reward modeling, environments, decision-making via RL, RLHF, DPO, post-training alignment
3. Inference Optimisation — model efficiency, inference speed, quantization, pruning, distillation, serving optimization, speculative decoding, long-context scaling, KV-cache, token prediction
4. Infrastructure — systems, distributed training, data pipelines, hardware/software stacks, ML platforms, compilers, scheduling
5. AI Safety, Ethics and Societal Impact — alignment, fairness, bias, interpretability, governance, risks, red-teaming, watermarking, privacy
6. AI for Life Sciences — biology, medicine, drug discovery, genomics, healthcare applications, molecular modeling, protein folding, scientific simulation
7. Robotics — physical robots, control systems, embodied AI, manipulation, navigation in the real world, autonomous driving
8. Media — generation or processing of text, images, video, audio, music, games, 3D, diffusion models, multimodal generation

Rules:
- Assign exactly ONE category.
- Prioritize the main contribution, not secondary applications.
- Do NOT classify based on buzzwords alone — understand the core research goal.
- If a paper applies a method to a domain (e.g., RL for robotics), classify based on the MAIN contribution:
  - If the novelty is in RL → RL
  - If the novelty is in robotics → Robotics
- If uncertain, choose the closest match based on the central problem being solved.
"""


# ── inference ───────────────────────────────────────────
async def classify_paper(openreview_id: str, title: str, abstract: str) -> dict:  # ty: ignore[invalid-return-type]
    user_msg = f"Title: {title}\n\nAbstract: {abstract}"

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    extra_body={"response_format": CLASSIFICATION_SCHEMA},
                    temperature=0.0,
                    max_tokens=1024,
                )
            content = resp.choices[0].message.content or "{}"
            result = json.loads(content)
            result["openreview_id"] = openreview_id
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                return {
                    "openreview_id": openreview_id,
                    "category": "UNCLASSIFIED",
                    "confidence": "low",
                    "reasoning": f"Error after {MAX_RETRIES} retries: {e}",
                }


async def classify_all(df: pl.DataFrame) -> pl.DataFrame:
    rows = df.select("openreview_id", "title", "abstract").to_dicts()

    tasks = [
        classify_paper(r["openreview_id"], r["title"], r["abstract"]) for r in rows
    ]
    results = await tqdm.gather(*tasks, desc="Classifying papers", total=len(tasks))

    return pl.DataFrame(results).rename(
        {
            "category": "llm_category",
            "confidence": "llm_confidence",
            "reasoning": "llm_reasoning",
        }
    )


# ── main ────────────────────────────────────────────────
if __name__ == "__main__":
    df = pl.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} papers from {INPUT_FILE}")

    # resume: skip already-classified papers, retry UNCLASSIFIED
    try:
        prev = pl.read_parquet(OUTPUT_FILE)
        done = prev.filter(pl.col("llm_category") != "UNCLASSIFIED")
        failed = prev.filter(pl.col("llm_category") == "UNCLASSIFIED")
        done_ids = set(done["openreview_id"].to_list())
        remaining = df.filter(~pl.col("openreview_id").is_in(done_ids))
        print(
            f"Already classified: {len(done_ids)}, "
            f"previously failed (will retry): {failed.shape[0]}, "
            f"remaining: {len(remaining)}"
        )
    except FileNotFoundError:
        done = None
        remaining = df

    if len(remaining) == 0:
        print("All papers already classified.")
    else:
        new_results = asyncio.run(classify_all(remaining))

        # merge with previously done results
        if done is not None:
            all_results = pl.concat([done, new_results])
        else:
            all_results = new_results

        all_results.write_parquet(OUTPUT_FILE)
        print(f"Saved {len(all_results)} results to {OUTPUT_FILE}")

        n_unclassified = all_results.filter(
            pl.col("llm_category") == "UNCLASSIFIED"
        ).shape[0]
        if n_unclassified > 0:
            print(f"WARNING: {n_unclassified} papers could not be classified")
