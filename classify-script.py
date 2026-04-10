"""Classify ICLR 2026 papers via vLLM-served Kimi-K2.5.

Run on GPU VM with vLLM serving:
  vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
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
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
MAX_CONCURRENT = 64
MAX_RETRIES = 3

INPUT_FILE = "iclr_2026_accepted.parquet"
OUTPUT_FILE = "classification_results_with_reasoning.parquet"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
sem = asyncio.Semaphore(MAX_CONCURRENT)

# ── categories & schema ─────────────────────────────────
CATEGORIES = [
    "SWE Agents",
    "Inference Optimisation",
    "Infrastructure",
    "AI for Life Sciences",
    "Robotics",
    "Other",
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
1. SWE Agents — coding agents, agentic coding, software engineering agents, SWE-agent, repository-level code generation, research agents that write or debug code
3. Inference Optimisation — test-time compute scaling, chain-of-thought reasoning, inference-time adaptation/algorithms, compute budget allocation, best-of-N, scaling test-time compute, visual/multimodal CoT
4. Infrastructure — distributed training, fully sharded data parallel (fsdp), GPU/CUDA/Triton kernels, memory-efficient training, gradient checkpointing/compression, sequence parallelism, federated learning, hardware acceleration, gpu-parallel optimization, offloading  
5. AI for Life Sciences & Healthcare — biology, medicine, drug discovery, genomics, healthcare applications, molecular modeling, protein folding, neuroscience, cognitive science
6. Robotics — physical robots, control systems, embodied AI, manipulation, navigation in the real world, autonomous driving, sim-to-real transfer
7. Other — papers that do not clearly fit any of the above 6 specific categories (e.g., safety, fairness, media generation, pure theory, training optimization, graph learning, representation learning, etc.)

Rules:
- Assign exactly ONE category.
- Prioritize the main contribution, not secondary applications.
- Do NOT classify based on buzzwords alone — understand the core research goal.
- If a paper applies a method to a domain (e.g., RL for robotics), classify based on the MAIN contribution:
  - If the novelty is in RL → RL
  - If the novelty is in robotics → Robotics
- Use "Other" only when the paper genuinely does not fit any of the 6 specific categories.
- If uncertain between a specific category and "Other", prefer "Other".
- Keep reasoning to 2 sentences maximum. Do NOT repeat the title or abstract.
- Reply strictly adhering to the JSON strict schema:
    ```
    {
  "type": "json_schema",
  "json_schema": {
    "name": "paper_classification",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "category": {
          "type": "string",
          "enum": [
            "SWE Agents",
            "Inference Optimisation",
            "Infrastructure",
            "AI for Life Sciences",
            "Robotics",
            "Other"
          ]
        },
        "confidence": {
          "type": "string",
          "enum": ["high", "medium", "low"]
        },
        "reasoning": {
          "type": "string"
        }
      },
      "required": ["category", "confidence", "reasoning"],
      "additionalProperties": false
    }
  }
}
    ```
"""

# One-shot example: agent paper that should NOT be classified as SWE Agents
# (the domain is cybersecurity, not software engineering)
ONESHOT_USER = """\
Title: Comparing AI Agents to Cybersecurity Professionals in Real-World Penetration Testing

Abstract: We present the first comprehensive evaluation of AI agents against human \
cybersecurity professionals in a live enterprise environment. We evaluate ten \
cybersecurity professionals alongside six existing AI agents and ARTEMIS, our new \
agent scaffold, on a large university network consisting of 8,000 hosts across 12 \
subnets. ARTEMIS is a multi-agent framework featuring dynamic prompt generation, \
arbitrary sub-agents, and automatic vulnerability triaging."""

ONESHOT_ASSISTANT = """\
{"category": "Other", "confidence": "high", "reasoning": "Despite using an agent \
framework, the core contribution is benchmarking AI in cybersecurity penetration \
testing — a security domain, not software engineering."}"""


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
                        {"role": "user", "content": ONESHOT_USER},
                        {"role": "assistant", "content": ONESHOT_ASSISTANT},
                        {"role": "user", "content": user_msg},
                    ],
                    extra_body={
                        "response_format": CLASSIFICATION_SCHEMA,
                    },
                    temperature=0.0,
                    max_tokens=12000,
                )
            content = resp.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Empty response from model")
            result = json.loads(content)
            if "category" not in result or result["category"] not in CATEGORIES:
                raise ValueError(f"Invalid category in response: {content[:200]}")
            result["openreview_id"] = openreview_id
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1**attempt)
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

    # resume: skip already-classified papers, retry UNCLASSIFIED/null
    try:
        prev = pl.read_parquet(OUTPUT_FILE)
        done = prev.filter(
            pl.col("llm_category").is_not_null()
            & pl.col("llm_category").is_in(CATEGORIES)
        )
        n_failed = prev.shape[0] - done.shape[0]
        done_ids = set(done["openreview_id"].to_list())
        remaining = df.filter(~pl.col("openreview_id").is_in(done_ids))
        print(
            f"Already classified: {len(done_ids)}, "
            f"previously failed (will retry): {n_failed}, "
            f"remaining: {len(remaining)}"
        )
    except FileNotFoundError:
        done = None
        remaining = df

    if len(remaining) == 0:
        print("All papers already classified.")
    else:
        new_results = asyncio.run(classify_all(remaining))

        # merge with previously done results (ensure matching column order)
        col_order = ["openreview_id", "llm_category", "llm_confidence", "llm_reasoning"]
        new_results = new_results.select(col_order)
        if done is not None:
            all_results = pl.concat([done.select(col_order), new_results])
        else:
            all_results = new_results

        all_results.write_parquet(OUTPUT_FILE)
        print(f"Saved {len(all_results)} results to {OUTPUT_FILE}")

        n_unclassified = all_results.filter(
            pl.col("llm_category") == "UNCLASSIFIED"
        ).shape[0]
        if n_unclassified > 0:
            print(f"WARNING: {n_unclassified} papers could not be classified")
