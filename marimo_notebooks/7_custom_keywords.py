import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path

    SEED = 42
    FIGURES = Path("figures")
    FIGURES.mkdir(exist_ok=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Custom Category Analysis of ICLR 2026

    We manually define 8 thematic categories that reflect our research interests:
    Agents/SWE, RL/Post-training, Inference Scaling, Infrastructure,
    Safety/Governance, AI for Science, Robotics, and Media/Video/Games.
    Each category is specified by a curated list of raw author keywords.
    A paper belongs to a category if any of its keywords match.

    The goal: find the best papers in each category, understand how categories
    relate to each other, and discover truly interdisciplinary work at their
    intersections.

    # 1. Define selected keywords
    """)
    return


@app.cell(hide_code=True)
def _():
    # List 1 – Agents, SWE & Software Engineering
    kw_1 = [
        "agent",
        "agent benchmark",
        "agent composition",
        "agent design",
        "agent discovery",
        "agent diversity",
        "agent ensemble learning",
        "agent evaluation",
        "agent framework",
        "agent memory",
        "agent reasoning",
        "agent reliability",
        "agent safety",
        "agent simulation",
        "agent society",
        "agent specialization",
        "agent system",
        "agent tool use",
        "agent training",
        "agent-awareness",
        "agent-based simulation",
        "agent-centric benchmark",
        "agent-to-agent protocol",
        "agentic",
        "agentic ai",
        "agentic augmentation",
        "agentic coding",
        "agentic collaboration",
        "agentic data synthesis",
        "agentic framwork",
        "agentic learning",
        "agentic llm",
        "agentic misalignment",
        "agentic multi-turn reasoning",
        "agentic multimodal model",
        "agentic rag",
        "agentic reasoning",
        "agentic reinforcement learning",
        "agentic rl",
        "agentic search",
        "agentic serving",
        "agentic simulation",
        "agentic system",
        "agentic systems",
        "agentic task",
        "agentic training",
        "agentic workflow",
        "agentic workflows",
        "agentless",
        "agents",
        "agents with memory",
        "ai agent",
        "ai agents",
        "app agent",
        "automated code repair",
        "autonomous agent",
        "autonomous agents",
        "benchmarks for agents",
        "browser-use agent",
        "chart-to-code generation",
        "code agent",
        "code generation",
        "code synthesis",
        "coding agent",
        "coding agent benchmark",
        "coding agents",
        "computer use",
        "computer use agent",
        "computer use agents",
        "computer use-agent",
        "computer-use agent",
        "computer-use agents",
        "computer-using agent",
        "computer-using agents",
        "constraint-aware tool use",
        "cooperative multi-agent rl",
        "cuda code generation",
        "data agent",
        "data science agents",
        "database agents",
        "deep research agent",
        "deep search agent",
        "digital agents",
        "doctor agent",
        "earth-agent",
        "embodied agent",
        "embodied agents",
        "function calling",
        "generalist agent",
        "generative agents",
        "geometry agent prover",
        "gui agent",
        "gui agent safety",
        "gui agents",
        "heterogeneous agent reinforcement learning",
        "heterogeneous agents",
        "human debugging",
        "human-agent cooperation",
        "human-agent interaction",
        "inter-agent communication",
        "language agent",
        "language agents",
        "language model agents",
        "large language models for code generation",
        "llm agent",
        "llm agents",
        "llm based agent",
        "llm search agents",
        "llm tool use",
        "llm-based agent",
        "llm-based agent systems",
        "llm-based agents",
        "llm/vlm agent",
        "lm agent",
        "long-horizon agent",
        "long-horizon agents",
        "medical agent",
        "medical agents",
        "memory agents benchmark",
        "mobile agent",
        "mobile agents",
        "mobile gui agent",
        "mobile-agent",
        "model debugging",
        "multi agent",
        "multi agent reinforcement learning",
        "multi agent system",
        "multi agent systems",
        "multi-agent",
        "multi-agent collaboration",
        "multi-agent communication",
        "multi-agent coordination",
        "multi-agent debate",
        "multi-agent discussion",
        "multi-agent framework",
        "multi-agent imitation learning",
        "multi-agent llm",
        "multi-agent llm systems",
        "multi-agent llms",
        "multi-agent optimization",
        "multi-agent orchestration",
        "multi-agent react",
        "multi-agent reinforcement learning",
        "multi-agent simulation",
        "multi-agent system",
        "multi-agent systems",
        "multi-agent systems (mas)",
        "multi-agent system，large language model，evidence-based reasoning",
        "multi-agent traffic simulation",
        "multi-modal agent",
        "multi-modal embodied agent",
        "multi-modal large language agent",
        "multi-social-agent self-distillation",
        "multi-turn tool use",
        "multiagent systems",
        "multimodal agent",
        "multimodal agents",
        "multimodal code generation",
        "neural program synthesis",
        "neurosymbolic agent",
        "proactive agent",
        "proactive agents",
        "program synthesis",
        "real-world github issues",
        "reasoning scaffolding",
        "rebuttal agent",
        "repository generation",
        "research agents",
        "robotic tool use",
        "role-playing character agent",
        "role-playing language agents",
        "runtime code execution",
        "safe multi-agent reinforcement learning",
        "scalable tool use",
        "science agents",
        "search agent",
        "secure code generation",
        "self-evolving agent",
        "software engineering",
        "spatial agent",
        "swe agent",
        "swe-agent",
        "tool agent",
        "tool use",
        "tool use evaluation",
        "tool-augmented agents",
        "tool-augmented llms",
        "tool-augmented lms",
        "tool-augmented reasoning",
        "tool-use",
        "tool-use alignment",
        "tool-use large language model",
        "tool-using agent",
        "verifiable code generation",
        "verifiable coding agents",
        "videoagenttrek",
        "vision centric agents",
        "vlm agent",
        "web agent",
        "web agents",
        "webagent",
        "Static analysis",
        "Continous integration",
        "Test case generation",
        "Code search",
        "Compiler",
        "Complier optimisation",
    ]
    kw_1 = [kw.lower() for kw in kw_1]
    len(kw_1)
    return (kw_1,)


@app.cell(hide_code=True)
def _():
    # List 2 – Reinforcement Learning & Post-training
    kw_2 = [
        "action free offline to online reinforcement learning",
        "actor-critic",
        "adversarial imitation learning",
        "adversarial reward learning",
        "adversarial rl",
        "agentic reinforcement learning",
        "agentic rl",
        "automatic reward design",
        "batch reinforcement learning",
        "behavioral cloning",
        "bi-level reinforcement learning",
        "black-box reward optimization",
        "constrained reinforcement learning",
        "continual instruction tuning",
        "continual reinforcement learning",
        "continuous-time reinforcement learning",
        "convex reinforcement learning",
        "critique reinforcement learning",
        "cross-domain reinforcement learning",
        "curriculum learning",
        "deep reinforcement learning",
        "deep rl",
        "delayed reinforcement learning",
        "deterministic policy gradient",
        "direct preference optimization",
        "direct preference optimization (dpo)",
        "distributed reinforcement learning",
        "distributional reinforcement learning",
        "domain-specific sft",
        "double q-learning",
        "dpo",
        "druggability‑tailored preference optimization",
        "dual critic reinforcement learning",
        "erasable reinforcement learning",
        "flow actor-critic",
        "forgetting-augmented reinforcement learning",
        "generative adversarial imitation learning",
        "generative process reward model",
        "generative reward model",
        "goal-conditioned reinforcement learning",
        "group relative policy optimization (grpo)",
        "grpo",
        "heterogeneous agent reinforcement learning",
        "hierarchial reinforcement learning",
        "hierarchical reinforcement learning",
        "hierarchical rl",
        "hybrid rewards for reinforcement learning",
        "image reward model",
        "imitation learning",
        "imitation learning for robotics",
        "in-context reinforcement learning",
        "instruction tuning",
        "inverse reinforcement learning",
        "language model post-training",
        "llm fine-tuning (sft",
        "llm post-training",
        "llm sft",
        "meta reinforcement learning",
        "meta-reinforcement learning",
        "model based reinforcement learning",
        "model-based reinforcement learning",
        "model-based rl",
        "model-basedreinforcement learning",
        "multi agent reinforcement learning",
        "multi-agent imitation learning",
        "multi-agent reinforcement learning",
        "multi-objective reinforcement learning",
        "multi-objective reinforcement learning (morl)",
        "multi-task reinforcement learning",
        "multimodal process reward model",
        "multimodal reinforcement learning",
        "multimodal reward model",
        "non-stationary reinforcement learning",
        "off-policy rl",
        "off-policy rlvr",
        "offline goal-conditioned reinforcement learning",
        "offline imitation learning",
        "offline reinforcement learning",
        "offline rl",
        "offline safe reinforcement learning",
        "offline-to-online reinforcement learning",
        "on-policy",
        "on-policy actor-critic",
        "on-policy optimization",
        "on-policy sampling",
        "online dpo",
        "online reinforcement learning",
        "opponent shaping",
        "pluralistic reward modeling",
        "policy gradient",
        "policy gradient learning",
        "policy gradient methods",
        "policy gradients",
        "post-training",
        "post-training compression",
        "post-training quantization",
        "ppo",
        "preference based reinforcement learning",
        "preference learning",
        "preference optimization",
        "preference-based reinforcement learning",
        "process reward",
        "process reward model",
        "process reward modeling",
        "process reward models",
        "q-learning",
        "ranking and preference learning",
        "reinfocement learning with verifiable reward",
        "reinforcement learning",
        "reinforcement learning (rl)",
        "reinforcement learning fine-tuning",
        "reinforcement learning finetuning",
        "reinforcement learning for language models",
        "reinforcement learning for llms",
        "reinforcement learning from human feedback",
        "reinforcement learning from human feedback (rlhf)",
        "reinforcement learning from offline data",
        "reinforcement learning from verifiable rewards",
        "reinforcement learning from verifier rewards",
        "reinforcement learning specifications",
        "reinforcement learning theory",
        "reinforcement learning with human feedback",
        "reinforcement learning with human preference",
        "reinforcement learning with verifiable reward",
        "reinforcement learning with verifiable reward (rlvr)",
        "reinforcement learning with verifiable rewards",
        "reinforcement learning，image aesthetic assessment",
        "reward for reinforcement learning",
        "reward hacking",
        "reward hacking detection",
        "reward hacking prevention",
        "reward learning",
        "reward model",
        "reward modeling",
        "reward modeling as reasoning",
        "reward models",
        "reward shaping",
        "reward-free reinforcement learning",
        "risk-sensitive reinforcement learning",
        "rl finetuning",
        "rl finetuning of llms",
        "rl for llms",
        "rl from verifiable rewards",
        "rlaif",
        "rlhf",
        "rlhf training dynamics",
        "rlvr",
        "robot imitation learning",
        "robust reinforcement learning",
        "safe multi-agent reinforcement learning",
        "safe reinforcement learing",
        "safe reinforcement learning",
        "sample-efficient reinforcement learning",
        "self-imitation learning",
        "self-play",
        "self-play optimization",
        "sft",
        "skill-based reinforcement learning",
        "soft actor-critic (sac)",
        "stable reinforcement learning",
        "support constraint",
        "test-time reinforcement learning",
        "theory of reinforcement learning",
        "token-level policy gradients reshape",
        "tool-call reward model",
        "unsupervised reinforcement learning",
        "value function",
        "value function estimation",
        "value function factorization",
        "zero-shot reinforcement learning",
        "MDP",
        "POMDP",
        "Bandit",
        "Multi-armed bandit",
        "Contextual bandit",
        "Combinatorial multi-armed bandit",
        "Combinatorial semi-bandits",
        "Monte Carlo",
        "Temporal difference",
        "Bellman equation",
        "Advantage estimation",
        "Entropy regularisation",
        "Off-policy",
        "Policy optimisation",
        "Intrinsic motivation",
    ]
    kw_2 = [kw.lower() for kw in kw_2]
    len(kw_2)
    return (kw_2,)


@app.cell(hide_code=True)
def _():
    # List 3 – Inference Time Scaling & Optimization
    kw_3 = [
        "activation quantization",
        "adaptive compute",
        "attention-aware kv cache update",
        "auto think",
        "beam search",
        "best-of-n",
        "chain of thought",
        "chain of thoughts",
        "chain-of-thought",
        "chain-of-thought (cot)",
        "chain-of-thought prompting",
        "chain-of-thought reasoning",
        "compute budget allocation",
        "context parallel machenism",
        "cot",
        "cot monitoring",
        "cot reasoning",
        "early exit",
        "efficient llm inference",
        "generation chain-of-thought",
        "grounded chain-of-thought",
        "inference optimization",
        "inference scaling",
        "inference time",
        "inference time techniques",
        "inference-time adaptation",
        "inference-time algorithms",
        "inference-time alignment",
        "inference-time compute",
        "inference-time control",
        "inference-time improvement",
        "inference-time knowledge integration",
        "inference-time method",
        "inference-time optimization",
        "inference-time scaling",
        "interleaved chain-of-thought",
        "joint model compression",
        "kv cache",
        "kv cache compression",
        "kv cache eviction",
        "kv cache quantization",
        "kv cache retrieval",
        "kv cache subselection",
        "kv compression",
        "latent chain-of-thought",
        "layer-aware kv cache update",
        "llm decoding",
        "llm inference",
        "llm inference acceleration",
        "llm inference optimization",
        "long chain of thought",
        "long chain-of-thought",
        "long cot distillation",
        "long cots",
        "long-cot reasoning",
        "majority voting",
        "model compression",
        "model quantization",
        "multi-modal chain-of-thought",
        "multimodal interleaved chain-of-thought (cot)",
        "multimodal large language model；chain-of-thought",
        "parallel decoding",
        "post training quantization",
        "post-training quantization",
        "quantization",
        "reasoning scaling",
        "scaling test-time compute",
        "self-consistency",
        "soft best-of-n",
        "soft-best-of-n",
        "spatial chain-of-thought",
        "speculative decoding",
        "test time compute",
        "test time scaling",
        "test-time compute",
        "test-time scaling",
        "unfaithful chain-of-thought",
        "vector-quantization",
        "visual chain of thought",
        "sampling",
        "Top-k sampling",
        "Top-p",
        "Nucleus sampling",
        "Contrastive decoding",
        "Batch decoding",
        "GPTQ",
    ]
    kw_3 = [kw.lower() for kw in kw_3]
    len(kw_3)
    return (kw_3,)


@app.cell(hide_code=True)
def _():
    # List 4 – Infrastructure
    kw_4 = [
        "activation checkpointing",
        "batch size",
        "batch size scheduling",
        "cloud computing",
        "continual pretraining",
        "critical batch size",
        "distributed system",
        "distributed systems",
        "distributed training",
        "efficient training",
        "expert offloading",
        "federated fine-tuning",
        "federated learning",
        "fsdp",
        "gpu-parallel optimization",
        "gradient checkpointing",
        "gradient compression",
        "hardware acceleration",
        "hardware-efficient attention",
        "hierarchical federated learning",
        "knowledge offloading",
        "learning rate schedule",
        "learning rate schedules",
        "llm pretraining",
        "memory efficient training",
        "memory-efficient training",
        "ml system",
        "muon optimizer",
        "offloading",
        "one-shot federated learning",
        "optimal batch size",
        "parameter server",
        "parameter-efficient training",
        "personalized federated fine-tuning",
        "riemannian federated learning",
        "semi-supervised federated learning",
        "sequence parallelism",
        "training efficiency",
        "vertical federated learning",
        "GPU",
        "GPUs",
        "GPU kernel",
        "Triton kernel",
        "Cuda kernels",
        "Distributed computing",
        "Distributed computing",
        "Checkpoint merging",
        "Parallelism",
        "Task parallelism",
    ]
    kw_4 = [kw.lower() for kw in kw_4]
    len(kw_4)
    return (kw_4,)


@app.cell(hide_code=True)
def _():
    # List 5 – AI Safety, Governance & Ethics
    kw_5 = [
        "accountability",
        "adversarial robustness",
        "ai alignment",
        "ai fairness",
        "ai privacy",
        "ai risk",
        "ai safety",
        "ai safety and security",
        "ai truthfulness and deception",
        "aigc copyright protection",
        "aigc watermarking",
        "algorithmic fairness",
        "applications of interpretability",
        "automated interpretability",
        "benchmarking interpretability",
        "bias and fairness",
        "bias detection",
        "black-box jailbreak",
        "black-box watermark",
        "brain–ai alignment",
        "chain-of-thought interpretability",
        "cognitive interpretability",
        "concept-based explainability",
        "content moderation",
        "copyright",
        "copyright protection",
        "counterfactuality",
        "data privacy",
        "data privacy in generative models",
        "dataset copyright protection",
        "deception",
        "deepfake detection",
        "deepfakes",
        "deepfakes attribution",
        "democratic ai alignment",
        "developmental interpretability",
        "differential privacy",
        "digital asset watermarking",
        "digital watermark",
        "digital watermarking",
        "emotion hallucination",
        "ethical ai",
        "explainability",
        "explainability ai",
        "factuality",
        "factuality evaluation",
        "fairness",
        "fairness auditing",
        "fairness benchmark",
        "fairness in machine learning",
        "fairness-accuracy tradeoff",
        "fake news",
        "green ai",
        "guardrail",
        "guardrails",
        "hallucination",
        "hallucination and confabulation",
        "hallucination control",
        "hallucination detection",
        "hallucination in vision language model",
        "hallucination mitigation",
        "hallucinations",
        "hate speech detection",
        "human-ai alignment",
        "human–ai alignment",
        "image watermark",
        "indirect prompt injection",
        "individualized differential privacy",
        "intellectual property",
        "intellectual property protection",
        "interpretability",
        "interpretability and analysis",
        "interpretability and explainable ai",
        "interpretability of neural networks",
        "interpretability techniques",
        "intrinsic interpretability",
        "jailbreak",
        "jailbreak attack",
        "jailbreak attacks",
        "jailbreak defense",
        "jailbreak detection",
        "jailbreak guard",
        "jailbreaking",
        "jailbreaking attacks",
        "jailbreaks",
        "label misinformation",
        "language model interpretability",
        "llm deception",
        "llm hallucination",
        "llm interpretability",
        "llm response factuality",
        "llm safety",
        "llm watermarking",
        "local differential privacy",
        "long-form hallucination",
        "mechanism interpretability",
        "mechanistic interpretability",
        "membership inference attack",
        "membership inference attacks",
        "misinformation",
        "mitigating hallucination",
        "mllm hallucination",
        "model analysis & interpretability",
        "model explainability",
        "model interpretability",
        "model safety",
        "multimodal misinformation detection",
        "pluralistic ai alignment",
        "prompt injection attack",
        "prompt injection attacks",
        "prompt injections",
        "re-watermarking",
        "red teaming",
        "relational hallucination",
        "relative value learning",
        "robust fairness",
        "safety alignment",
        "semantic-level watermark",
        "social impact",
        "structure hallucination",
        "text watermark",
        "top-down interpretability",
        "trustworthy",
        "trustworthy ai",
        "trustworthy llms",
        "trustworthy machine learning",
        "truthfulness",
        "utility-fairness trade-off",
        "value alignment",
        "visual hallucination snowballing",
        "watermark",
        "watermark-based attribution",
        "watermarking",
        "watermarks",
        "white-box jailbreak",
        "Adversarial attacks",
        "Adversarial examples",
        "Adversarial training",
        "Data poisoning",
        "Data poisoning attack",
        "Poisoning attacks",
        "Backdoor attack",
        "Backdoor defence",
        "Backdoor training",
        "Finetuning data stealing",
        "Membership inference",
    ]
    kw_5 = [kw.lower() for kw in kw_5]
    len(kw_5)
    return (kw_5,)


@app.cell(hide_code=True)
def _():
    # List 6 – AI for Science & Life Sciences
    kw_6 = [
        "3d medical image analysis",
        "3d molecular generation",
        "3d molecule generation",
        "ai for biology",
        "ai for healthcare",
        "ai for materials",
        "ai for metascience",
        "ai for physics",
        "ai for science",
        "ai4biology",
        "alphafold3",
        "alternating exposures",
        "alternating gradient descent-ascent",
        "applications to drug discovery",
        "atomistic protein design",
        "automated scientific discovery",
        "biology",
        "biology experimental operation",
        "biology foundation model",
        "brain decoding",
        "cancer patient",
        "cancer survival prediction",
        "cardiac diagnosis",
        "cell biology",
        "chemistry",
        "clinical",
        "clinical decision making",
        "clinical inquiry",
        "clinical natural language processing",
        "clinical reasoning",
        "computational neuroscience",
        "computational pathology",
        "computer aided diagnosis",
        "computer vision for healthcare",
        "convolution alternatives",
        "coupled physics simulation",
        "cryo-em",
        "data-driven scientific discovery",
        "deep learning for health",
        "deficiency diagnosis",
        "diffusion models for computational chemistry",
        "digital pathology",
        "drug discovery",
        "efficient medical segmentation",
        "ehealth",
        "electronic health records",
        "electronic medical record",
        "evolutionary biology",
        "external memory",
        "external validity",
        "fault diagnosis",
        "fluid dynamics",
        "fmri",
        "fmri decoding",
        "fmri prediction",
        "fmri-to-image reconstruction",
        "fragment-based drug design",
        "fragment-based drug discovery",
        "functional magnetic resonance imaging (fmri)",
        "gene expression prediction",
        "generative chemistry",
        "genome",
        "genomics",
        "healthcare",
        "histopathology image representation learning",
        "internal signal",
        "learnability",
        "learnable compression",
        "learnable fractional superlets",
        "learnable non-uniform dft",
        "llm-based drug discovery",
        "machine learning for healthcare",
        "materials science",
        "medical",
        "medical agent",
        "medical agents",
        "medical ai",
        "medical ai evaluation",
        "medical benchmark",
        "medical diagnosis",
        "medical dialogue",
        "medical foundation model",
        "medical generative pre-trained models",
        "medical image",
        "medical image analysis",
        "medical image classification",
        "medical image generation",
        "medical image segmentation",
        "medical imaging",
        "medical imaging analysis",
        "medical literature",
        "medical mllm",
        "medical multi-modal alignment",
        "medical multimodal benchmark",
        "medical multimodal large language model",
        "medical question answering",
        "medical reasoning",
        "medical segmentation",
        "medical time seris",
        "medical ultrasound",
        "medical visual question answering",
        "medical visual reasoning",
        "medical vlm",
        "medical vqa",
        "mental health",
        "molecular dynamics",
        "molecular generation",
        "molecular property prediction",
        "molecule generation",
        "multi-modal diagnosis",
        "multi-scale protein structure",
        "multimodal medical benchmark",
        "multimodal medical reasoning",
        "neural decoding",
        "neuroscience",
        "organic reaction prediction",
        "pan-cancer modeling",
        "pan-cancer screening",
        "physics simulation",
        "physics simulations",
        "policy internalization",
        "pre-trained protein language model",
        "protein design",
        "protein language model",
        "protein language modeling",
        "protein language models",
        "protein structure",
        "protein structure generation",
        "protein structure modeling",
        "protein structure prediction",
        "protein structure refinement",
        "quantum computing",
        "radiology",
        "radiology report generation",
        "rna design",
        "rna foundation model",
        "rna inverse design",
        "rna non-canonical base pair",
        "rna secondry structure prediction",
        "rna structure",
        "rnn transformer alternatives",
        "scientific discovery",
        "scrna-seq",
        "single cell",
        "single cell genomics",
        "single-cell biology",
        "single-cell genomics",
        "single-cell rna sequencing",
        "structural bioinformatics",
        "structural biology",
        "structure based drug design",
        "surrogate gradient alternatives",
        "synthesizable drug design",
        "ternarization",
        "ternary quantization",
        "unlearnable examples",
        "visual neural decoding",
        "Environmental science",
        "Machine learning in environmental science",
        "Climate",
        "Climate change",
        "Climate downscaling",
        "Energy efficiency",
        "Energy saving",
        "Carbon sequestration",
        "Material discovery",
        "Material generation",
        "Material identification",
        "Material property prediction",
        "Differentiable physics",
        "Geophysics",
        "Quantum",
        "Quantum algorithms",
        "Quantum machine learning",
        "Quantum neural networks",
        "Quantum information",
        "Quantum deep learning",
        "Quantum physics",
        "Variational quantum eigensolver",
        "Molecular docking",
    ]
    kw_6 = [kw.lower() for kw in kw_6]
    len(kw_6)
    return (kw_6,)


@app.cell(hide_code=True)
def _():
    # List 7 – Physical AI & Robotics
    kw_7 = [
        "3d manipulation",
        "autonomous vehicles",
        "bimanual manipulation",
        "bimanual mobile manipulation",
        "data generation for robot learning",
        "dexterous grasping",
        "dexterous hand",
        "dexterous manipulation",
        "drag-style manipulation",
        "dual-arm robots",
        "dynamic tactile perception",
        "efficient robot reasoning",
        "efficient robotic manipulations",
        "embodied agent",
        "embodied agents",
        "embodied ai",
        "embodied intelligence",
        "embodied memory",
        "embodied navigation",
        "embodied reasoning",
        "embodied urban navigation",
        "foundation models based robot manipulation",
        "general robotic manipulation",
        "generalist robot policies",
        "generalist robot policy",
        "generalizable grasping",
        "hand manipulation synthesis",
        "heterogeneous tactile data",
        "humanoid",
        "humanoid control",
        "humanoid locomotion",
        "humanoid robot",
        "humanoid robots",
        "image manipulation localization",
        "imitation learning for robotics",
        "key-value cache manipulation",
        "learning robotic policies from videos",
        "llms for robotics",
        "locomotion",
        "locomotion and manipulation",
        "long-horizon manipulation",
        "manipulation",
        "manipulation video",
        "mobile manipulation",
        "motion planning",
        "mujoco benchmarks",
        "multi-joint robot locomotion",
        "multi-modal embodied agent",
        "multi-robot cooperation",
        "real-to-sim",
        "real-to-sim-to-real",
        "robot co-design",
        "robot data generation",
        "robot datasets and benchmarking",
        "robot foundation model",
        "robot hands",
        "robot imitation learning",
        "robot learning",
        "robot learning，forward dynamics，inverse dynamics",
        "robot manipulation",
        "robot navigation",
        "robot planning",
        "robot policy learning",
        "robot simulation",
        "robot skill acquisition",
        "robot task planning",
        "robotic foundation models",
        "robotic manipulation",
        "robotic tool use",
        "robotics",
        "robotics evaluation",
        "robotics foundation model",
        "robotics learning",
        "robotics manipulation",
        "robotics planning",
        "robotics policy",
        "robots",
        "sim-to-real",
        "sim-to-real matching",
        "streaming video manipulation",
        "tactile",
        "tactile dataset",
        "tactile representation learning",
        "tactile sensing",
        "task and motion planning",
        "text image manipulation",
        "uav",
        "uav-based search and rescue",
        "video manipulation localization",
        "vision-based robotics",
        "vision-proprioception policy",
        "whole-body control",
        "zero-shot robotic manipulation",
        "zero-shot sim2real",
        "SLAM",
        "Dense semantic SLAM",
        "Localisation",
        "Planning",
        "Path planning",
        "Dynamic path planning",
        "Domain randomisation",
    ]
    kw_7 = [kw.lower() for kw in kw_7]
    len(kw_7)
    return (kw_7,)


@app.cell(hide_code=True)
def _():
    # List 8 – Media, Video, Games & Music
    kw_8 = [
        "3d avatar",
        "3d avatar modeling",
        "3d diffusion model",
        "3d generative models",
        "3d-aware video generation",
        "4d video generation and editing",
        "animation",
        "atari",
        "audio and music generation",
        "audio generation",
        "audio-to-video generation",
        "auto-regressive image generation",
        "auto-regressive video generation",
        "autoregressive image generation",
        "autoregressive video generation",
        "avatar",
        "board games",
        "camera-controllable video generation",
        "camera-guided video generation",
        "causal text to video generation",
        "character animation",
        "chess",
        "coherent video generation",
        "conditional diffusion model",
        "conditional diffusion models",
        "conditional image generation",
        "continuous image generation",
        "controllable hand image generation",
        "controllable image generation",
        "controllable image synthesis",
        "controllable speech synthesis",
        "controllable video diffusion models",
        "controllable video editing",
        "controllable video generation",
        "creative writing",
        "denoising diffusion models",
        "diffusion model",
        "diffusion model alignment",
        "diffusion models",
        "discrete diffusion model",
        "discrete diffusion models",
        "efficient autoregressive image generation",
        "efficient video generation",
        "efficient video understanding",
        "facevideo editing",
        "filmmaking",
        "first-last frame video generation",
        "game ai",
        "game generative ai models",
        "game playing",
        "game reasoning",
        "games",
        "gaming",
        "gaussian avatar",
        "generative video models",
        "high-resolution text-to-video generation",
        "human animation",
        "human video generation",
        "image and video generation",
        "image generation",
        "image generation benchmark",
        "image generation model unlearning",
        "image synthesis",
        "infinite-length video generation",
        "instruction-guided video editing",
        "interactive video generation",
        "interleaved text and image generation",
        "interpretable diffusion model",
        "joint audio-video generation",
        "keyframe narratives",
        "latent diffusion model",
        "latent diffusion models",
        "long video generation",
        "long video understanding",
        "long-video understanding",
        "mask diffusion model",
        "masked diffusion model",
        "masked diffusion models",
        "minecraftocc dataset",
        "mobile video generation",
        "movie scene boundary detection",
        "multi-person interactive video generation",
        "multi-shot video generation",
        "multi-track music generation",
        "multi-view diffusion model",
        "multi-view diffusion models",
        "music generation",
        "narrative coherence",
        "narrative modeling",
        "narratives",
        "online ultra-long video understanding",
        "online video understanding",
        "pixel diffusion model",
        "portrait animation",
        "pose-free animation",
        "real-time video generation",
        "relation aware text-to-audio generaion",
        "scene-consistent video generation",
        "score-based diffusion model",
        "score-based diffusion models",
        "score-based generative models",
        "shortcut diffusion models",
        "sounding video generation",
        "speech generation",
        "speech synthesis",
        "stable diffusion model",
        "story generation",
        "streaming video understanding",
        "subject-consistent image generation",
        "subject-driven image generation",
        "talking person video generation",
        "text diffusion model",
        "text to image synthesis",
        "text to video generation",
        "text-to-audio-video generation",
        "text-to-image",
        "text-to-image diffusion model",
        "text-to-image diffusion models",
        "text-to-image generation",
        "text-to-image generative evaluation",
        "text-to-image models",
        "text-to-image synthesis",
        "text-to-speech",
        "text-to-speech (tts)",
        "text-to-speech synthesis",
        "text-to-video",
        "text-to-video dataset",
        "text-to-video generation",
        "ultra-long video synthesis",
        "video compression",
        "video diffusion",
        "video diffusion acceleration",
        "video diffusion model",
        "video diffusion models",
        "video diffusion transfer",
        "video editing",
        "video generation",
        "video generation models",
        "video generative model",
        "video model",
        "video modeling",
        "video models",
        "video understanding",
        "video understanding & activity analysis",
        "video-to-audio generation",
        "world2minecraft",
        "TTS",
        "AI for music",
        "Audio coding",
        "Audio comprehension",
        "Audio editing",
        "Audio inpainting",
        "Audio language model",
        "Sound separation",
        "Image-grounded video perception and reasoning",
        "Universal sound separation",
        "Sound source localisation",
    ]
    kw_8 = [kw.lower() for kw in kw_8]
    len(kw_8)
    return (kw_8,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Load

    Loading the fully scored dataset (output of notebooks 1-5): 5,358 accepted
    papers with review features, cluster assignments, UMAP coordinates, and
    8 archetype scores.
    """)
    return


@app.cell
def _():
    df_all = pl.read_parquet("iclr_2026_scored.parquet")
    print(f"Loaded {df_all.shape[0]} rows × {df_all.shape[1]} columns")
    return (df_all,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 Datasets & Benchmarks split

    Before diving into thematic categories, we separate out papers whose
    `primary_area` is "datasets and benchmarks". These are important contributions
    but have a different evaluation profile — they are judged on data quality and
    coverage rather than methodological novelty. We analyze their distribution
    across categories and then exclude them from the main analysis.
    """)
    return


@app.cell
def _(df_all):
    BENCH_AREA = "datasets and benchmarks"
    df_bench = df_all.filter(pl.col("primary_area") == BENCH_AREA)
    df = df_all.filter(pl.col("primary_area") != BENCH_AREA)
    print(
        f"Datasets & Benchmarks: {df_bench.shape[0]} papers ({df_bench.shape[0] / df_all.shape[0] * 100:.1f}%)"
    )
    print(
        f"Remaining for analysis: {df.shape[0]} papers ({df.shape[0] / df_all.shape[0] * 100:.1f}%)"
    )
    return (BENCH_AREA, df, df_bench)


@app.cell
def _(BENCH_AREA, df_all, df_bench, kw_1, kw_2, kw_3, kw_4, kw_5, kw_6, kw_7, kw_8):
    # How many benchmark papers fall into each of our 8 categories?
    bench_kw_lists = {
        "Agents": kw_1,
        "RL": kw_2,
        "Inference": kw_3,
        "Infra": kw_4,
        "Safety": kw_5,
        "Science": kw_6,
        "Robotics": kw_7,
        "Media": kw_8,
    }
    bench_rows = []
    for name, kws in bench_kw_lists.items():
        in_cat_all = df_all.filter(
            pl.col("keywords")
            .list.eval(pl.element().str.to_lowercase().is_in(kws))
            .list.any()
        ).shape[0]
        in_cat_bench = df_bench.filter(
            pl.col("keywords")
            .list.eval(pl.element().str.to_lowercase().is_in(kws))
            .list.any()
        ).shape[0]
        bench_rows.append(
            {
                "category": name,
                "total_in_cat": in_cat_all,
                "bench_in_cat": in_cat_bench,
                "bench_pct": round(in_cat_bench / max(in_cat_all, 1) * 100, 1),
            }
        )

    bench_overview = pl.DataFrame(bench_rows)
    overall_bench_pct = round(df_bench.shape[0] / df_all.shape[0] * 100, 1)
    print(f"Overall benchmark proportion: {overall_bench_pct}%\n")
    print("Benchmark share per category:")
    bench_overview
    return (bench_overview,)


@app.cell
def _(bench_overview):
    fig_bench = px.bar(
        bench_overview.to_pandas(),
        x="category",
        y="bench_pct",
        text="bench_in_cat",
        title="Share of 'Datasets & Benchmarks' papers in each category",
        labels={"bench_pct": "% that are benchmarks", "category": "Category"},
    )
    fig_bench.add_hline(
        y=8.3,
        line_dash="dash",
        line_color="red",
        annotation_text="overall avg (8.3%)",
        annotation_position="top right",
    )
    fig_bench.update_layout(height=400)
    fig_bench.write_html(str(FIGURES / "benchmark_share.html"))
    fig_bench
    return (fig_bench,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.2 Filter by keywords

    Each paper (excluding benchmarks) is matched against the 8 keyword lists
    above. A paper can belong to multiple categories (e.g. an "RL agent" paper
    matches both Agents and RL).
    """)
    return


@app.cell
def _(df, kw_1):
    df_1 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_1))
        .list.any()
    )
    df_1
    return (df_1,)


@app.cell
def _(df, kw_2):
    df_2 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_2))
        .list.any()
    )
    df_2
    return (df_2,)


@app.cell
def _(df, kw_3):
    df_3 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_3))
        .list.any()
    )
    df_3
    return (df_3,)


@app.cell
def _(df, kw_4):
    df_4 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_4))
        .list.any()
    )
    df_4
    return (df_4,)


@app.cell
def _(df, kw_5):
    df_5 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_5))
        .list.any()
    )
    df_5
    return (df_5,)


@app.cell
def _(df, kw_6):
    df_6 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_6))
        .list.any()
    )
    df_6
    return (df_6,)


@app.cell
def _(df, kw_7):
    df_7 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_7))
        .list.any()
    )
    df_7
    return (df_7,)


@app.cell
def _(df, kw_8):
    df_8 = df.filter(
        pl.col("keywords")
        .list.eval(pl.element().str.to_lowercase().is_in(kw_8))
        .list.any()
    )
    df_8
    return (df_8,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Category Metadata & Helpers

    Scoring helpers recompute archetype z-scores **locally within each category
    subset** rather than reusing global scores. A paper that is globally average
    may still be the best in its category.
    """)
    return


@app.cell
def _(
    df_1,
    df_2,
    df_3,
    df_4,
    df_5,
    df_6,
    df_7,
    df_8,
    kw_1,
    kw_2,
    kw_3,
    kw_4,
    kw_5,
    kw_6,
    kw_7,
    kw_8,
):
    CATEGORIES = {
        1: {"label": "Agents / SWE", "short": "Agents", "kw": kw_1, "df": df_1},
        2: {"label": "RL / Post-training", "short": "RL", "kw": kw_2, "df": df_2},
        3: {"label": "Inference Scaling", "short": "Inference", "kw": kw_3, "df": df_3},
        4: {"label": "Infrastructure", "short": "Infra", "kw": kw_4, "df": df_4},
        5: {"label": "Safety / Governance", "short": "Safety", "kw": kw_5, "df": df_5},
        6: {"label": "AI for Science", "short": "Science", "kw": kw_6, "df": df_6},
        7: {"label": "Robotics", "short": "Robotics", "kw": kw_7, "df": df_7},
        8: {"label": "Media / Video / Games", "short": "Media", "kw": kw_8, "df": df_8},
    }
    CAT_NAMES = {k: _v["short"] for k, _v in CATEGORIES.items()}

    for _idx in CATEGORIES:
        short = CATEGORIES[_idx]["short"]
        n_papers = CATEGORIES[_idx]["df"].shape[0]  # ty: ignore[unresolved-attribute]
        n_kw = len(CATEGORIES[_idx]["kw"])  # ty: ignore[unresolved-attribute]
        print(f"  Cat {_idx} ({short:>10s}): {n_papers:4d} papers, {n_kw:3d} keywords")
    return (CATEGORIES, CAT_NAMES)


@app.cell
def _():
    def _z(col: pl.Expr) -> pl.Expr:
        """Standardize a column to zero mean, unit variance."""
        return (col - col.mean()) / col.std()

    def compute_local_scores(subset: pl.DataFrame) -> pl.DataFrame:
        """Recompute 4 archetype scores using category-LOCAL z-scores."""
        if subset.shape[0] < 10:
            return subset.with_columns(
                pl.col("score_top_overall").alias("local_top_overall"),
                pl.col("score_hidden_gem").alias("local_hidden_gem"),
                pl.col("score_controversial").alias("local_controversial"),
                pl.col("score_consensus").alias("local_consensus"),
            )
        return subset.with_columns(
            (
                _z(pl.col("rating_mean")) * 0.35
                + _z(pl.col("soundness_mean")) * 0.25
                + _z(pl.col("contribution_mean")) * 0.25
                - _z(pl.col("rating_std")) * 0.10
                + _z(pl.col("confidence_mean")) * 0.05
            )
            .fill_nan(0.0)
            .alias("local_top_overall"),
            (
                _z(pl.col("rating_mean")) * 0.30
                + _z(pl.col("soundness_mean")) * 0.25
                + _z(pl.col("contribution_mean")) * 0.25
                - _z(pl.col("n_replies").cast(pl.Float64)) * 0.10
                - _z(pl.col("total_review_wc").cast(pl.Float64)) * 0.10
            )
            .fill_nan(0.0)
            .alias("local_hidden_gem"),
            (
                _z(pl.col("rating_std")) * 0.25
                + _z(pl.col("rating_range").cast(pl.Float64)) * 0.25
                + _z(pl.col("wc_questions_mean")) * 0.20
                + _z(pl.col("n_replies").cast(pl.Float64)) * 0.15
                + _z(pl.col("corr_rating_confidence")).abs() * 0.15
            )
            .fill_nan(0.0)
            .alias("local_controversial"),
            (
                _z(pl.col("rating_mean")) * 0.40
                - _z(pl.col("rating_std")) * 0.30
                + _z(pl.col("confidence_mean")) * 0.30
            )
            .fill_nan(0.0)
            .alias("local_consensus"),
        )

    def diversified_top_n_cat(
        scored_df: pl.DataFrame,
        score_col: str,
        n: int,
        max_per_cluster: int = 2,
    ) -> pl.DataFrame:
        """Select top-N papers within a category with cluster diversity."""
        ranked = scored_df.sort(score_col, descending=True)
        selected: list[dict] = []
        cluster_counts: dict[int, int] = {}
        for _r in ranked.iter_rows(named=True):
            if len(selected) >= n:
                break
            cl = _r.get("cluster_ward")
            if cl is not None and cluster_counts.get(cl, 0) >= max_per_cluster:
                continue
            selected.append(_r)
            if cl is not None:
                cluster_counts[cl] = cluster_counts.get(cl, 0) + 1
        return pl.DataFrame(selected)

    def format_paper_cat(row: dict, reason: str) -> str:
        """Format a paper for display."""
        return (
            f"  [{row['status']:8s}] rating={row['rating_mean']:.1f} "
            f"sound={row['soundness_mean']:.1f} contrib={row['contribution_mean']:.1f}"
            f" cluster={row['cluster_ward']}\n"
            f"    {row['title'][:90]}\n"
            f"    area={row['primary_area'][:50]}\n"
            f"    {row['site']}\n"
            f"    -> {reason}"
        )

    return (compute_local_scores, diversified_top_n_cat, format_paper_cat)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Category Overview Dashboard

    RL / Post-training is the largest category (740 papers), reflecting ICLR's
    core identity. Infrastructure is the smallest (100) — a specialized niche.
    Media/Video/Games is surprisingly large (512), driven by the generative AI
    wave. Robotics has the highest mean rating (~5.55), while Science has the
    lowest (~5.27) — possibly because frontier science is harder to evaluate.
    """)
    return


@app.cell
def _(CATEGORIES):
    rows = []
    for _idx, _cat in CATEGORIES.items():
        _sub = _cat["df"]
        _n = _sub.shape[0]
        if _n == 0:
            continue
        oral_pct = (_sub.filter(pl.col("status") == "Oral").shape[0] / _n) * 100
        kw_col = (
            "canonical_keywords" if "canonical_keywords" in _sub.columns else "keywords"
        )
        kw_flat = (
            _sub.select(pl.col(kw_col).explode().str.to_lowercase())
            .to_series()
            .drop_nulls()
        )
        top_kw = ", ".join(
            kw_flat.value_counts(sort=True).head(3).get_column(kw_flat.name).to_list()
        )
        rows.append(
            {
                "category": _cat["short"],
                "papers": _n,
                "oral_pct": round(oral_pct, 1),
                "mean_rating": round(_sub["rating_mean"].mean(), 2),
                "median_rating": round(_sub["rating_mean"].median(), 2),
                "mean_soundness": round(_sub["soundness_mean"].mean(), 2),
                "mean_contribution": round(_sub["contribution_mean"].mean(), 2),
                "top_keywords": top_kw,
            }
        )
    overview_df = pl.DataFrame(rows)
    overview_df
    return (overview_df,)


@app.cell
def _(overview_df):
    fig_overview = px.bar(
        overview_df.to_pandas(),
        y="category",
        x="papers",
        color="mean_rating",
        color_continuous_scale="RdYlGn",
        orientation="h",
        text="papers",
        hover_data=["oral_pct", "mean_soundness", "mean_contribution", "top_keywords"],
        title="Papers per Category (colored by mean rating)",
    )
    fig_overview.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    fig_overview.write_html(str(FIGURES / "category_overview.html"))
    fig_overview
    return (fig_overview,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Per-Category Top 10 Papers

    For each category we select 10 papers via a budget-based approach using
    **category-local** z-scores: 3 top-overall + 2 hidden gems (poster only) +
    2 controversial + 3 consensus standouts, deduplicated and diversified across
    Ward clusters.
    """)
    return


@app.cell
def _(CATEGORIES, compute_local_scores, diversified_top_n_cat, format_paper_cat):
    all_category_picks: dict[int, list[dict]] = {}

    for _idx, _cat in CATEGORIES.items():
        _sub = _cat["df"]
        _n = _sub.shape[0]
        print(f"\n{'=' * 70}")
        print(f"Category {_idx}: {_cat['label']} ({_n} papers)")
        print(f"{'=' * 70}")

        if _n == 0:
            all_category_picks[_idx] = []
            continue

        scored = compute_local_scores(_sub)
        seen: set[str] = set()
        picks: list[dict] = []

        budget = [
            ("local_top_overall", 3, None, "Top overall (local)"),
            ("local_hidden_gem", 2, "Poster", "Hidden gem (local)"),
            ("local_controversial", 2, None, "Controversial (local)"),
            ("local_consensus", 3, None, "Consensus standout (local)"),
        ]
        for sc, want, sf, label in budget:
            pool = scored if not sf else scored.filter(pl.col("status") == sf)
            top = diversified_top_n_cat(pool, sc, want + 5)
            added = 0
            for _r in top.iter_rows(named=True):
                if len(picks) >= 10 or added >= want:
                    break
                if _r["openreview_id"] in seen:
                    continue
                seen.add(_r["openreview_id"])
                rc = dict(_r)
                rc["label"] = label
                rc["_score"] = _r[sc]
                picks.append(rc)
                added += 1

        if len(picks) < 10:
            bf = diversified_top_n_cat(scored, "local_top_overall", 15)
            for _r in bf.iter_rows(named=True):
                if len(picks) >= 10:
                    break
                if _r["openreview_id"] in seen:
                    continue
                seen.add(_r["openreview_id"])
                rc = dict(_r)
                rc["label"] = "Top overall (backfill)"
                rc["_score"] = _r["local_top_overall"]
                picks.append(rc)

        all_category_picks[_idx] = picks
        for _i, _p in enumerate(picks, 1):
            _lbl = _p["label"]
            scr = _p["_score"]
            print(f"\n{_i}. {format_paper_cat(_p, f'{_lbl} = {scr:.3f}')}")
    return (all_category_picks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Category x Archetype Heatmap

    Each cell shows what percentage of a category's papers fall into the global
    top-20% for that archetype. The baseline is 20% — values above indicate
    over-representation.

    Key patterns: **Science** papers are disproportionately semantically novel
    (38.8%, nearly 2x baseline) and controversial (30.4%) — frontier research
    explores uncharted territory and sparks disagreement. **Safety** is the most
    debated category (24.9% controversial, 24.0% high-engagement) with the lowest
    consensus (14.6%). **Media** and **Robotics** show the opposite profile:
    high top-overall quality and strong reviewer consensus. **Agents** stand out
    as the most interdisciplinary (24.6% bridge). **Inference** is the most
    well-trodden ground — only 4.6% semantic novel, the lowest of all categories.
    """)
    return


@app.cell
def _(CATEGORIES, df):
    arch_cols = [
        "score_top_overall",
        "score_hidden_gem",
        "score_controversial",
        "score_high_engagement",
        "score_semantic_novel",
        "score_bridge",
        "score_area_leader",
        "score_consensus",
    ]
    arch_labels = [
        _c.replace("score_", "").replace("_", " ").title() for _c in arch_cols
    ]
    thresholds = {_c: df[_c].quantile(0.80) for _c in arch_cols}

    hm_data: list[list[float]] = []
    cat_labels_hm = []
    for _idx in sorted(CATEGORIES.keys()):
        _cat = CATEGORIES[_idx]
        _sub = _cat["df"]
        _n = _sub.shape[0]
        if _n == 0:
            hm_data.append([0.0] * len(arch_cols))
        else:
            row_pcts = []
            for _c in arch_cols:
                pct = (_sub.filter(pl.col(_c) > thresholds[_c]).shape[0] / _n) * 100
                row_pcts.append(round(pct, 1))
            hm_data.append(row_pcts)
        cat_labels_hm.append(f"{_cat['short']} ({_sub.shape[0]})")

    hm_array = np.array(hm_data)
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=hm_array,
            x=arch_labels,
            y=cat_labels_hm,
            colorscale="YlOrRd",
            text=hm_array.astype(str),
            texttemplate="%{text}%",
            textfont={"size": 11},
            colorbar={"title": "% in top 20%"},
        )
    )
    fig_heatmap.update_layout(
        title="Category x Archetype: % of papers in global top-20%<br>(20% = baseline; above = over-indexed)",
        height=450,
        width=900,
        xaxis_title="Archetype",
        yaxis_title="Category",
        yaxis={"autorange": "reversed"},
    )
    fig_heatmap.write_html(str(FIGURES / "category_archetype_heatmap.html"))
    fig_heatmap
    return (fig_heatmap,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7. Category Intersections (UpSet-style)

    Papers can belong to multiple categories simultaneously. The UpSet plot below
    shows multi-category intersections on the left (sorted by intersection
    degree), with single-category sizes on the right for context. Agents & RL is
    typically the dominant overlap — natural since agents rely on RL for training.
    """)
    return


@app.cell
def _(CATEGORIES, df):
    membership = df.select("openreview_id")
    cat_col_names = []
    for _idx in sorted(CATEGORIES.keys()):
        col = f"in_cat_{_idx}"
        cat_col_names.append(col)
        ids = set(CATEGORIES[_idx]["df"]["openreview_id"].to_list())
        membership = membership.with_columns(
            pl.col("openreview_id").is_in(ids).alias(col)
        )
    membership = membership.with_columns(
        pl.sum_horizontal(*cat_col_names).alias("n_categories")
    )
    print(
        f"Papers in at least 1 category: {membership.filter(pl.col('n_categories') >= 1).shape[0]}"
    )
    print(
        f"Papers in 2+ categories: {membership.filter(pl.col('n_categories') >= 2).shape[0]}"
    )
    print(
        f"Papers in 3+ categories: {membership.filter(pl.col('n_categories') >= 3).shape[0]}"
    )
    return (membership, cat_col_names)


@app.cell
def _(CAT_NAMES, CATEGORIES, cat_col_names, membership):
    grouped = (
        membership.group_by(cat_col_names)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    combo_labels: list[str] = []
    combo_counts: list[int] = []
    combo_sets: list[set[int]] = []

    for _row in grouped.iter_rows(named=True):
        participating = set()
        for _idx in sorted(CATEGORIES.keys()):
            if _row[f"in_cat_{_idx}"]:
                participating.add(_idx)
        if len(participating) == 0:
            combo_labels.append("(none)")
        else:
            combo_labels.append(
                " & ".join(CAT_NAMES[_i] for _i in sorted(participating))
            )
        combo_counts.append(_row["count"])
        combo_sets.append(participating)

    # Print single-category sizes, then all multi-category intersections
    print("Single-category sizes:")
    for _lbl, cnt, _s in zip(combo_labels, combo_counts, combo_sets):
        if len(_s) == 1:
            print(f"  {cnt:5d}  {_lbl}")

    # All multi-category intersections (2+), sorted by n_cats desc then count desc
    multi = [
        (_lbl, cnt, _s)
        for _lbl, cnt, _s in zip(combo_labels, combo_counts, combo_sets)
        if len(_s) >= 2
    ]
    multi.sort(key=lambda _x: (-len(_x[2]), -_x[1]))
    print(
        f"\nMulti-category intersections ({len(multi)} groups, {sum(_x[1] for _x in multi)} papers):"
    )
    for _lbl, cnt, _s in multi:
        print(f"  {cnt:5d}  ({len(_s)} cats)  {_lbl}")
    return (combo_labels, combo_counts, combo_sets)


@app.cell
def _(CAT_NAMES, CATEGORIES, combo_counts, combo_labels, combo_sets):
    # Sort: multi-category first (by n_cats desc, then count desc), then singles by count desc
    combos = list(zip(combo_labels, combo_counts, combo_sets, range(len(combo_sets))))
    combos = [(lbl, _c, _s, _i) for lbl, _c, _s, _i in combos if len(_s) > 0]
    combos.sort(key=lambda _x: (-len(_x[2]), -_x[1]))
    # Take all multi-cat groups + top single-cat groups to fill up to 25 bars
    multi_bars = [_x for _x in combos if len(_x[2]) >= 2]
    single_bars = [_x for _x in combos if len(_x[2]) == 1]
    bars = multi_bars + single_bars[: max(0, 25 - len(multi_bars))]

    bar_labels = [_x[0] for _x in bars]
    bar_counts = [_x[1] for _x in bars]
    bar_sets = [_x[2] for _x in bars]
    cat_list = sorted(CATEGORIES.keys())
    cat_names_ord = [CAT_NAMES[_c] for _c in cat_list]

    # Color bars by number of participating categories
    bar_colors = [
        "#c0392c"
        if len(_s) >= 4
        else "#e67e22"
        if len(_s) >= 3
        else "#f39c12"
        if len(_s) >= 2
        else "#3498db"
        for _s in bar_sets
    ]

    fig_upset = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    fig_upset.add_trace(
        go.Bar(
            x=list(range(len(bar_counts))),
            y=bar_counts,
            marker_color=bar_colors,
            text=bar_counts,
            textposition="outside",
            name="Papers",
            hovertext=bar_labels,
        ),
        row=1,
        col=1,
    )
    for ci in range(len(bar_sets)):
        for ri, cid in enumerate(cat_list):
            active = cid in bar_sets[ci]
            fig_upset.add_trace(
                go.Scatter(
                    x=[ci],
                    y=[ri],
                    mode="markers",
                    marker=dict(size=12, color="#2c3e50" if active else "#d5d8dc"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        active_rows = [ri for ri, cid in enumerate(cat_list) if cid in bar_sets[ci]]
        if len(active_rows) > 1:
            fig_upset.add_trace(
                go.Scatter(
                    x=[ci, ci],
                    y=[min(active_rows), max(active_rows)],
                    mode="lines",
                    line=dict(color="#2c3e50", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
    # Vertical separator line between multi-cat and single-cat sections
    if multi_bars and single_bars:
        sep_x = len(multi_bars) - 0.5
        for row_n in [1, 2]:
            fig_upset.add_vline(
                x=sep_x,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=row_n,
                col=1,
            )
    fig_upset.update_layout(
        title="Category Intersections (UpSet-style)"
        "<br><sub>Left of dashed line: multi-category | Right: single-category</sub>",
        height=650,
        width=max(900, len(bar_counts) * 40),
        showlegend=False,
        bargap=0.3,
    )
    fig_upset.update_xaxes(
        tickvals=list(range(len(bar_labels))),
        ticktext=[""] * len(bar_labels),
        row=1,
        col=1,
    )
    fig_upset.update_xaxes(
        tickvals=list(range(len(bar_labels))),
        ticktext=[""] * len(bar_labels),
        row=2,
        col=1,
    )
    fig_upset.update_yaxes(title_text="Papers", row=1, col=1)
    fig_upset.update_yaxes(
        tickvals=list(range(len(cat_names_ord))),
        ticktext=cat_names_ord,
        row=2,
        col=1,
    )
    fig_upset.write_html(str(FIGURES / "category_upset.html"))
    fig_upset
    return (fig_upset,)


@app.cell
def _(CATEGORIES, CAT_NAMES, df, membership):
    multi_mask = membership.filter(pl.col("n_categories") >= 2)
    multi_ids = set(multi_mask["openreview_id"].to_list())

    cat_ranks: dict[int, dict[str, int]] = {}
    for _idx in sorted(CATEGORIES.keys()):
        _sub = CATEGORIES[_idx]["df"].sort("score_top_overall", descending=True)
        for rank, _oid in enumerate(_sub["openreview_id"].to_list(), 1):
            if _oid in multi_ids:
                cat_ranks.setdefault(_idx, {})[_oid] = rank

    multi_rows = []
    for _row in df.filter(pl.col("openreview_id").is_in(list(multi_ids))).iter_rows(
        named=True
    ):
        _oid = _row["openreview_id"]
        n_cats = (
            multi_mask.filter(pl.col("openreview_id") == _oid)
            .select("n_categories")
            .item()
        )
        cats_in = []
        rinfo: dict[str, int | None] = {}
        for _idx in sorted(CATEGORIES.keys()):
            is_in = (
                multi_mask.filter(pl.col("openreview_id") == _oid)
                .select(f"in_cat_{_idx}")
                .item()
            )
            if is_in:
                cats_in.append(CAT_NAMES[_idx])
                rinfo[f"rank_{CAT_NAMES[_idx]}"] = cat_ranks.get(_idx, {}).get(_oid)
        multi_rows.append(
            {
                "title": _row["title"][:70],
                "rating": _row["rating_mean"],
                "status": _row["status"],
                "n_cats": n_cats,
                "categories": ", ".join(cats_in),
                **rinfo,
            }
        )

    multi_cat_df = pl.DataFrame(multi_rows).sort("n_cats", descending=True)
    print(f"\nPapers in 2+ categories: {multi_cat_df.shape[0]}")
    multi_cat_df.head(30)
    return (multi_cat_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 8. UMAP: Papers Colored by Category

    Reusing the UMAP coordinates from the embedding pipeline (notebook 3), we
    color each paper by its category. Categories form coherent spatial clusters:
    Media/Video/Games occupies the far right, RL concentrates in the lower-left,
    and Science papers scatter broadly across the upper region. Multi-category
    papers (cyan) tend to sit at cluster boundaries. "Other" papers (gray) fill
    the gaps — the 53% of ICLR not captured by our 8 categories.
    """)
    return


@app.cell
def _(CATEGORIES, CAT_NAMES, df_all):
    id_to_cats: dict[str, list[int]] = {}
    for _idx in sorted(CATEGORIES.keys()):
        for _oid in CATEGORIES[_idx]["df"]["openreview_id"].to_list():
            id_to_cats.setdefault(_oid, []).append(_idx)

    labels = []
    details = []
    for _oid in df_all["openreview_id"].to_list():
        _cats = id_to_cats.get(_oid, [])
        if len(_cats) == 0:
            labels.append("Other")
            details.append("")
        elif len(_cats) == 1:
            labels.append(CAT_NAMES[_cats[0]])
            details.append(CAT_NAMES[_cats[0]])
        else:
            labels.append("Multi-category")
            details.append(", ".join(CAT_NAMES[_c] for _c in _cats))

    umap_df = df_all.select(
        "umap_x", "umap_y", "title", "rating_mean", "status"
    ).with_columns(
        pl.Series("cat_label", labels),
        pl.Series("categories", details),
    )

    color_map = {
        "Agents": "#e74c3c",
        "RL": "#3498db",
        "Inference": "#2ecc71",
        "Infra": "#9b59b6",
        "Safety": "#e67e22",
        "Science": "#1abc9c",
        "Robotics": "#f39c12",
        "Media": "#e91e63",
        "Multi-category": "#00bcd4",
        "Other": "#d5d8dc",
    }
    cat_order = [
        "Agents",
        "RL",
        "Inference",
        "Infra",
        "Safety",
        "Science",
        "Robotics",
        "Media",
        "Multi-category",
        "Other",
    ]

    fig_umap_cat = px.scatter(
        umap_df.to_pandas(),
        x="umap_x",
        y="umap_y",
        color="cat_label",
        color_discrete_map=color_map,
        category_orders={"cat_label": cat_order},
        hover_name="title",
        hover_data=["rating_mean", "status", "categories"],
        opacity=0.6,
        title="UMAP: Papers Colored by Selected Category",
    )
    for trace in fig_umap_cat.data:
        if trace.name == "Other":
            trace.marker.opacity = 0.35
            trace.marker.size = 4
            trace.marker.color = "#bdc3c7"
    fig_umap_cat.update_layout(height=700, width=1100)
    fig_umap_cat.write_html(str(FIGURES / "umap_by_category.html"))
    fig_umap_cat
    return (fig_umap_cat,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 9. Cross-Category Bridge Papers

    A paper is a "bridge" if it belongs to 2+ keyword categories **and** sits
    semantically equidistant between Ward clusters (low `bridge_ratio` from
    notebook 4). These are truly interdisciplinary: multi-thematic by keyword
    and multi-cluster by embedding. The best bridges connect research communities
    that rarely overlap.
    """)
    return


@app.cell
def _(CATEGORIES, CAT_NAMES, df, format_paper_cat, membership):
    multi_ids_br = set(
        membership.filter(pl.col("n_categories") >= 2)["openreview_id"].to_list()
    )
    bridge_df = df.filter(
        pl.col("openreview_id").is_in(list(multi_ids_br))
        & pl.col("bridge_ratio").is_not_null()
    ).sort("bridge_ratio")

    id_cats_br: dict[str, list[int]] = {}
    for _idx in sorted(CATEGORIES.keys()):
        for _oid in CATEGORIES[_idx]["df"]["openreview_id"].to_list():
            id_cats_br.setdefault(_oid, []).append(_idx)

    print(f"Papers in 2+ categories with bridge_ratio: {bridge_df.shape[0]}")
    print(
        "\nTop 15 cross-category bridge papers (lowest bridge_ratio = most bridging):"
    )
    for _i, _row in enumerate(bridge_df.head(15).iter_rows(named=True), 1):
        _oid = _row["openreview_id"]
        _cats = ", ".join(CAT_NAMES[_c] for _c in id_cats_br.get(_oid, []))
        br = _row["bridge_ratio"]
        print(
            f"\n{_i}. {format_paper_cat(_row, f'bridge_ratio={br:.3f}, cats=[{_cats}]')}"
        )
    return (bridge_df,)


@app.cell
def _(df, membership):
    bridge_plot = df.join(
        membership.select("openreview_id", "n_categories"),
        on="openreview_id",
    ).filter(pl.col("n_categories") >= 1)

    fig_bridge = px.scatter(
        bridge_plot.to_pandas(),
        x="bridge_ratio",
        y="n_categories",
        color="rating_mean",
        color_continuous_scale="RdYlGn",
        hover_name="title",
        hover_data=["status", "primary_area"],
        opacity=0.5,
        title="Semantic Bridge Ratio vs Number of Categories",
    )
    fig_bridge.update_layout(
        height=500,
        width=800,
        xaxis_title="Bridge Ratio (lower = more bridging)",
        yaxis_title="Number of Categories",
    )
    fig_bridge.write_html(str(FIGURES / "bridge_vs_categories.html"))
    fig_bridge
    return (fig_bridge,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10. Summary

    After separating out benchmark papers, our 8 categories carve distinct slices
    of ICLR 2026, each with its own "personality": Science papers push into novel
    semantic territory but divide reviewers; Safety work sparks the most debate;
    Media and Robotics enjoy strong reviewer consensus and high quality; and
    Agents serve as the conference's connective tissue, bridging the most
    clusters. Multi-category papers — especially those spanning 3-4 categories —
    represent the most interdisciplinary work at the conference.
    """)
    return


@app.cell
def _(CATEGORIES, all_category_picks, membership):
    total = sum(len(_v) for _v in all_category_picks.values())
    unique_ids = set()
    for _pl in all_category_picks.values():
        for _p in _pl:
            unique_ids.add(_p["openreview_id"])
    in_any = membership.filter(pl.col("n_categories") >= 1).shape[0]
    in_multi = membership.filter(pl.col("n_categories") >= 2).shape[0]

    print(f"Categories: {len(CATEGORIES)}")
    print(f"Papers in at least 1 category: {in_any}")
    print(f"Papers in 2+ categories: {in_multi}")
    print(f"Total per-category picks: {total}")
    print(f"Unique picks across all categories: {len(unique_ids)}")
    print(f"Figures saved to: {FIGURES}/")
    return


if __name__ == "__main__":
    app.run()
