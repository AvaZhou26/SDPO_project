"""
In-context oracle baseline evaluation.

For each style, generates completions from the base model with the user profile
injected as a system prompt ("oracle"), and compares against the same base model
without any system prompt ("baseline"). Evaluates all styles in a single run.

Usage:
    python auxiliary/eval_incontext_oracle.py \
        --val_jsonl data/tldr_prompts_unique/validation.jsonl \
        --model Qwen/Qwen3-8B \
        --judge_model Qwen/Qwen3-32B
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from auxiliary.user_simulator import STYLE_PERSONAS
from auxiliary.vllm_user_simulator import VLLMStyleJudge

# ── Metrics (copied from eval_online_sdpo.py) ──

def _non_tie_outcomes(decisions: List[int]) -> np.ndarray:
    return np.array([1 if d == 0 else 0 for d in decisions if d != -1], dtype=np.int8)


def bootstrap_prop_se(y: np.ndarray, B: int = 10_000, seed: Optional[int] = None) -> float:
    n = int(y.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = y[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def compute_metrics(decisions: List[int], bootstrap_seed: Optional[int] = None) -> Dict:
    n = len(decisions)
    if n == 0:
        return {"n": 0, "coverage": 0.0}

    ties = sum(d == -1 for d in decisions)
    wins_a = sum(d == 0 for d in decisions)
    wins_b = sum(d == 1 for d in decisions)
    n_eff = wins_a + wins_b
    coverage = 1.0 - (ties / n)

    if n_eff == 0:
        return {"n": n, "wins_a": 0, "wins_b": 0, "ties": ties,
                "coverage": coverage, "winrate_a": float("nan"),
                "se": float("nan"), "n_effective": 0}

    p_hat = wins_a / n_eff
    se_analytic = float(np.sqrt(p_hat * (1.0 - p_hat) / n_eff) * 100.0) if n_eff > 1 else 0.0
    se_boot = float(bootstrap_prop_se(_non_tie_outcomes(decisions), seed=bootstrap_seed) * 100.0)

    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "coverage": float(coverage),
        "winrate_a": float(p_hat * 100.0),
        "se": se_analytic,
        "se_bootstrap": se_boot,
        "n_effective": n_eff,
    }


# ── Generation ──

def generate_vllm_batch(llm, tokenizer, messages_list, sampling_params):
    prompts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for m in messages_list
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


CORE_STYLES = [
    "concise_casual_beginner",
    "concise_casual_expert",
    "concise_professional_beginner",
    "concise_professional_expert",
    "detailed_casual_beginner",
    "detailed_casual_expert",
    "detailed_professional_beginner",
    "detailed_professional_expert",
]


def build_oracle_system_prompt(style: str) -> str:
    return (
        "You are a helpful assistant. You are interacting with a user with the following preferences:\n"
        f"{STYLE_PERSONAS[style]}\n"
        "Tailor your responses to match these preferences."
    )


def parse_args():
    p = argparse.ArgumentParser(description="In-context oracle baseline evaluation")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--judge_model", default="Qwen/Qwen3-32B")
    p.add_argument("--val_jsonl", default="data/tldr_prompts_unique/validation.jsonl")
    p.add_argument("--eval_n", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_gpu", type=int, default=0, help="GPU for the generation model")
    p.add_argument("--judge_gpu_start", type=int, default=2, help="First GPU for the judge model")
    p.add_argument("--judge_tp", type=int, default=1, help="Tensor parallelism for judge vLLM")
    p.add_argument("--eval_dir", default="eval")
    p.add_argument("--run_name", default="incontext_oracle")
    p.add_argument("--styles", nargs="*", default=CORE_STYLES)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    print("=== IN-CONTEXT ORACLE EVALUATION ===", flush=True)
    print(f"Model:       {args.model}", flush=True)
    print(f"Judge:       {args.judge_model}", flush=True)
    print(f"Styles:      {args.styles}", flush=True)
    print(f"Eval N:      {args.eval_n}", flush=True)
    print(f"Seed:        {args.seed}", flush=True)
    print()

    # ── 1. Load and filter validation prompts (same logic as eval_online_sdpo.py) ──
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dsd = load_dataset("json", data_files={"validation": args.val_jsonl})
    eval_ds = dsd["validation"]

    def add_len(example):
        user_content = example["prompt"].strip()
        messages = [{"role": "user", "content": user_content}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return {"lengths": len(ids)}

    eval_ds = eval_ds.map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_n, len(eval_ds))))

    eval_raw_prompts = [ex["prompt"].strip() for ex in eval_ds]
    print(f"[ORACLE] Eval size: {len(eval_raw_prompts)}", flush=True)

    # ── 2. Initialize generation vLLM ──
    print(f"[ORACLE] Initializing generation vLLM on GPU {args.gen_gpu}...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gen_gpu)
    gen_llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # ── 3. Generate baseline completions (no system prompt) ──
    baseline_messages = [[{"role": "user", "content": p}] for p in eval_raw_prompts]
    print(f"[ORACLE] Generating baseline completions...", flush=True)
    t0 = time.time()
    baseline_completions = generate_vllm_batch(gen_llm, tokenizer, baseline_messages, sampling_params)
    baseline_avg_len = sum(len(c) for c in baseline_completions) / max(len(baseline_completions), 1)
    print(f"[ORACLE] Baseline done in {time.time() - t0:.1f}s  avg_len={baseline_avg_len:.0f} chars", flush=True)

    # ── 4. Generate oracle completions for each style ──
    oracle_completions = {}
    oracle_avg_lens = {}
    for style in args.styles:
        sys_prompt = build_oracle_system_prompt(style)
        oracle_messages = [
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            for p in eval_raw_prompts
        ]
        print(f"[ORACLE] Generating oracle completions for {style}...", flush=True)
        t0 = time.time()
        completions = generate_vllm_batch(gen_llm, tokenizer, oracle_messages, sampling_params)
        avg_len = sum(len(c) for c in completions) / max(len(completions), 1)
        print(f"[ORACLE]   done in {time.time() - t0:.1f}s  avg_len={avg_len:.0f} chars", flush=True)
        oracle_completions[style] = completions
        oracle_avg_lens[style] = avg_len

    # ── 5. Free generation model, load judge ──
    del gen_llm
    torch.cuda.empty_cache()

    judge_gpu_ids = list(range(args.judge_gpu_start, args.judge_gpu_start + args.judge_tp))
    print(f"[ORACLE] Initializing judge vLLM on GPUs {judge_gpu_ids}...", flush=True)
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in judge_gpu_ids)
    judge_llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.judge_tp,
        gpu_memory_utilization=0.9,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # ── 6. Judge each style ──
    results = {}
    for style in args.styles:
        print(f"[ORACLE] Judging {style}...", flush=True)
        judge = VLLMStyleJudge(
            llm=judge_llm, tokenizer=judge_tok, style=style,
        )
        t0 = time.time()
        decisions = judge.choose_batch_generated(
            prompts=eval_raw_prompts,
            completions_a=oracle_completions[style],
            completions_b=baseline_completions,
        )
        metrics = compute_metrics(decisions, bootstrap_seed=args.seed)
        metrics["judge_time_s"] = round(time.time() - t0, 1)
        results[style] = metrics

        wr = metrics.get("winrate_a", float("nan"))
        se = metrics.get("se", float("nan"))
        print(f"[ORACLE] {style}: winrate={wr:.1f}% ±{se:.1f}  "
              f"wins={metrics.get('wins_a', 0)} losses={metrics.get('wins_b', 0)} ties={metrics.get('ties', 0)}",
              flush=True)

    # ── 7. Save results ──
    output = {
        "meta": {
            "method": "incontext_oracle",
            "model": args.model,
            "judge_model": args.judge_model,
            "eval_n": len(eval_raw_prompts),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "styles": args.styles,
            "run_name": args.run_name,
            "baseline_avg_completion_len": baseline_avg_len,
        },
        "results": results,
        "oracle_avg_lens": oracle_avg_lens,
    }

    os.makedirs(args.eval_dir, exist_ok=True)
    idx = 1
    while os.path.exists(os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")):
        idx += 1
    out_path = os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ORACLE] Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
