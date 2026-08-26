"""
Evaluate saved checkpoints (no training) against a baseline model.

For each checkpoint, generates completions on a validation set and runs
pairwise judging against the baseline for one or more eval styles.

Checkpoints are Tinker-hosted training-state paths (the strings printed as
"[LIVE SDPO] Checkpoint saved -> ..." by OnlineSDPOUpdater.save_checkpoint
during a training run, e.g. "tinker://run-id/weights/step_20"), not local
directories — Tinker only serves its own hosted weights, not arbitrary
local safetensors folders.

Usage:
    python eval_checkpoints.py \
        --checkpoints tinker://run-id/weights/step_10 tinker://run-id/weights/step_20 \
        --baseline_model Qwen/Qwen3-8B \
        --eval_styles no_emojis less_filler_praise_sycophancy \
        --val_jsonl data/helpsteer_prompts/validation.jsonl \
        --eval_n 100 --run_name my_ckpt_eval
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import tinker
from tinker import types
from datasets import load_dataset
from transformers import AutoTokenizer

from auxiliary.vllm_user_simulator import VLLMStyleJudge
from auxiliary.style_judge import StyleJudge
from auxiliary.deepseek_style_judge import DeepSeekStyleJudge


class TinkerSamplerWrapper:
    """Thin sampling-only wrapper around a Tinker SamplingClient.

    Checkpoint evaluation only ever generates text — it never trains — so
    it has no need for a full OnlineSDPOUpdater/TrainingClient, just a
    tokenizer (for chat-template rendering) and a SamplingClient.
    """

    def __init__(self, tokenizer, sampling_client, max_context_length: int = 4096):
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client
        self.max_context_length = max_context_length

    def _render_prompt_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self.tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=self.max_context_length,
        )["input_ids"]

    def generate_responses_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        sampling_params = types.SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=1.0,
        )
        # sample() (not sample_async()) is correct: this SDK's plain-named
        # methods are synchronous and return a ConcurrentFuture immediately,
        # while *_async methods are real asyncio coroutines.
        futures = [
            self.sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=self._render_prompt_ids(m)),
                sampling_params=sampling_params,
                num_samples=1,
            )
            for m in messages_list
        ]
        results = [f.result() for f in futures]
        return [
            self.tokenizer.decode(r.sequences[0].tokens, skip_special_tokens=True)
            for r in results
        ]


# ── Metrics (shared with eval_online_sdpo.py) ────────────────────────

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


# ── Evaluation helpers ────────────────────────────────────────────────

def generate_eval_completions(
    updater: TinkerSamplerWrapper,
    eval_messages: List[List[Dict[str, str]]],
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
) -> List[str]:
    return updater.generate_responses_batch(
        eval_messages, max_new_tokens=max_new_tokens, temperature=temperature,
    )

def run_evaluation(
    judge,
    raw_prompts: List[str],
    completions_current: List[str],
    completions_baseline: List[str],
    bootstrap_seed: Optional[int] = None,
) -> Dict:
    decisions = judge.choose_batch_generated(
        prompts=raw_prompts,
        completions_a=completions_current,
        completions_b=completions_baseline,
    )
    metrics = compute_metrics(decisions, bootstrap_seed=bootstrap_seed)
    metrics["decisions"] = decisions
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved checkpoints against a baseline")

    # Checkpoints
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help="Tinker training-state paths to evaluate, e.g. "
                         "tinker://run-id/weights/step_20 (printed by "
                         "OnlineSDPOUpdater.save_checkpoint during training)")
    p.add_argument("--baseline_model", type=str, required=True,
                    help="Baseline base model name known to Tinker (e.g. Qwen/Qwen3-8B)")

    # Eval styles
    p.add_argument("--eval_styles", type=str, nargs="+", required=True,
                    help="Style(s) to evaluate (e.g. no_emojis less_filler_praise_sycophancy)")

    # Dataset
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--eval_n", type=int, default=100)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=424242)

    # Generation
    p.add_argument("--eval_max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)

    # Judge
    p.add_argument("--judge_model", type=str, default="deepseek-v4-flash")
    p.add_argument("--judge_local", action="store_true",
                    help="Use local judge (requires --user_model_name_or_path)")
    p.add_argument("--user_model_name_or_path", type=str, default=None,
                    help="Model for local judge")
    p.add_argument("--user_vllm", action="store_true",
                    help="Use vLLM for the judge model")
    p.add_argument("--user_vllm_tp", type=int, default=1)
    p.add_argument("--user_vllm_gpu_start", type=int, default=2)

    # Output
    p.add_argument("--out_dir", type=str, default="./eval",
                    help="Directory for results JSON")
    p.add_argument("--run_name", type=str, required=True)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load dataset ──
    # Use baseline model tokenizer for prompt length filtering
    tok = AutoTokenizer.from_pretrained(
        args.baseline_model, use_fast=True, padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dsd = load_dataset("json", data_files={"validation": args.val_jsonl})
    eval_ds = dsd["validation"]

    def add_len(example):
        user_content = example["prompt"].strip()
        messages = [{"role": "user", "content": user_content}]
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        return {"lengths": len(ids)}

    eval_ds = eval_ds.map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_n, len(eval_ds))))

    eval_raw_prompts = [ex["prompt"].strip() for ex in eval_ds]
    eval_messages = [[{"role": "user", "content": p}] for p in eval_raw_prompts]

    print(f"[CKPT-EVAL] Eval size:       {len(eval_ds)}", flush=True)
    print(f"[CKPT-EVAL] Baseline:        {args.baseline_model}", flush=True)
    print(f"[CKPT-EVAL] Checkpoints:     {args.checkpoints}", flush=True)
    print(f"[CKPT-EVAL] Eval styles:     {args.eval_styles}", flush=True)

    # ── 2. Generate baseline completions ──
    print(f"\n[CKPT-EVAL] Connecting to Tinker for baseline model: {args.baseline_model}", flush=True)
    service_client = tinker.ServiceClient()
    baseline_sampling_client = service_client.create_sampling_client(base_model=args.baseline_model)
    baseline_updater = TinkerSamplerWrapper(tokenizer=tok, sampling_client=baseline_sampling_client)

    print(f"[CKPT-EVAL] Generating baseline completions ({len(eval_messages)} prompts)...", flush=True)
    t0 = time.time()
    baseline_completions = generate_eval_completions(
        baseline_updater, eval_messages,
        max_new_tokens=args.eval_max_new_tokens, temperature=args.temperature,
    )
    baseline_avg_len = sum(len(c) for c in baseline_completions) / max(len(baseline_completions), 1)
    print(f"[CKPT-EVAL] Baseline generation took {time.time() - t0:.1f}s  avg_len={baseline_avg_len:.0f} chars", flush=True)

    del baseline_updater

    # ── 3. Initialize judges (one per eval style) ──
    from transformers import AutoModelForCausalLM
    user_llm = None
    judges = {}

    if args.judge_local:
        if args.user_model_name_or_path is None:
            raise ValueError("--judge_local requires --user_model_name_or_path")

        user_tok = AutoTokenizer.from_pretrained(args.user_model_name_or_path, use_fast=True)
        if user_tok.pad_token is None:
            user_tok.pad_token = user_tok.eos_token

        if args.user_vllm:
            from vllm import LLM
            gpu_ids = list(range(args.user_vllm_gpu_start, args.user_vllm_gpu_start + args.user_vllm_tp))
            print(f"[CKPT-EVAL] Initializing vLLM judge: {args.user_model_name_or_path} "
                  f"(tp={args.user_vllm_tp}, GPUs={gpu_ids})", flush=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            user_llm = LLM(
                model=args.user_model_name_or_path,
                tensor_parallel_size=args.user_vllm_tp,
                gpu_memory_utilization=0.9,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            for style in args.eval_styles:
                judges[style] = VLLMStyleJudge(llm=user_llm, tokenizer=user_tok, style=style)
        else:
            user_hf = AutoModelForCausalLM.from_pretrained(
                args.user_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
            user_hf.eval()
            for style in args.eval_styles:
                judges[style] = StyleJudge(
                    model=user_hf, tokenizer=user_tok,
                    device=next(user_hf.parameters()).device, style=style,
                )
    else:
        for style in args.eval_styles:
            judges[style] = DeepSeekStyleJudge(style=style, model=args.judge_model)

    print(f"[CKPT-EVAL] Judges initialized for styles: {list(judges.keys())}", flush=True)

    # ── 4. Evaluate each checkpoint ──
    all_results = []

    for ckpt_path in args.checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n{'='*60}", flush=True)
        print(f"[CKPT-EVAL] Evaluating checkpoint: {ckpt_path}", flush=True)
        print(f"{'='*60}", flush=True)

        # Restore the checkpoint's weights and get a sampling-only client.
        # No optimizer state is needed since we're only generating, not resuming training.
        # create_training_client_from_state() (not the _async variant) returns the TrainingClient directly (no .result() needed)
        ckpt_training_client = service_client.create_training_client_from_state(
            path=ckpt_path,
        )
        ckpt_sampling_client = ckpt_training_client.save_weights_and_get_sampling_client()
        ckpt_updater = TinkerSamplerWrapper(tokenizer=tok, sampling_client=ckpt_sampling_client)

        # Generate completions
        print(f"[CKPT-EVAL] Generating completions for {ckpt_name} ({len(eval_messages)} prompts)...", flush=True)
        t0 = time.time()
        ckpt_completions = generate_eval_completions(
            ckpt_updater, eval_messages,
            max_new_tokens=args.eval_max_new_tokens, temperature=args.temperature,
        )
        gen_time = time.time() - t0
        avg_len = sum(len(c) for c in ckpt_completions) / max(len(ckpt_completions), 1)
        print(f"[CKPT-EVAL] Generation took {gen_time:.1f}s  avg_len={avg_len:.0f} chars (baseline={baseline_avg_len:.0f})", flush=True)

        # Judge for each eval style
        ckpt_result = {
            "checkpoint": ckpt_path,
            "checkpoint_name": ckpt_name,
            "gen_time_s": round(gen_time, 1),
            "avg_completion_len": round(avg_len, 0),
            "eval_styles": {},
        }

        for style, judge in judges.items():
            print(f"[CKPT-EVAL] Judging {ckpt_name} with style '{style}' ({len(eval_raw_prompts)} comparisons)...", flush=True)
            t_judge = time.time()
            eval_metrics = run_evaluation(
                judge, eval_raw_prompts, ckpt_completions,
                baseline_completions, bootstrap_seed=args.seed,
            )
            judge_time = time.time() - t_judge
            eval_metrics.pop("decisions", None)
            eval_metrics["judge_time_s"] = round(judge_time, 1)

            wr = eval_metrics.get("winrate_a", float("nan"))
            se = eval_metrics.get("se", float("nan"))
            wr_inc_ties = (eval_metrics['wins_a'] + 0.5 * eval_metrics['ties']) / max(eval_metrics['n'], 1) * 100.0
            eval_metrics["winrate_inc_ties"] = round(wr_inc_ties, 2)

            print(
                f"  [{style}] winrate={wr:.1f}% +/-{se:.1f}  "
                f"wins={eval_metrics['wins_a']} losses={eval_metrics['wins_b']} "
                f"ties={eval_metrics['ties']}  wr_inc_ties={wr_inc_ties:.1f}%  "
                f"judge={judge_time:.1f}s",
                flush=True,
            )
            ckpt_result["eval_styles"][style] = eval_metrics

        all_results.append(ckpt_result)

        del ckpt_updater

    # ── 5. Save results ──
    results = {
        "meta": {
            "baseline_model": args.baseline_model,
            "checkpoints": args.checkpoints,
            "eval_styles": args.eval_styles,
            "eval_n": len(eval_ds),
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "judge_local": args.judge_local,
            "judge_model": args.judge_model if not args.judge_local else "local",
            "user_model": args.user_model_name_or_path,
            "run_name": args.run_name,
            "baseline_avg_completion_len": round(baseline_avg_len, 0),
        },
        "results": all_results,
    }

    # Auto-number output file
    idx = 1
    while os.path.exists(os.path.join(args.out_dir, f"{args.run_name}_{idx}.json")):
        idx += 1
    out_path = os.path.join(args.out_dir, f"{args.run_name}_{idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[CKPT-EVAL] Results saved -> {out_path}", flush=True)

    # Print summary table
    print(f"\n{'='*60}", flush=True)
    print(f"[CKPT-EVAL] SUMMARY (vs {args.baseline_model})", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        print(f"\n  {r['checkpoint_name']}:")
        for style, m in r["eval_styles"].items():
            wr = m.get("winrate_a", float("nan"))
            se = m.get("se", float("nan"))
            wr_inc = m.get("winrate_inc_ties", float("nan"))
            print(f"    {style:40s}  winrate={wr:5.1f}% +/-{se:4.1f}  (inc_ties={wr_inc:5.1f}%)")
    print(flush=True)


if __name__ == "__main__":
    main()
