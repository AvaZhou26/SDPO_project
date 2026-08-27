"""
Sequential multi-preference online SDPO evaluation (reproduces paper Fig. 6).

Trains a single model online with SDPO across a *curriculum* of user-simulator
styles introduced one after another (e.g. 500 interactions per style). At the
moment each style is introduced, the current model is snapshotted as a frozen
reference point for that style. At every periodic eval round, the *current*
model is compared against every already-introduced style's own frozen
snapshot (via that style's judge) -- producing one winrate curve per style,
each starting at ~50% at its own introduction step and continuing to the end
of training, so you can see whether earlier styles are retained while later
ones are learned.

Usage:
    python eval_online_sdpo_curriculum.py \
        --train_jsonl data/helpsteer_prompts/train.jsonl \
        --val_jsonl data/helpsteer_prompts/validation.jsonl \
        --styles no_emojis less_filler_praise_sycophancy answer_directly_reduce_formatting \
        --steps_per_style 500 \
        --eval_n 256 \
        --eval_every 100 \
        --run_name curriculum_experiment
"""
import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tinker import types
from transformers import AutoTokenizer

from online_sdpo_updater_config import OnlineSDPOConfig
from online_sdpo_updater import OnlineSDPOUpdater
from auxiliary.deepseek_user_simulator import DeepSeekStyleUserSimulator
from auxiliary.deepseek_style_judge import DeepSeekStyleJudge


# ── Metrics (mirrors eval_online_sdpo.py / auxiliary/eval_checkpoints.py) ──
# Duplicated rather than imported, matching this repo's existing convention
# of keeping each eval script's metrics self-contained.

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


def compute_lr(step: int, total_steps: int, base_lr: float, min_lr: float, schedule: str) -> float:
    """LR for the given (0-indexed) training step.

    'flat' (default) always returns base_lr -- identical to prior behavior.
    'cosine' decays from base_lr down to min_lr over the whole run, so the
    LR is highest (fastest adaptation) right when a new style is introduced
    and gentlest (least disruptive to earlier styles) by the end.
    """
    if schedule == "flat":
        return base_lr
    if schedule == "cosine":
        progress = step / max(total_steps - 1, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown --lr_schedule '{schedule}'")


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


# ── Frozen-snapshot sampler ─────────────────────────────────────────────
# Small, self-contained wrapper around a Tinker SamplingClient -- mirrors
# TinkerSamplerWrapper in auxiliary/eval_checkpoints.py, but defined locally
# rather than imported so this script doesn't have to pull in that file's
# unrelated heavy imports (vllm, local judge/simulator modules) just for one
# helper class.

class FrozenSnapshotSampler:
    """Wraps a single frozen SamplingClient snapshot -- lets us generate
    completions from 'the model at the moment a style was introduced' even
    though the live training client has since moved on."""

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
        self, messages_list: List[List[Dict[str, str]]], max_new_tokens: int, temperature: float,
    ) -> List[str]:
        sampling_params = types.SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=1.0)
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


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sequential multi-preference online SDPO evaluation")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lr_schedule", type=str, choices=["flat", "cosine"], default="flat",
                    help="'flat' (default) keeps --lr constant for the whole run, unchanged "
                         "from prior behavior. 'cosine' decays from --lr down to --lr_min over "
                         "the full curriculum, so each newly-introduced style still gets a "
                         "strong initial LR but later steps disturb earlier-learned styles less.")
    p.add_argument("--lr_min", type=float, default=1e-6,
                    help="Floor LR for --lr_schedule cosine. Ignored when --lr_schedule=flat.")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--train_steps_per_example", type=int, default=1)

    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    # Curriculum schedule
    p.add_argument("--styles", type=str, nargs="+", required=True,
                    help="Ordered list of user-simulator styles, introduced one after another.")
    p.add_argument("--steps_per_style", type=int, nargs="+", default=[500],
                    help="Interactions per style. Either one value (applied to every style) "
                         "or one value per style.")

    # Judge
    p.add_argument("--judge_model", type=str, default="deepseek-v4-flash")
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=100)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--eval_max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--save_checkpoints", action="store_true",
                    help="Also persist durable Tinker checkpoints (save_state) at each style "
                         "introduction, independent of the in-memory baseline snapshots used "
                         "for scoring.")

    p.add_argument("--eval_dir", type=str, default="./eval")
    p.add_argument("--run_name", type=str, required=True)

    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.eval_dir, exist_ok=True)

    styles = args.styles
    if len(args.steps_per_style) == 1:
        steps_per_style = args.steps_per_style * len(styles)
    elif len(args.steps_per_style) == len(styles):
        steps_per_style = args.steps_per_style
    else:
        raise ValueError(
            f"--steps_per_style must have length 1 or {len(styles)} (got {len(args.steps_per_style)})"
        )

    # Cumulative step at which each style is introduced (0-indexed into the
    # training loop): style[0] at step 0, style[1] at step steps_per_style[0], etc.
    introduction_steps = [sum(steps_per_style[:k]) for k in range(len(styles))]
    total_steps = sum(steps_per_style)

    print(f"[CURRICULUM] Styles:            {styles}", flush=True)
    print(f"[CURRICULUM] Steps per style:    {steps_per_style}", flush=True)
    print(f"[CURRICULUM] Introduced at step: {introduction_steps}", flush=True)
    print(f"[CURRICULUM] Total interactions: {total_steps}", flush=True)

    # ── 1. Build config + updater ──
    config = OnlineSDPOConfig(
        model_name_or_path=args.model,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        train_steps_per_example=args.train_steps_per_example,
        checkpoint_dir=args.run_name,
        checkpoint_every_n_steps=0,  # durable checkpoints (if any) are saved explicitly below
    )

    # ── 2. Load datasets ──
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dsd = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})
    train_ds = dsd["train"]
    eval_ds = dsd["validation"]

    def add_len(example):
        messages = [{"role": "user", "content": example["prompt"].strip()}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"lengths": len(tok(rendered, add_special_tokens=False)["input_ids"])}

    train_ds = train_ds.map(add_len)
    train_ds = train_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")
    eval_ds = eval_ds.map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(total_steps, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_n, len(eval_ds))))

    print(f"[CURRICULUM] Train size: {len(train_ds)}  Eval size: {len(eval_ds)}", flush=True)

    eval_raw_prompts = [ex["prompt"].strip() for ex in eval_ds]
    eval_messages = [[{"role": "user", "content": p}] for p in eval_raw_prompts]

    # ── 3. Build one simulator + one judge per style ──
    simulators = {s: DeepSeekStyleUserSimulator(style=s, max_tokens=256, temperature=0.0) for s in styles}
    judges = {s: DeepSeekStyleJudge(style=s, model=args.judge_model) for s in styles}

    # ── 4. Initialize the (single, continuously-trained) updater ──
    updater = OnlineSDPOUpdater(config)

    def snapshot_baseline_for(style: str) -> None:
        """Freeze the model's current weights as `style`'s reference point,
        and cache its eval-set completions once (the frozen snapshot never
        changes, so there's no need to re-query it at every later eval round)."""
        print(f"[CURRICULUM] Introducing '{style}' at step {updater.step} — capturing baseline snapshot...", flush=True)
        snapshot_client = updater.training_client.save_weights_and_get_sampling_client()
        snapshot_sampler = FrozenSnapshotSampler(tokenizer=updater.tokenizer, sampling_client=snapshot_client)
        baseline_completions[style] = snapshot_sampler.generate_responses_batch(
            eval_messages, max_new_tokens=args.eval_max_new_tokens, temperature=args.temperature,
        )
        if args.save_checkpoints:
            updater.save_checkpoint(tag=f"introduce_{style}")

    baseline_completions: Dict[str, List[str]] = {}
    eval_history: Dict[str, List[Dict]] = {s: [] for s in styles}
    training_metrics = []

    # style[0]'s baseline is the untrained model, captured before any training happens.
    snapshot_baseline_for(styles[0])

    # ── 5. Single continuous training loop across the whole curriculum ──
    print(f"\n[CURRICULUM] Starting training loop ({total_steps} total interactions)...\n", flush=True)

    for i, example in enumerate(train_ds):
        # A later style becomes active the moment its introduction step is reached.
        if i in introduction_steps[1:]:
            snapshot_baseline_for(styles[introduction_steps.index(i)])

        active_style = styles[max(k for k, s in enumerate(introduction_steps) if s <= i)]

        raw_prompt = example["prompt"].strip()
        messages = [{"role": "user", "content": raw_prompt}]

        response = updater.generate_response(messages)
        feedback = simulators[active_style].generate_feedback([raw_prompt], [response])[0]

        updater.config.learning_rate = compute_lr(
            i, total_steps, args.lr, args.lr_min, args.lr_schedule,
        )
        metrics = updater.train_step(
            messages_before_response=messages,
            assistant_response=response,
            user_follow_up=feedback,
        )
        metrics["active_style"] = active_style
        metrics["lr"] = updater.config.learning_rate
        training_metrics.append(metrics)

        step = metrics["step"]
        print(
            f"[STEP {step:4d}] active_style={active_style}  loss={metrics.get('loss', 0.0):.6f}  "
            f"lr={metrics['lr']:.2e}",
            flush=True,
        )

        # Periodic evaluation: one judge comparison per already-introduced style.
        if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            print(f"\n[EVAL @ step {step}] Generating current-model completions...", flush=True)
            current_completions = updater.generate_responses_batch(
                eval_messages, max_new_tokens=args.eval_max_new_tokens, temperature=args.temperature,
            )

            for style, baseline in baseline_completions.items():
                decisions = judges[style].choose_batch_generated(
                    prompts=eval_raw_prompts,
                    completions_a=current_completions,
                    completions_b=baseline,
                )
                style_metrics = compute_metrics(decisions, bootstrap_seed=args.seed)
                style_metrics["step"] = step
                eval_history[style].append(style_metrics)
                print(
                    f"[EVAL @ step {step}] [{style}] winrate={style_metrics.get('winrate_a', float('nan')):.1f}% "
                    f"±{style_metrics.get('se', float('nan')):.1f}",
                    flush=True,
                )
            print("", flush=True)

    # ── 6. Save results ──
    results = {
        "meta": {
            "model": args.model,
            "styles": styles,
            "steps_per_style": steps_per_style,
            "introduction_steps": dict(zip(styles, introduction_steps)),
            "learning_rate": args.lr,
            "lr_schedule": args.lr_schedule,
            "lr_min": args.lr_min,
            "lora_rank": args.lora_rank,
            "eval_n": len(eval_ds),
            "eval_every": args.eval_every,
            "judge_model": args.judge_model,
            "seed": args.seed,
            "run_name": args.run_name,
        },
        "eval_history": eval_history,
        "training_metrics": training_metrics,
    }

    idx = 1
    while os.path.exists(os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")):
        idx += 1
    out_path = os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[CURRICULUM] Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
