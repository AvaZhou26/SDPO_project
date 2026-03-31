"""
Automated evaluation of OnlineSDPOUpdater on dataset prompts.

Simulates the live chat flow: for each training prompt, generate a response,
get simulated user feedback, and run one SDPO training step. Periodically
evaluates the current model against the pre-training baseline using a judge.

Usage:
    python eval_online_sdpo.py \
        --train_jsonl data/helpsteer_prompts/train.jsonl \
        --val_jsonl data/helpsteer_prompts/validation.jsonl \
        --style concise_casual_beginner \
        --run_name my_experiment
"""
import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from online_sdpo_updater_config import OnlineSDPOConfig
from online_sdpo_updater import OnlineSDPOUpdater
from auxiliary.claude_user_simulator import ClaudeStyleUserSimulator
from auxiliary.user_simulator import StyleUserSimulator
from auxiliary.claude_style_judge import ClaudeStyleJudge
from auxiliary.style_judge import StyleJudge
from auxiliary.vllm_user_simulator import VLLMStyleUserSimulator, VLLMStyleJudge



# ── Metrics (from eval_style_pairwise_accelerate.py) ────────────────────

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


# ── Evaluation helpers ──────────────────────────────────────────────────

def generate_eval_completions(
    updater: OnlineSDPOUpdater,
    eval_messages: List[List[Dict[str, str]]],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
) -> List[str]:
    """Generate completions for all eval prompts using the current model."""
    # Batched path (vLLM)
    if updater.config.use_vllm:
        return updater.generate_responses_batch(
            eval_messages, max_new_tokens=max_new_tokens, temperature=temperature,
        )

    # Sequential fallback (HF generate)
    original_config = updater.generation_config
    updater.generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=1.0,
        eos_token_id=updater.tokenizer.eos_token_id,
        pad_token_id=updater.tokenizer.pad_token_id,
    )
    completions = []
    for messages in eval_messages:
        completions.append(updater.generate_response(messages))
    updater.generation_config = original_config
    return completions


def save_completions_snapshot(
    step: int,
    raw_prompts: List[str],
    current_completions: List[str],
    baseline_completions: List[str],
    run_name: str,
    max_chars: int = 300,
) -> None:
    """Append truncated completions to a JSONL file in ~/output/sdpo/."""
    out_dir = os.path.join(os.path.expanduser("~"), "output", "sdpo")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}_completions.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for prompt, current, baseline in zip(
            raw_prompts[:10], current_completions[:10], baseline_completions[:10],
        ):
            f.write(json.dumps({
                "step": step,
                "prompt": prompt[:max_chars],
                "current": current[:max_chars],
                "baseline": baseline[:max_chars],
            }, ensure_ascii=False) + "\n")


def run_evaluation(
    judge,
    raw_prompts: List[str],
    completions_current: List[str],
    completions_baseline: List[str],
    bootstrap_seed: Optional[int] = None,
) -> Dict:
    """Run pairwise judge comparison, return metrics dict with per-example decisions."""
    decisions = judge.choose_batch_generated(
        prompts=raw_prompts,
        completions_a=completions_current,
        completions_b=completions_baseline,
    )
    metrics = compute_metrics(decisions, bootstrap_seed=bootstrap_seed)
    metrics["decisions"] = decisions  # per-example: 0=current wins, 1=baseline wins, -1=tie
    return metrics


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OnlineSDPOUpdater on dataset prompts")

    # Model + training
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=256)
    p.add_argument("--loss_mode", type=str, default="full_distillation",
                    choices=["simple_signal", "full_distillation"])
    p.add_argument("--distillation_topk", type=int, default=20)
    p.add_argument("--use_vllm", action="store_true")

    # Dataset
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--train_n", type=int, default=10)
    p.add_argument("--eval_n", type=int, default=50)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    # User simulator
    p.add_argument("--style", type=str, default="concise_casual_beginner")
    p.add_argument("--user_model_name_or_path", type=str, default=None,
                    help="If set, use local StyleUserSimulator. Else uses Claude API.")

    # Judge
    p.add_argument("--judge_model", type=str, default="claude-haiku-4-5-20251001")
    p.add_argument("--judge_local", action="store_true",
                    help="Use local StyleJudge (shares model with user sim) instead of Claude API.")

    # Secondary eval styles
    p.add_argument("--eval_styles", type=str, nargs="*", default=[],
                    help="Additional styles to evaluate against (secondary winrates). "
                         "E.g. --eval_styles poetic no_emojis")

    # vLLM for user sim / judge
    p.add_argument("--user_vllm", action="store_true",
                    help="Use vLLM for the user sim and judge model (requires --user_model_name_or_path and --judge_local).")
    p.add_argument("--user_vllm_tp", type=int, default=1,
                    help="Tensor parallel size for the user sim / judge vLLM instance (default: 1).")
    p.add_argument("--user_vllm_gpu_start", type=int, default=2,
                    help="First GPU id for user sim / judge vLLM (default: 2).")

    # Baseline model for evaluation
    p.add_argument("--baseline_model", type=str, default=None,
                    help="Model to generate baseline completions for winrate comparison. "
                         "Defaults to the --model before any training updates.")

    # Evaluation schedule
    p.add_argument("--train_steps_per_example", type=int, default=1,
                   help="K gradient steps per interaction (default: 1)")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_max_new_tokens", type=int, default=2048)

    # Generation (training)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)

    # Checkpointing
    p.add_argument("--save_checkpoints", action="store_true",
                    help="Save a model checkpoint after every training interaction (default: off)")

    # Output
    p.add_argument("--out_dir", type=str, default="./live_eval_results",
                    help="Scratch dir for checkpoints and large artifacts")
    p.add_argument("--eval_dir", type=str, default="./eval",
                    help="Dir for results JSON (persisted in home dir)")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--debug_first_example", action="store_true",
                    help="Print full formatted inputs for the first simulator and judge call")

    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    # ── 1. Build config ──
    config = OnlineSDPOConfig(
        model_name_or_path=args.model,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        loss_mode=args.loss_mode,
        distillation_topk=args.distillation_topk,
        use_vllm=args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_to_wandb=args.wandb,
        wandb_project="live-sdpo-eval",
        wandb_run_name=args.run_name,
        checkpoint_every_n_steps=1 if args.save_checkpoints else 0,
        checkpoint_dir=os.path.join(args.out_dir, "checkpoints"),
        max_checkpoints=999,  # keep all checkpoints
        train_steps_per_example=args.train_steps_per_example,
    )

    print(f"[EVAL] Model:      {config.model_name_or_path}", flush=True)
    print(f"[EVAL] LoRA:       {config.use_lora} (r={config.lora_r})", flush=True)
    print(f"[EVAL] LR:         {config.learning_rate}", flush=True)
    print(f"[EVAL] Loss mode:  {config.loss_mode}", flush=True)
    print(f"[EVAL] vLLM:       {config.use_vllm}", flush=True)
    print(f"[EVAL] Steps/ex:   {config.train_steps_per_example}", flush=True)
    print(f"[EVAL] Style:      {args.style}", flush=True)

    # ── 2. Load datasets ──
    is_local = os.path.isdir(config.model_name_or_path)
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True, padding_side="left", local_files_only=is_local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dsd = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})
    train_ds = dsd["train"]
    eval_ds = dsd["validation"]

    def add_len(example):
        user_content = example["prompt"].strip()
        messages = [{"role": "user", "content": user_content}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        return {"lengths": len(ids)}

    train_ds = train_ds.map(add_len)
    train_ds = train_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")
    eval_ds = eval_ds.map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_n, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_n, len(eval_ds))))

    print(f"[EVAL] Train size: {len(train_ds)}", flush=True)
    print(f"[EVAL] Eval size:  {len(eval_ds)}", flush=True)

    # Pre-build eval messages and raw prompts (for simulator/judge)
    eval_raw_prompts = [ex["prompt"].strip() for ex in eval_ds]
    eval_messages = [
        [{"role": "user", "content": ex["prompt"].strip()}]
        for ex in eval_ds
    ]

    # ── 3. Generate baseline completions (before loading other models to avoid OOM) ──
    if args.baseline_model is not None and args.baseline_model != args.model:
        print(f"[EVAL] Loading separate baseline model: {args.baseline_model}", flush=True)
        baseline_config = OnlineSDPOConfig(
            model_name_or_path=args.baseline_model,
            use_vllm=args.use_vllm,
            use_lora=False,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        baseline_updater = OnlineSDPOUpdater(baseline_config)
        print(f"[EVAL] Generating baseline completions from {args.baseline_model} ({len(eval_messages)} prompts)...", flush=True)
        t0 = time.time()
        baseline_completions = generate_eval_completions(
            baseline_updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
            temperature=args.temperature,
        )
        baseline_avg_len = sum(len(c) for c in baseline_completions) / max(len(baseline_completions), 1)
        print(f"[EVAL] Baseline generation took {time.time() - t0:.1f}s  avg_len={baseline_avg_len:.0f} chars", flush=True)
        # Free baseline model memory before loading main model + user sim
        del baseline_updater
        torch.cuda.empty_cache()
        baseline_generated = True
    else:
        baseline_generated = False

    # ── 4. Initialize updater ──
    print("[EVAL] Initializing OnlineSDPOUpdater...", flush=True)
    updater = OnlineSDPOUpdater(config)

    # Generate baseline from main model (before training) if no separate baseline
    if not baseline_generated:
        print(f"[EVAL] Generating baseline completions ({len(eval_messages)} prompts)...", flush=True)
        t0 = time.time()
        baseline_completions = generate_eval_completions(
            updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
            temperature=args.temperature,
        )
        baseline_avg_len = sum(len(c) for c in baseline_completions) / max(len(baseline_completions), 1)
        print(f"[EVAL] Baseline generation took {time.time() - t0:.1f}s  avg_len={baseline_avg_len:.0f} chars", flush=True)

    # ── 5. Initialize user simulator ──
    user_llm = None  # shared vLLM instance for user sim + judge
    if args.user_model_name_or_path is not None and args.user_vllm:
        # vLLM-backed user simulator
        from vllm import LLM
        user_tok = AutoTokenizer.from_pretrained(args.user_model_name_or_path, use_fast=True)
        if user_tok.pad_token is None:
            user_tok.pad_token = user_tok.eos_token
        gpu_ids = list(range(args.user_vllm_gpu_start, args.user_vllm_gpu_start + args.user_vllm_tp))
        print(f"[EVAL] Initializing vLLM for user sim/judge: {args.user_model_name_or_path} "
              f"(tp={args.user_vllm_tp}, GPUs={gpu_ids})", flush=True)
        # Pin user sim vLLM to its designated GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        user_llm = LLM(
            model=args.user_model_name_or_path,
            tensor_parallel_size=args.user_vllm_tp,
            gpu_memory_utilization=0.9,
        )
        # Restore full GPU visibility for other components
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        simulator = VLLMStyleUserSimulator(
            llm=user_llm, tokenizer=user_tok, style=args.style,
        )
    elif args.user_model_name_or_path is not None:
        print(f"[EVAL] Loading local user simulator: {args.user_model_name_or_path}", flush=True)
        # Pin user model to GPU 2 when vLLM is active (GPU 0=vLLM, GPU 1=training)
        user_device_map = {"": "cuda:2"} if args.use_vllm else "auto"
        user_hf = AutoModelForCausalLM.from_pretrained(
            args.user_model_name_or_path, torch_dtype=torch.bfloat16, device_map=user_device_map,
        )
        user_hf.eval()
        user_tok = AutoTokenizer.from_pretrained(args.user_model_name_or_path, use_fast=True)
        if user_tok.pad_token is None:
            user_tok.pad_token = user_tok.eos_token
        simulator = StyleUserSimulator(
            model=user_hf, tokenizer=user_tok,
            device=next(user_hf.parameters()).device, style=args.style,
        )
    else:
        print("[EVAL] Using Claude user simulator.", flush=True)
        simulator = ClaudeStyleUserSimulator(style=args.style, max_tokens=256, temperature=0.0)

    # ── 6. Initialize judge ──
    if args.judge_local:
        if args.user_model_name_or_path is None:
            raise ValueError("--judge_local requires --user_model_name_or_path (shares the same model)")
        if user_llm is not None:
            print(f"[EVAL] Using VLLMStyleJudge (shared vLLM engine with user sim)", flush=True)
            judge = VLLMStyleJudge(
                llm=user_llm, tokenizer=user_tok, style=args.style,
            )
        else:
            print(f"[EVAL] Using local StyleJudge (shared with user sim)", flush=True)
            judge = StyleJudge(
                model=user_hf, tokenizer=user_tok,
                device=next(user_hf.parameters()).device, style=args.style,
            )
    else:
        print(f"[EVAL] Using ClaudeStyleJudge ({args.judge_model})", flush=True)
        judge = ClaudeStyleJudge(style=args.style, model=args.judge_model)

    # ── 6b. Initialize secondary judges for additional eval styles ──
    secondary_judges = {}
    for sec_style in args.eval_styles:
        if sec_style == args.style:
            continue  # skip duplicate of primary
        if args.judge_local:
            if user_llm is not None:
                secondary_judges[sec_style] = VLLMStyleJudge(
                    llm=user_llm, tokenizer=user_tok, style=sec_style,
                )
            else:
                secondary_judges[sec_style] = StyleJudge(
                    model=user_hf, tokenizer=user_tok,
                    device=next(user_hf.parameters()).device, style=sec_style,
                )
        else:
            secondary_judges[sec_style] = ClaudeStyleJudge(
                style=sec_style, model=args.judge_model,
            )
    if secondary_judges:
        print(f"[EVAL] Secondary eval styles: {list(secondary_judges.keys())}", flush=True)

    # ── 7. Training loop ──
    training_metrics = []
    eval_history = []
    eval_completions_history = []  # per-eval-step: {step, current, baseline}

    print(f"\n[EVAL] Starting training loop ({len(train_ds)} steps)...\n", flush=True)

    for i, example in enumerate(train_ds):
        raw_prompt = example["prompt"].strip()
        messages = [{"role": "user", "content": raw_prompt}]

        # Generate response
        t0 = time.time()
        response = updater.generate_response(messages)
        gen_time = time.time() - t0

        # Get simulated user feedback
        t1 = time.time()
        feedback = simulator.generate_feedback([raw_prompt], [response])[0]
        sim_time = time.time() - t1
        if i == 0 and args.debug_first_example:
            print("=" * 60, flush=True)
            print("[DEBUG] User simulator — first example", flush=True)
            if hasattr(simulator, "_build_prompt_text"):
                full_text = simulator._build_prompt_text(raw_prompt, response)
                print(f"  FULL FORMATTED INPUT:\n{full_text[:2000]}", flush=True)
            else:
                print(f"  SYSTEM:\n{simulator.system_persona}", flush=True)
                print(f"  INPUT prompt:\n{raw_prompt[:500]}", flush=True)
                print(f"  INPUT completion:\n{response[:500]}", flush=True)
            print(f"  OUTPUT feedback:\n{feedback}", flush=True)
            print("=" * 60, flush=True)
        print(f"  feedback: {feedback[:]!r}", flush=True)

        # Train step
        metrics = updater.train_step(
            messages_before_response=messages,
            assistant_response=response,
            user_follow_up=feedback,
        )
        metrics["gen_time_s"] = round(gen_time, 2)
        metrics["sim_time_s"] = round(sim_time, 2)
        metrics["prompt"] = raw_prompt[:300]
        metrics["response"] = response[:300]
        metrics["feedback"] = feedback[:300]
        training_metrics.append(metrics)

        # Print training metrics
        step = metrics["step"]
        loss = metrics.get("loss", 0.0)
        logp = metrics.get("policy_logp_mean", 0.0)
        train_t = metrics.get("train_time_s", 0.0)
        print(
            f"[STEP {step:4d}] loss={loss:.6f}  policy_logp={logp:.4f}  "
            f"gen={gen_time:.1f}s  sim={sim_time:.1f}s  train={train_t:.1f}s",
            flush=True,
        )

        # Periodic evaluation
        if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            print(f"\n[EVAL @ step {step}] Generating eval completions...", flush=True)
            t_eval = time.time()
            current_completions = generate_eval_completions(
                updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
                temperature=args.temperature,
            )
            eval_gen_time = time.time() - t_eval
            current_avg_len = sum(len(c) for c in current_completions) / max(len(current_completions), 1)
            print(f"[EVAL @ step {step}] avg_len={current_avg_len:.0f} chars (baseline={baseline_avg_len:.0f})", flush=True)

            print(f"[EVAL @ step {step}] Running judge ({len(eval_raw_prompts)} comparisons)...", flush=True)
            t_judge = time.time()
            eval_metrics = run_evaluation(
                judge, eval_raw_prompts, current_completions,
                baseline_completions, bootstrap_seed=args.seed,
            )
            judge_time = time.time() - t_judge

            eval_metrics["step"] = step
            eval_metrics["eval_gen_time_s"] = round(eval_gen_time, 1)
            eval_metrics["judge_time_s"] = round(judge_time, 1)
            if not eval_history and args.debug_first_example:
                print("=" * 60, flush=True)
                print("[DEBUG] Judge — first eval, first example", flush=True)
                print(f"  SYSTEM:\n{judge.system_persona}", flush=True)
                if hasattr(judge, "_build_prompt_text"):
                    full_text = judge._build_prompt_text(
                        eval_raw_prompts[0], current_completions[0], baseline_completions[0],
                    )
                    print(f"  FULL FORMATTED INPUT:\n{full_text[:2500]}", flush=True)
                else:
                    print(f"  INPUT prompt:\n{eval_raw_prompts[0][:500]}", flush=True)
                    print(f"  INPUT completion_a (current):\n{current_completions[0][:500]}", flush=True)
                    print(f"  INPUT completion_b (baseline):\n{baseline_completions[0][:500]}", flush=True)
                d = eval_metrics.get("decisions", [])
                print(f"  OUTPUT decision[0]: {d[0] if d else 'N/A'}", flush=True)
                raw = getattr(judge, "last_raw_outputs", None)
                if raw:
                    print(f"  JUDGE REASONING[0]:\n{raw[0][:1000]}", flush=True)
                print("=" * 60, flush=True)
            eval_history.append(eval_metrics)

            wr = eval_metrics.get("winrate_a", float("nan"))
            se = eval_metrics.get("se", float("nan"))
            wr_inc_ties = (eval_metrics['wins_a'] + 0.5 * eval_metrics['ties']) / max(eval_metrics['n'], 1) * 100.0
            print(
                f"[EVAL @ step {step}] winrate={wr:.1f}% ±{se:.1f}  "
                f"wins={eval_metrics['wins_a']} losses={eval_metrics['wins_b']} "
                f"ties={eval_metrics['ties']}  "
                f"wr_inc_ties={wr_inc_ties:.1f}%  "
                f"gen={eval_gen_time:.1f}s  judge={judge_time:.1f}s",
                flush=True,
            )

            # Secondary style evaluations (reuse same completions)
            for sec_style, sec_judge in secondary_judges.items():
                sec_metrics = run_evaluation(
                    sec_judge, eval_raw_prompts, current_completions,
                    baseline_completions, bootstrap_seed=args.seed,
                )
                sec_wr = sec_metrics.get("winrate_a", float("nan"))
                sec_se = sec_metrics.get("se", float("nan"))
                sec_wr_inc = (sec_metrics['wins_a'] + 0.5 * sec_metrics['ties']) / max(sec_metrics['n'], 1) * 100.0
                print(
                    f"  [secondary: {sec_style}] winrate={sec_wr:.1f}% ±{sec_se:.1f}  "
                    f"wins={sec_metrics['wins_a']} losses={sec_metrics['wins_b']} "
                    f"ties={sec_metrics['ties']}  wr_inc_ties={sec_wr_inc:.1f}%",
                    flush=True,
                )
                sec_metrics.pop("decisions", None)
                eval_metrics[f"secondary_{sec_style}"] = sec_metrics
            print(flush=True)

            save_completions_snapshot(
                step, eval_raw_prompts, current_completions,
                baseline_completions, run_name=args.run_name,
            )
            eval_completions_history.append({
                "step": step,
                "current": current_completions,
                "baseline": baseline_completions,
            })

    # ── 8. Final evaluation ──
    final_step = updater.step
    if not eval_history or eval_history[-1]["step"] != final_step:
        print(f"\n[EVAL FINAL @ step {final_step}] Generating eval completions...", flush=True)
        t_eval = time.time()
        current_completions = generate_eval_completions(
            updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
        )
        eval_gen_time = time.time() - t_eval
        current_avg_len = sum(len(c) for c in current_completions) / max(len(current_completions), 1)
        print(f"[EVAL FINAL @ step {final_step}] avg_len={current_avg_len:.0f} chars (baseline={baseline_avg_len:.0f})", flush=True)

        print(f"[EVAL FINAL @ step {final_step}] Running judge...", flush=True)
        t_judge = time.time()
        eval_metrics = run_evaluation(
            judge, eval_raw_prompts, current_completions,
            baseline_completions, bootstrap_seed=args.seed,
        )
        judge_time = time.time() - t_judge

        eval_metrics["step"] = final_step
        eval_metrics["eval_gen_time_s"] = round(eval_gen_time, 1)
        eval_metrics["judge_time_s"] = round(judge_time, 1)
        eval_history.append(eval_metrics)

        wr = eval_metrics.get("winrate_a", float("nan"))
        se = eval_metrics.get("se", float("nan"))
        wr_inc_ties = (eval_metrics['wins_a'] + 0.5 * eval_metrics['ties']) / max(eval_metrics['n'], 1) * 100.0
        print(
            f"[EVAL FINAL @ step {final_step}] winrate={wr:.1f}% ±{se:.1f}  "
            f"wins={eval_metrics['wins_a']} losses={eval_metrics['wins_b']} "
            f"ties={eval_metrics['ties']}  "
            f"wr_inc_ties={wr_inc_ties:.1f}%",
            flush=True,
        )

        for sec_style, sec_judge in secondary_judges.items():
            sec_metrics = run_evaluation(
                sec_judge, eval_raw_prompts, current_completions,
                baseline_completions, bootstrap_seed=args.seed,
            )
            sec_wr = sec_metrics.get("winrate_a", float("nan"))
            sec_se = sec_metrics.get("se", float("nan"))
            sec_wr_inc = (sec_metrics['wins_a'] + 0.5 * sec_metrics['ties']) / max(sec_metrics['n'], 1) * 100.0
            print(
                f"  [secondary: {sec_style}] winrate={sec_wr:.1f}% ±{sec_se:.1f}  "
                f"wins={sec_metrics['wins_a']} losses={sec_metrics['wins_b']} "
                f"ties={sec_metrics['ties']}  wr_inc_ties={sec_wr_inc:.1f}%",
                flush=True,
            )
            sec_metrics.pop("decisions", None)
            eval_metrics[f"secondary_{sec_style}"] = sec_metrics
        print(flush=True)

        save_completions_snapshot(
            final_step, eval_raw_prompts, current_completions,
            baseline_completions, run_name=args.run_name,
        )
        eval_completions_history.append({
            "step": final_step,
            "current": current_completions,
            "baseline": baseline_completions,
        })

    # ── 9. Save results ──
    results = {
        "meta": {
            "model": args.model,
            "style": args.style,
            "loss_mode": args.loss_mode,
            "distillation_topk": args.distillation_topk,
            "learning_rate": args.lr,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "use_vllm": args.use_vllm,
            "train_n": len(train_ds),
            "eval_n": len(eval_ds),
            "eval_every": args.eval_every,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "train_steps_per_example": args.train_steps_per_example,
            "judge_model": args.judge_model if not args.judge_local else "local",
            "judge_local": args.judge_local,
            "baseline_model": args.baseline_model or args.model,
            "user_model": args.user_model_name_or_path,
            "run_name": args.run_name,
            "eval_styles": args.eval_styles,
        },
        "eval_history": eval_history,
        "training_metrics": training_metrics,
    }

    # Strip per-example decisions from eval_history (only keep aggregate metrics)
    for entry in results["eval_history"]:
        entry.pop("decisions", None)

    # Auto-number: find next available index for this run_name
    os.makedirs(args.eval_dir, exist_ok=True)
    idx = 1
    while os.path.exists(os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")):
        idx += 1
    out_path = os.path.join(args.eval_dir, f"{args.run_name}_{idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] Results saved → {out_path}", flush=True)

    # Save completions separately (current + baseline only, no prompts)
    completions_path = os.path.join(args.eval_dir, f"{args.run_name}_completions_{idx}.json")
    with open(completions_path, "w") as f:
        json.dump(eval_completions_history, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] Completions saved → {completions_path}", flush=True)


if __name__ == "__main__":
    main()
