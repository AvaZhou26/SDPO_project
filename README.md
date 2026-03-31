# Aligning Language Models from User Interactions

This repository contains the training and evaluation code for **Self-Distillation Policy Optimization (SDPO) from User Interactions**.

The core idea: at each step the policy generates a response `y` to a prompt `x`, a user simulator produces a follow-up `o`, and the per-token log-ratio `log p(y | x, o) - log p(y | x)` serves as a token-level advantage signal to update the policy. This enables language models to adapt to individual user preferences through natural interaction, without explicit reward models or preference labels.

The repo supports two settings:

- **Online SDPO** — the policy generates responses on-the-fly; the signal is computed immediately against the current model. Supports both local (Qwen) and API-based (Claude) user simulators.
- **Offline SDPO** — the signal is computed from existing interaction data (e.g. WildFeedback, WildChat).

> **Paper:** [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273)

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.7.0`, `transformers==4.57.6`, `accelerate==1.6.0`, `trl==0.24.0`, `datasets==3.5.0`, `peft==0.15.1`, `vllm>=0.8.5`, `wandb`, `anthropic`.

Set your credentials (or place them in a `.env` file in the repo root — all scripts source it automatically):

```bash
export HF_TOKEN=...           # if model downloads require authentication
export ANTHROPIC_API_KEY=...  # needed for Claude user simulator / judge
export WANDB_API_KEY=...      # optional, for experiment tracking
```

---

## Data Preparation

Prepare the datasets before running any experiments. Each script downloads the data from HuggingFace and writes JSONL files locally.

| Dataset | Command | Output |
|---------|---------|--------|
| HelpSteer2 (`nvidia/HelpSteer2`) | `python auxiliary/preprocess_helpsteer.py --out_dir data/helpsteer_prompts` | `data/helpsteer_prompts/{train,validation}.jsonl` |
| TL;DR (`openai/summarize_from_feedback`) | `python auxiliary/preprocess_tldr_dataset.py --out_dir data/tldr_prompts_unique` | `data/tldr_prompts_unique/{train,validation}.jsonl` |
| WildFeedback (`microsoft/WildFeedback`) | `python auxiliary/preprocess_wildfeedback.py` | `data/wildfeedback/wildfeedback_interactions.jsonl` |
| WildChat (`allenai/WildChat`) | `python auxiliary/preprocess_wildchat.py` | `data/wildchat/wildchat_interactions_v1.jsonl` |

---

## Online SDPO

`eval_online_sdpo.sh` runs an interleaved training and evaluation loop: for each training prompt, the model generates a response, the user simulator provides feedback, and the SDPO signal is used for an immediate gradient update. Evaluation runs periodically on held-out prompts.

### Quick start (HelpSteer2 + Claude user simulator)

```bash
./scripts/eval_online_sdpo.sh
```

### Using a different dataset or configuration

```bash
MODEL="Qwen/Qwen3-8B" \
USER_MODEL="Qwen/Qwen3-8B" \
STYLE="concise_casual_beginner" \
DATA_DIR=./data/tldr_prompts_unique \
./scripts/eval_online_sdpo.sh
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | Policy model (HuggingFace ID or local path) |
| `USER_MODEL` | `Qwen/Qwen3-32B` | User simulator model |
| `STYLE` | `no_emojis` | Target user style profile |
| `EVAL_STYLES` | *(empty)* | Additional styles to evaluate on |
| `LR` | `5e-6` | Learning rate |
| `LOSS_MODE` | `full_distillation` | Loss function variant |
| `TRAIN_N` | `15` | Number of training examples |
| `EVAL_N` | `100` | Number of evaluation examples |
| `EVAL_EVERY` | `3` | Evaluate every N training steps |
| `TRAIN_STEPS_PER_EXAMPLE` | `1` | Gradient steps per example |
| `SEED` | `1234` | Random seed |
| `DATA_DIR` | `data/helpsteer_prompts` | Directory containing `train.jsonl` and `validation.jsonl` |
| `BASELINE_MODEL` | *(empty)* | Baseline for comparison (defaults to initial model) |
| `OUTPUT_DIR` | auto-generated | Output directory for checkpoints and results |

---

## Offline SDPO

Train on pre-collected interaction data (WildFeedback or WildChat). Uses `accelerate` for multi-GPU training.

```bash
TRAIN_JSONL=./data/wildfeedback/wildfeedback_interactions.jsonl \
./scripts/train_offline_sdpo.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_JSONL` | *(required)* | Path to interaction data JSONL |
| `BASE_MODEL` | `Qwen/Qwen3-4B` | Policy model |
| `LR` | `2e-6` | Learning rate |
| `BS` | `4` | Per-device batch size |
| `GA` | `8` | Gradient accumulation steps |
| `NUM_EPOCHS` | `2` | Training epochs |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `ACCELERATE_CONFIG` | `multigpu_accelerate_config.yaml` | Accelerate config file |

---

## Evaluation

### Checkpoint evaluation

Compare one or more saved checkpoints against a baseline model across multiple user styles:

```bash
CHECKPOINTS="/path/to/ckpt1 /path/to/ckpt2" \
BASELINE_MODEL="Qwen/Qwen3-8B" \
./scripts/eval_checkpoints.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINTS` | *(required)* | Space-separated list of checkpoint paths |
| `BASELINE_MODEL` | *(required)* | Baseline model path or HuggingFace ID |
| `EVAL_STYLES` | `less_filler_praise_sycophancy no_emojis answer_directly_reduce_formatting` | Styles to evaluate |
| `USER_MODEL` | `Qwen/Qwen3-32B` | User simulator model |
| `EVAL_N` | `100` | Number of evaluation examples |

### In-context oracle

Upper-bound baseline where the style instruction is given directly in the system prompt (no learning from interactions):

```bash
./scripts/eval_incontext_oracle.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | Model to evaluate |
| `JUDGE_MODEL` | `Qwen/Qwen3-32B` | Judge model |
| `EVAL_N` | `100` | Number of evaluation examples |
| `DATA_DIR` | `data/tldr_prompts_unique` | Data directory |

---

## Signal Visualization

Compute and visualize the per-token SDPO signal for a set of prompt/feedback cases. Generates heatmaps comparing the signal under an unrelated follow-up (should be near zero) versus a relevant follow-up (should have structure).

```bash
./scripts/run_signal_analysis.sh
```

Outputs: `sdpo_signals.json`, `unrelated.png`, `followup.png`, `stacked.png`, `side_by_side.png`, `case{N}_tokens.png`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | Model to score with |
| `CASES_JSON` | `auxiliary/signal_analysis_cases.json` | Input cases |
| `N_CASES` | `24` | Number of cases to process |

---

## Repository Structure

```
.
├── eval_online_sdpo.py              # Online SDPO training + evaluation loop
├── online_sdpo_updater.py           # Core online training logic
├── online_sdpo_updater_config.py    # Configuration dataclass
├── auxiliary/
│   ├── eval_checkpoints.py          # Checkpoint evaluation
│   ├── eval_incontext_oracle.py     # In-context oracle baseline
│   ├── sdpo_signal_analysis.py      # Per-token signal visualization
│   ├── user_simulator.py            # Local user simulator (Qwen-based)
│   ├── claude_user_simulator.py     # Claude API user simulator
│   ├── vllm_user_simulator.py       # vLLM-accelerated user simulator
│   ├── style_judge.py               # Local style judge
│   ├── claude_style_judge.py        # Claude API style judge
│   ├── evaluation_helpers.py        # Shared evaluation utilities
│   ├── preprocess_helpsteer.py      # HelpSteer2 data preparation
│   ├── preprocess_tldr_dataset.py   # TL;DR data preparation
│   ├── preprocess_wildfeedback.py   # WildFeedback data preparation
│   └── preprocess_wildchat.py       # WildChat data preparation
├── offline_sdpo/
│   ├── main_offline_sdpo.py         # Offline SDPO training entry point
│   └── offline_sdpo_trainer.py      # Offline trainer implementation
├── scripts/
│   ├── eval_online_sdpo.sh          # Run online SDPO
│   ├── eval_checkpoints.sh          # Evaluate checkpoints
│   ├── eval_incontext_oracle.sh     # Run in-context oracle
│   ├── train_offline_sdpo.sh        # Train offline SDPO
│   └── run_signal_analysis.sh       # Signal visualization
└── requirements.txt
```

---

## Common Options

**Dry-run mode** — all scripts accept `--dry-run` to print the resolved command without executing:

```bash
./scripts/eval_online_sdpo.sh --dry-run
```

**Output directories** — scripts use a portable fallback chain for output and cache directories:

```bash
BASE_WORK="${SCRATCH:-${TMPDIR:-/tmp}}"
```

Override with `BASE_WORK`, `OUTPUT_DIR`, or `CACHE_DIR` as needed.

**Multi-GPU** — offline training scripts support multi-GPU via accelerate:

```bash
WORLD_SIZE=4 ACCELERATE_CONFIG=./multigpu_accelerate_config.yaml \
TRAIN_JSONL=... ./scripts/train_offline_sdpo.sh
```

---

## Citation

```bibtex
@article{buening2026aligning,
  title={Aligning language models from user interactions},
  author={Buening, Thomas Kleine and H{\"u}botter, Jonas and P{\'a}sztor, Barna and Shenfeld, Idan and Ramponi, Giorgia and Krause, Andreas},
  journal={arXiv preprint arXiv:2603.12273},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
