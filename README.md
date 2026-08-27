# Aligning Language Models from User Interactions

This repository contains the training and evaluation code for **Self-Distillation Policy Optimization (SDPO) from User Interactions**.

The core idea: at each step the policy generates a response `y` to a prompt `x`, a user simulator produces a follow-up `o`, and the per-token log-ratio `log p(y | x, o) - log p(y | x)` serves as a token-level advantage signal to update the policy. This enables language models to adapt to individual user preferences through natural interaction, without explicit reward models or preference labels.

- **Online SDPO** — the policy generates responses on-the-fly; the signal is computed immediately against the current model. Supports both local (Qwen) and API-based (DeepSeek) user simulators.

> **Base Code Paper:** [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273)

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.7.0`, `transformers==4.57.6`, `accelerate==1.6.0`, `trl==0.24.0`, `datasets==3.5.0`, `peft==0.15.1`, `vllm>=0.8.5`, `wandb`, `anthropic`, `tinker`.

Online SDPO training/generation runs on the [Tinker](https://tinker-docs.thinkingmachines.ai/) managed training API rather than a local GPU — the policy model's weights, optimizer, and generation all live on Tinker's servers.

Set your credentials (or place them in a `.env` file in the repo root — all scripts source it automatically):

```bash
export HF_TOKEN=...           # if model downloads require authentication
export DEEPSEEK_API_KEY=...   # needed for DeepSeek user simulator / judge (via DeepSeek's Anthropic-compatible endpoint)
export TINKER_API_KEY=...     # needed for online SDPO (training + generation)
```

---

## Data Preparation

Prepare the datasets before running any experiments. Each script downloads the data from HuggingFace and writes JSONL files locally.

| Dataset | Command | Output |
|---------|---------|--------|
| HelpSteer2 (`nvidia/HelpSteer2`) | `python auxiliary/preprocess_helpsteer.py --out_dir data/helpsteer_prompts` | `data/helpsteer_prompts/{train,validation}.jsonl` |

---

## Repository Structure

```
.
├── eval_online_sdpo_curriculum.py    # Entry point: sequential multi-preference curriculum experiment
├── online_sdpo_updater.py            # Core online SDPO training logic (Tinker-backed)
├── online_sdpo_updater_config.py     # Training configuration dataclass
├── auxiliary/
│   ├── deepseek_user_simulator.py    # DeepSeek API user simulator (generates per-style follow-ups)
│   ├── deepseek_style_judge.py       # DeepSeek API pairwise style judge
│   ├── user_simulator.py             # Shared STYLE_PERSONAS + base simulator class
│   └── preprocess_helpsteer.py       # One-time script that generated data/helpsteer_prompts/

```

Two other files aren't executed by the curriculum script but are worth knowing about, since it was built directly from them: `eval_online_sdpo.py` (the single-style predecessor whose `compute_metrics()` logic and training/eval loop structure were generalized into the curriculum version) and `auxiliary/eval_checkpoints.py` (whose frozen-snapshot sampling pattern the curriculum script re-implements locally).

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
