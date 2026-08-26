from dataclasses import dataclass
from typing import Optional


@dataclass
class OnlineSDPOConfig:
    # ── Model ──
    model_name_or_path: str = "Qwen/Qwen3-8B"

    # ── LoRA (Tinker's TrainingClient trains via LoRA only) ──
    lora_rank: int = 32

    # ── SDPO signal ──
    signal_clip: float = 0.0  # 0 = no clipping
    ignore_first_k: int = 0

    # ── Training schedule ──
    async_training: bool = False  # True = train in background thread after generation
    train_steps_per_example: int = 1  # K gradient steps per interaction

    # ── Optimizer (passed straight through to tinker.types.AdamParams) ──
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-6
    grad_clip_norm: float = 1.0  # 0 = no clipping

    # ── Generation ──
    max_new_tokens: int = 2048
    max_context_length: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0

    # ── Hindsight prompt format (matches offline_trainer.py:64-70) ──
    hindsight_block_template: str = (
        "\n\n=== HINDSIGHT CONTEXT ===\n"
        "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
        "{follow_up}"
    )

    # ── Checkpointing ── name prefix for Tinker-hosted saves (not a local path)
    checkpoint_dir: str = "live-sdpo"
    checkpoint_every_n_steps: int = 10

    # ── Logging ──
    log_to_wandb: bool = False
    wandb_project: str = "live-sdpo"
    wandb_run_name: Optional[str] = None

    # ── Gradio ──
    server_port: int = 7860
    share: bool = False
