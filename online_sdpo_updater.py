"""
Core SDPO update logic for live, single-user chat training.

Backed by the Tinker-managed training API: Tinker's TrainingClient owns the weights, optimizer, and forward/backward compute; 
Tinker's SamplingClient handles generation. Only tokenization / chat-template rendering happens locally

"""
import copy
import re
import threading
import time
import traceback
from typing import Dict, List, Optional

import torch
import tinker
from tinker import types
from transformers import AutoTokenizer

from online_sdpo_updater_config import OnlineSDPOConfig


class OnlineSDPOUpdater:

    def __init__(self, config: OnlineSDPOConfig):
        self.config = config
        self.step = 0
        self.metrics_history: List[Dict] = []

        # Async training state
        self._train_thread: Optional[threading.Thread] = None
        self._train_lock = threading.Lock()
        self._last_async_metrics: Optional[Dict] = None
        self._async_error: Optional[str] = None

        print("[LIVE SDPO] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, use_fast=True, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[LIVE SDPO] Connecting to Tinker...", flush=True)
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=config.model_name_or_path,
            rank=config.lora_rank,
        )

        self._run_name = re.sub(r"[^a-zA-Z0-9._-]", "_", config.checkpoint_dir)
        print("[LIVE SDPO] Fetching initial sampling client...", flush=True)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()
        print("[LIVE SDPO] Updater ready.", flush=True)

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    def _render_prompt_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self.tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=self.config.max_context_length,
        )["input_ids"]

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate an assistant response given the conversation so far."""
        return self.generate_responses_batch([messages])[0]

    def generate_responses_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts via Tinker's sampling client."""
        temp = temperature if temperature is not None else self.config.temperature
        sampling_params = types.SamplingParams(
            max_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temp,
            top_p=self.config.top_p,
        )

        futures = [
            self.sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=self._render_prompt_ids(messages)),
                sampling_params=sampling_params,
                num_samples=1,
            )
            for messages in messages_list
        ]
        results = [f.result() for f in futures]
        return [
            self.tokenizer.decode(r.sequences[0].tokens, skip_special_tokens=True)
            for r in results
        ]

    # ------------------------------------------------------------------ #
    #  Hindsight prompt construction  (offline_trainer.py:64-70)
    # ------------------------------------------------------------------ #

    def _build_hindsight_messages(
        self,
        messages: List[Dict[str, str]],
        follow_up: str,
    ) -> List[Dict[str, str]]:
        """Append [HINDSIGHT CONTEXT] block to last user message."""
        conditional = copy.deepcopy(messages)
        block = self.config.hindsight_block_template.format(
            follow_up=follow_up.strip(),
        )
        # Walk backwards to find the last user message
        for i in range(len(conditional) - 1, -1, -1):
            if conditional[i]["role"] == "user":
                conditional[i]["content"] += block
                break
        return conditional

    # ------------------------------------------------------------------ #
    #  Datum Construction
    # ------------------------------------------------------------------ #

    def _build_datum(
        self, context_ids: List[int], completion_ids: List[int],
    ) -> "types.Datum":
        """
        Build a Tinker Datum scoring `completion_ids` as the continuation of
        `context_ids`. `weights` is 1 on completion-token positions (after
        ignore_first_k) and 0 everywhere else, so the returned per-token
        logprobs line up 1:1 with the local `_compute_token_logprobs` this
        replaces.
        """
        full_ids = context_ids + completion_ids
        model_input_tokens = full_ids[:-1]
        target_tokens = full_ids[1:]

        n_ignored = min(self.config.ignore_first_k, len(completion_ids))
        completion_start = len(context_ids) - 1  # index into target_tokens

        weights = [0] * len(target_tokens)
        for i in range(completion_start + n_ignored, len(target_tokens)):
            weights[i] = 1

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=model_input_tokens),
            loss_fn_inputs=dict(target_tokens=target_tokens, weights=weights),
        )

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        messages_before_response: List[Dict[str, str]],
        assistant_response: str,
        user_follow_up: str,
    ) -> Dict[str, float]:
        """
        One SDPO gradient step using the simple_signal loss:

            loss = -(signal.detach() * log pi(y|x)),  length-normalised
            signal = log pi(y|x,o) - log pi(y|x)

        x and xo are sent as two Datums in a single forward_backward_custom
        call; the custom loss function only backprops through the x side
        (xo's logprobs are detached before use), so one remote call yields
        the exact same signal as the local two-pass implementation did.
        """
        t0 = time.time()

        # ---- tokenize the completion ----
        completion_text = assistant_response.rstrip()
        if self.tokenizer.eos_token is not None:
            completion_text += self.tokenizer.eos_token
        completion_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]

        # ---- build x and xo prompts ----
        x_ids = self._render_prompt_ids(messages_before_response)
        xo_messages = self._build_hindsight_messages(messages_before_response, user_follow_up)
        xo_ids = self._render_prompt_ids(xo_messages)

        x_datum = self._build_datum(x_ids, completion_ids)
        xo_datum = self._build_datum(xo_ids, completion_ids)

        # ---- Simple Signal Loss  ----
        signal_clip = self.config.signal_clip
        step_metrics: Dict[str, float] = {}

        def _simple_signal_loss(data, logprobs):
            logps_x, logps_xo = logprobs[0], logprobs[1]

            # logps_x and logps_xo cover the *entire* target_tokens sequence
            # of each Datum -- and x/xo have different context lengths (xo
            # includes the hindsight block), so their full sequences are
            # different lengths. Each Datum's own "weights" mask selects
            # exactly the completion-token span, which *is* the same length
            # (and the same underlying tokens) in both -- so we must slice
            # down to that span before comparing x against xo.
            mask_x = data[0].loss_fn_inputs["weights"].to_torch().to(
                dtype=torch.bool, device=logps_x.device,
            )
            mask_xo = data[1].loss_fn_inputs["weights"].to_torch().to(
                dtype=torch.bool, device=logps_xo.device,
            )
            comp_logps_x = logps_x[mask_x]
            comp_logps_xo = logps_xo[mask_xo]

            length = max(comp_logps_x.numel(), 1)

            per_token_diff = (comp_logps_xo.detach() - comp_logps_x.detach())
            if signal_clip > 0:
                per_token_diff = per_token_diff.clamp(-signal_clip, signal_clip)

            loss = -(per_token_diff * comp_logps_x).sum() / length

            # Tinker requires every Datum's logprobs tensor to receive a
            # gradient from the loss. xo is meant to be a frozen, no-grad
            # reference (its logprobs are only ever used detached above), so
            # this registers an explicit zero-valued gradient for it -- a
            # no-op for the actual update, but satisfies that requirement.
            loss = loss + 0.0 * logps_xo.sum()

            step_metrics["signal_mean"] = (per_token_diff.sum() / length).item()
            step_metrics["policy_logp_mean"] = (comp_logps_x.detach().sum() / length).item()
            step_metrics["critic_logp_mean"] = (comp_logps_xo.detach().sum() / length).item()
            step_metrics["completion_tokens"] = int(length)
            step_metrics["loss"] = loss.item()
            return loss, dict(step_metrics)

        K = self.config.train_steps_per_example
        for _ in range(K):
            fwdbwd_future = self.training_client.forward_backward_custom(
                data=[x_datum, xo_datum], loss_fn=_simple_signal_loss,
            )
            fwdbwd_future.result()

            optim_future = self.training_client.optim_step(
                types.AdamParams(
                    learning_rate=self.config.learning_rate,
                    beta1=self.config.adam_beta1,
                    beta2=self.config.adam_beta2,
                    eps=self.config.adam_epsilon,
                    grad_clip_norm=self.config.grad_clip_norm,
                )
            )
            optim_future.result()

        self.step += 1
        step_metrics["step"] = self.step
        step_metrics["train_steps_per_example"] = K
        step_metrics["train_time_s"] = round(time.time() - t0, 2)
        self.metrics_history.append(step_metrics)

        # Refresh the sampling client so the next generation sees this update.
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()

        # ---- auto-checkpoint ----
        if (
            self.config.checkpoint_every_n_steps > 0
            and self.step % self.config.checkpoint_every_n_steps == 0
        ):
            self.save_checkpoint()

        return step_metrics

    # ------------------------------------------------------------------ #
    #  Async training
    # ------------------------------------------------------------------ #

    def wait_for_training(self) -> Optional[Dict]:
        """Block until any in-flight async training finishes. Returns metrics or None."""
        if self._train_thread is not None:
            self._train_thread.join()
            self._train_thread = None
        metrics = self._last_async_metrics
        self._last_async_metrics = None
        error = self._async_error
        self._async_error = None
        if error:
            print(f"[LIVE SDPO] Async training error: {error}", flush=True)
        return metrics

    def train_step_async(
        self,
        messages_before_response: List[Dict[str, str]],
        assistant_response: str,
        user_follow_up: str,
    ) -> None:
        """Launch train_step in a background thread."""
        # Ensure any previous async step is done before starting a new one
        self.wait_for_training()

        def _run():
            try:
                with self._train_lock:
                    metrics = self.train_step(
                        messages_before_response=messages_before_response,
                        assistant_response=assistant_response,
                        user_follow_up=user_follow_up,
                    )
                self._last_async_metrics = metrics
            except Exception as e:
                self._async_error = f"{e}\n{traceback.format_exc()}"

        self._train_thread = threading.Thread(target=_run, daemon=True)
        self._train_thread.start()

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Save full (weights + optimizer) state to Tinker, resumable via
        service_client.create_training_client_from_state_with_optimizer().
        """
        tag = tag or f"step_{self.step}"
        name = f"{self._run_name}_{tag}"
        result = self.training_client.save_state(name=name).result()
        path = result.path
        print(f"[LIVE SDPO] Checkpoint saved -> {path}", flush=True)
        return path
