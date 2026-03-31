# claude_style_judge.py
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from anthropic import Anthropic

from .user_simulator import STYLE_PERSONAS


class ClaudeStyleJudge:
    """
    Claude-based LLM-as-a-judge for style preference.

    Decision encoding:
      0 -> prefer A
      1 -> prefer B
     -1 -> tie/uncertain

    Uses symmetric judging:
      - judge(A,B)
      - judge(B,A) and invert
      - if agree -> keep, else tie
    """

    def __init__(
        self,
        style: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 8,
        base_backoff_s: float = 0.75,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}")

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = Anthropic(api_key=api_key)
        self.model = model

        self.system_persona = STYLE_PERSONAS[style]
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        self.max_retries = int(max_retries)
        self.base_backoff_s = float(base_backoff_s)

    def _build_system(self) -> str:
        return (
            f"{self.system_persona}\n\n"
            "You are acting as a strict evaluator of WHICH response better matches your preference described in the USER PROFILE.\n"
            "You must follow these rules:\n"
            "- Judge ONLY style, tone, formatting, verbosity, and complexity relative to the persona.\n"
            "- Do NOT judge factual correctness.\n"
            "- Do NOT rewrite responses.\n"
            "- You may reason briefly about your decision.\n"
            "- You MUST end your response with exactly: Judgement: A, Judgement: B, or Judgement: C\n"
            "- C means tie/uncertain.\n"
        )

    def get_system_persona(self) -> str:
        return self.system_persona

    def _build_user(self, prompt: str, a: str, b: str) -> str:
        return (
            "User prompt:\n"
            f"{prompt}\n\n"
            "Response A:\n"
            f"{a}\n\n"
            "Response B:\n"
            f"{b}\n\n"
            "Which response do you prefer as this user? Reason briefly, then end with Judgement: A, Judgement: B, or Judgement: C."
        )

    def _call_once(self, prompt: str, a: str, b: str) -> int:
        resp = self.client.messages.create(
            model=self.model,
            system=self._build_system(),
            messages=[{"role": "user", "content": self._build_user(prompt, a, b)}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        out = "".join(text_parts).strip()
        return self._parse_judgement(out)

    @staticmethod
    def _parse_judgement(text: str) -> int:
        """Extract decision from 'Judgement: A/B/C' at the end of the output.
        Falls back to last standalone A/B/C if the structured format is missing."""
        clean = text.strip()
        if not clean:
            return -1

        # Look for 'Judgement: X' (case-insensitive, allow 'Judgment' too)
        m = re.search(r'[Jj]udge?ment:\s*([ABCabc])', clean)
        if m:
            letter = m.group(1).upper()
            if letter == 'A':
                return 0
            if letter == 'B':
                return 1
            return -1

        # Fallback: last standalone A/B/C character
        for char in reversed(clean.upper()):
            if char == 'A':
                return 0
            if char == 'B':
                return 1
            if char == 'C':
                return -1
        return -1

    def _call_with_retries(self, prompt: str, a: str, b: str) -> int:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_once(prompt, a, b)
            except Exception as e:
                last_err = e
                time.sleep(self.base_backoff_s * (2**attempt))
        raise RuntimeError(f"Claude judge failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _invert_ab(decisions: List[int]) -> List[int]:
        out: List[int] = []
        for d in decisions:
            if d == 0:
                out.append(1)
            elif d == 1:
                out.append(0)
            else:
                out.append(-1)
        return out

    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        batch_size: int = 16,
        max_workers: int = 16,
    ) -> List[int]:
        assert len(prompts) == len(completions_a) == len(completions_b)
        n = len(prompts)

        final: List[int] = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # Forward AB + backward BA — all concurrent
                futs_ab = [
                    pool.submit(self._call_with_retries, prompts[i], completions_a[i], completions_b[i])
                    for i in range(start, end)
                ]
                futs_ba = [
                    pool.submit(self._call_with_retries, prompts[i], completions_b[i], completions_a[i])
                    for i in range(start, end)
                ]

                d_ab = [f.result() for f in futs_ab]
                d_ba = [f.result() for f in futs_ba]

            d_ba_inv = self._invert_ab(d_ba)

            for x, y in zip(d_ab, d_ba_inv):
                final.append(x if x == y else -1)

        return final
