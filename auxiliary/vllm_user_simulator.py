# vllm_user_simulator.py
"""
vLLM-backed user simulator and style judge.

Both classes share a single vLLM LLM instance for fast batched inference.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .user_simulator import UserSimulator, STYLE_PERSONAS


class VLLMStyleUserSimulator(UserSimulator):
    """
    vLLM-backed style user simulator.

    Accepts a pre-initialized vLLM LLM instance and tokenizer so that the
    same engine can be shared with VLLMStyleJudge.
    """

    def __init__(
        self,
        llm,
        tokenizer,
        style: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}")

        self.llm = llm
        self.tok = tokenizer
        self.style = style
        self.system_persona = STYLE_PERSONAS[style]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _build_system(self) -> str:
        return (
            f"{self.system_persona}\n\n"
            "You are simulating the *user's next message* in a chat with an AI assistant.\n"
            "Rules:\n"
            "- Respond as the user with the USER PROFILE above in simple language.\n"
            "- Provide feedback ONLY to the assistant's response with respect to the preferences in the USER PROFILE.\n"
            "- Do NOT answer the original request yourself.\n"
            "- Do NOT give general feedback that does not relate to the USER PROFILE.\n"
            "- Output ONLY the user message text.\n"
        )

    def _build_prompt_text(self, raw_prompt: str, completion: str) -> str:
        system = self._build_system()
        user_msg = (
            "Original user request:\n"
            f"{raw_prompt}\n\n"
            "Assistant response:\n"
            f"{completion}\n\n"
            "Write the user's next message:"
        )
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        return self.tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def generate_feedback(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: Optional[List[float]] = None,
    ) -> List[str]:
        from vllm import SamplingParams

        prompt_texts = [
            self._build_prompt_text(p, c) for p, c in zip(prompts, completions)
        ]
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params=sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]


class VLLMStyleJudge:
    """
    vLLM-backed pairwise style judge with symmetric judging.

    Shares a vLLM LLM instance with VLLMStyleUserSimulator.
    """

    def __init__(
        self,
        llm,
        tokenizer,
        style: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}")

        self.llm = llm
        self.tok = tokenizer
        self.style = style
        self.system_persona = STYLE_PERSONAS[style]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_raw_outputs: List[str] = []

    def get_system_persona(self) -> str:
        return self.system_persona

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

    def _build_prompt_text(self, raw_prompt: str, ca: str, cb: str) -> str:
        system = self._build_system()
        user_msg = (
            "User prompt:\n"
            f"{raw_prompt}\n\n"
            "Response A:\n"
            f"{ca}\n\n"
            "Response B:\n"
            f"{cb}\n\n"
            "Which response do you prefer as this user? Reason briefly, then end with Judgement: A, Judgement: B, or Judgement: C."
        )
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        return self.tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    @staticmethod
    def _parse_judgement(text: str) -> int:
        clean = text.strip()
        if not clean:
            return -1

        m = re.search(r'[Jj]udge?ment:\s*([ABCabc])', clean)
        if m:
            letter = m.group(1).upper()
            if letter == 'A':
                return 0
            if letter == 'B':
                return 1
            return -1

        for char in reversed(clean.upper()):
            if char == 'A':
                return 0
            if char == 'B':
                return 1
            if char == 'C':
                return -1
        return -1

    @staticmethod
    def _invert_ab(decisions: List[int]) -> List[int]:
        out = []
        for d in decisions:
            if d == 0:
                out.append(1)
            elif d == 1:
                out.append(0)
            else:
                out.append(-1)
        return out

    def _get_generation_decisions(self, p, ca, cb) -> List[int]:
        from vllm import SamplingParams

        texts = [self._build_prompt_text(p[i], ca[i], cb[i]) for i in range(len(p))]
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.llm.generate(texts, sampling_params=sampling_params)
        gen_texts = [out.outputs[0].text for out in outputs]

        self.last_raw_outputs = gen_texts

        return [self._parse_judgement(t) for t in gen_texts]

    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        batch_size: int = 16,
    ) -> List[int]:
        """Symmetric judge: runs (A,B) and (B,A), agrees or ties."""
        assert len(prompts) == len(completions_a) == len(completions_b)

        # No need to batch manually — vLLM handles continuous batching.
        # Send all AB and BA at once.
        d_ab = self._get_generation_decisions(prompts, completions_a, completions_b)
        d_ba = self._get_generation_decisions(prompts, completions_b, completions_a)
        d_ba_inv = self._invert_ab(d_ba)

        return [x if x == y else -1 for x, y in zip(d_ab, d_ba_inv)]
