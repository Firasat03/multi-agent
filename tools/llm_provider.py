"""
tools/llm_provider.py — Pluggable LLM Provider

Supports: Gemini, OpenAI, Anthropic, Ollama (and any OpenAI-compatible endpoint).

Changes vs original:
  - Retry with exponential backoff added to ALL providers (not just OpenAI)
  - Input token budget estimation before every call — warns when approaching
    the model's context window limit
  - Cost estimation per call — accumulated in PipelineState
  - Optional structured JSON output via generate_structured()
  - LLM call timeout enforced via concurrent.futures
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from config import (
    CHARS_PER_TOKEN,
    CONTEXT_WARN_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_COST_PER_1M,
    LLM_CALL_TIMEOUT_SECS,
    MAX_LLM_RETRIES,
    MODEL_CONTEXT_WINDOWS,
    MODEL_COSTS_PER_1M,
)


# ─── Abstract Interface ───────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Common interface every provider must implement."""

    model: str = ""

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """
        Call the LLM and return (response_text, total_token_count).
        Enforces timeout, budget check, and retry automatically.
        """
        self._check_token_budget(system_prompt, user_prompt)
        return self._call_with_retry(system_prompt, user_prompt)

    def generate_structured(
        self, system_prompt: str, user_prompt: str, schema_hint: str = ""
    ) -> tuple[dict | list, int]:
        """
        Call the LLM expecting a JSON response. Strips markdown fences,
        parses, and returns (parsed_object, tokens).
        Falls back to regex extraction if the response contains prose around JSON.
        """
        json_instruction = (
            "\n\nIMPORTANT: Respond with ONLY valid JSON. "
            "No explanation, no markdown fences, no preamble. "
            f"{'Schema: ' + schema_hint if schema_hint else ''}"
        )
        text, tokens = self.generate(system_prompt, user_prompt + json_instruction)
        return self._parse_json_response(text), tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for this call."""
        input_cost, output_cost = MODEL_COSTS_PER_1M.get(self.model, DEFAULT_COST_PER_1M)
        return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000

    # ── Internal ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _generate_once(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Single LLM call — no retry, no timeout wrapping."""
        ...

    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Retry with exponential backoff on transient errors."""
        base_wait = 5.0
        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(MAX_LLM_RETRIES):
            try:
                return self._call_with_timeout(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                is_transient = any(
                    marker in err_str
                    for marker in ("429", "rate_limit", "RateLimitError",
                                   "503", "overloaded", "timeout", "Timeout")
                )
                if not is_transient or attempt == MAX_LLM_RETRIES - 1:
                    raise

                # Parse provider-supplied retry-after if present
                m = re.search(r"try again in\s+([\d.]+)s", err_str, re.IGNORECASE)
                wait = float(m.group(1)) + 2 if m else base_wait * (2 ** attempt)
                print(
                    f"\n[LLM Retry {attempt + 1}/{MAX_LLM_RETRIES - 1}] "
                    f"{type(exc).__name__}: {err_str[:80]}. "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)

        raise last_error

    def _call_with_timeout(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Wrap the LLM call in a thread with a hard timeout."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._generate_once, system_prompt, user_prompt)
            try:
                return future.result(timeout=LLM_CALL_TIMEOUT_SECS)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"LLM call timed out after {LLM_CALL_TIMEOUT_SECS}s"
                )

    def _check_token_budget(self, system_prompt: str, user_prompt: str) -> None:
        """
        Raise RuntimeError if the estimated prompt size would overflow the model's
        context window (leaving no room for a response).
        Warn when usage exceeds CONTEXT_WARN_THRESHOLD.
        """
        estimated_tokens = int(
            (len(system_prompt) + len(user_prompt)) / CHARS_PER_TOKEN
        )
        context_window = MODEL_CONTEXT_WINDOWS.get(self.model, DEFAULT_CONTEXT_WINDOW)

        # Reserve ~10 % of the window for the response; hard-fail if exceeded
        hard_limit = int(context_window * 0.90)
        if estimated_tokens > hard_limit:
            raise RuntimeError(
                f"Prompt too large for model {self.model!r}: estimated ~{estimated_tokens:,} "
                f"input tokens exceeds the hard limit of {hard_limit:,} "
                f"(90 % of {context_window:,} context window). "
                "Trim file contents or split the task into smaller steps."
            )

        usage_fraction = estimated_tokens / context_window
        if usage_fraction >= CONTEXT_WARN_THRESHOLD:
            print(
                f"\n[Token Budget Warning] Estimated input ~{estimated_tokens:,} tokens "
                f"({usage_fraction:.0%} of {context_window:,} context window for {self.model!r}). "
                "Consider trimming file contents in the prompt."
            )

    @staticmethod
    def _parse_json_response(text: str) -> dict | list:
        """Strip markdown fences and parse JSON from LLM response."""
        # Strip ```json ... ``` or ``` ... ```
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        
        # Try direct parse first
        try:
            return json.loads(clean)
        except json.JSONDecodeError as first_error:
            pass
        
        # Fallback: find the first valid JSON block using bracket matching
        # This avoids the greedy regex problem that included extra data
        for start_pos in range(len(clean)):
            char = clean[start_pos]
            if char == '[':
                try:
                    # Try to find matching closing bracket
                    bracket_count = 0
                    for end_pos in range(start_pos, len(clean)):
                        if clean[end_pos] == '[':
                            bracket_count += 1
                        elif clean[end_pos] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_str = clean[start_pos:end_pos + 1]
                                return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    continue
            elif char == '{':
                try:
                    # Try to find matching closing brace
                    brace_count = 0
                    for end_pos in range(start_pos, len(clean)):
                        if clean[end_pos] == '{':
                            brace_count += 1
                        elif clean[end_pos] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = clean[start_pos:end_pos + 1]
                                return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")


# ─── Gemini Provider ──────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):

    def __init__(self, model: str, generation_config: dict) -> None:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        self.model = model
        self._client = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

    def _generate_once(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        full_prompt = f"{system_prompt}\n\n[TASK]\n{user_prompt}"
        response = self._client.generate_content(full_prompt)
        text = response.text.strip()
        tokens = 0
        try:
            tokens = response.usage_metadata.total_token_count
        except Exception:
            pass
        return text, tokens


# ─── OpenAI Provider ──────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):

    def __init__(self, model: str, generation_config: dict, base_url: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY", "")
        kwargs: dict = {"api_key": api_key or "sk-placeholder"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self._temperature = generation_config.get("temperature", 0.2)
        self._max_tokens = generation_config.get("max_output_tokens", 8192)

    def _generate_once(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text   = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def generate_structured(
        self, system_prompt: str, user_prompt: str, schema_hint: str = ""
    ) -> tuple[dict | list, int]:
        """Use OpenAI JSON mode for reliable structured output."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            text   = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
            return json.loads(text), tokens
        except Exception:
            # Fall back to base implementation
            return super().generate_structured(system_prompt, user_prompt, schema_hint)


# ─── Anthropic Provider ───────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):

    def __init__(self, model: str, generation_config: dict) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._client = __import__("anthropic").Anthropic(api_key=api_key)
        self.model = model
        self._max_tokens = generation_config.get("max_output_tokens", 8192)

    def _generate_once(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        tokens = (
            response.usage.input_tokens + response.usage.output_tokens
            if response.usage else 0
        )
        return text, tokens


# ─── Ollama Provider ──────────────────────────────────────────────────────────

class OllamaProvider(OpenAIProvider):

    def __init__(self, model: str, generation_config: dict) -> None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        super().__init__(model=model, generation_config=generation_config, base_url=base_url)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_provider(provider_name: str, model: str, generation_config: dict) -> LLMProvider:
    name = provider_name.lower().strip()

    if name == "gemini":
        return GeminiProvider(model=model, generation_config=generation_config)
    elif name == "openai":
        return OpenAIProvider(model=model, generation_config=generation_config)
    elif name == "anthropic":
        return AnthropicProvider(model=model, generation_config=generation_config)
    elif name == "ollama":
        return OllamaProvider(model=model, generation_config=generation_config)
    elif name == "openai_compat":
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            raise EnvironmentError(
                "LLM_BASE_URL must be set when using provider 'openai_compat'."
            )
        return OpenAIProvider(model=model, generation_config=generation_config, base_url=base_url)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            "Choose from: gemini, openai, anthropic, ollama, openai_compat"
        )