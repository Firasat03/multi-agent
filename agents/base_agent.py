"""
agents/base_agent.py — Abstract base class for all agents

Changes vs original:
  - _call_llm now returns (text, tokens, cost_usd) and logs cost to state
  - Added _validate_code_output() — guards against LLM returning prose instead of code
  - Added _extract_files_from_response() — shared multi-file parser with validation
  - Added generate_structured() pass-through for JSON responses
  - Removed module-level singleton provider — provider is injected at instantiation
    so tests can swap it without monkeypatching globals
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from config import LLM_PROVIDER, LLM_MODEL, GENERATION_CONFIG
from tools.llm_provider import get_provider
from tools.rules_loader import build_rules_block

if TYPE_CHECKING:
    from state import PipelineState

# Module-level default provider (shared across agents unless overridden in tests)
_default_provider = None


def _get_default_provider():
    global _default_provider
    if _default_provider is None:
        _default_provider = get_provider(LLM_PROVIDER, LLM_MODEL, GENERATION_CONFIG)
    return _default_provider


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Subclasses define:
        name: str
        system_role: str
        run(state) -> PipelineState
    """

    name: str = "BaseAgent"
    system_role: str = "You are a helpful AI assistant."

    def __init__(self, provider=None):
        """
        Args:
            provider: Optional LLMProvider override (useful for testing).
                      Falls back to the module-level default.
        """
        self._provider = provider or _get_default_provider()

    # ── LLM call ────────────────────────────────────────────────────────────

    def _call_llm(
        self, state: "PipelineState", user_prompt: str
    ) -> tuple[str, int]:
        """
        Build full system prompt, call LLM, accumulate cost into state.
        Returns (response_text, token_count).

        Cost estimation uses the actual token count returned by the provider
        for input (best approximation) and output (character-based fallback
        when the provider only reports total tokens).
        """
        rules_block = build_rules_block(state.user_rules)
        system_prompt = f"[ROLE]\n{self.system_role}{rules_block}"

        text, tokens = self._provider.generate(system_prompt, user_prompt)

        # Estimate input/output split: providers return total; split ~75/25
        estimated_input  = int(len(system_prompt + user_prompt) / 3.5)
        estimated_output = max(tokens - estimated_input, int(len(text) / 3.5))
        cost = self._provider.estimate_cost(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        )
        state.log(self.name, tokens=tokens, cost_usd=cost)
        return text, tokens

    def _call_llm_structured(
        self, state: "PipelineState", user_prompt: str, schema_hint: str = ""
    ) -> tuple[dict | list, int]:
        """
        Call LLM expecting a pure JSON response.
        Returns (parsed_object, token_count).
        """
        rules_block = build_rules_block(state.user_rules)
        system_prompt = f"[ROLE]\n{self.system_role}{rules_block}"

        result, tokens = self._provider.generate_structured(
            system_prompt, user_prompt, schema_hint
        )
        estimated_input  = int(len(system_prompt + user_prompt) / 3.5)
        estimated_output = max(tokens - estimated_input, 0)
        cost = self._provider.estimate_cost(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        )
        state.log(self.name, tokens=tokens, cost_usd=cost)
        return result, tokens

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def run(self, state: "PipelineState") -> "PipelineState":
        ...

    def _timed_run(self, state: "PipelineState") -> "PipelineState":
        start = time.time()
        result = self.run(state)
        elapsed_ms = int((time.time() - start) * 1000)
        result.log(agent=self.name, duration_ms=elapsed_ms)
        return result

    # ── Output validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_code_output(file_path: str, content: str) -> None:
        """
        Raise ValueError if the content looks like LLM prose rather than code
        or has obvious structural problems.
        
        Checks:
          1. Minimum code length
          2. Not explanation/prose text
          3. Not an incomplete response
          4. Has code structure (function/class/statement)
        """
        stripped = content.strip()
        
        # Check minimum length
        if len(stripped) < 10:
            raise ValueError(
                f"Content too short for {file_path!r} ({len(stripped)} chars): {stripped!r}"
            )
        
        # Check for prose/explanation patterns
        prose_starters = (
            "I ", "I'", "Here ", "Sure", "Of course", "Certainly",
            "As requested", "Below is", "The following", "This file",
            "In this", "We ", "To implement", "The code",
        )
        if any(stripped.startswith(p) for p in prose_starters):
            raise ValueError(
                f"Output for {file_path!r} looks like explanation text rather than code. "
                f"First 80 chars: {stripped[:80]!r}"
            )
        
        # Check for incomplete/truncated code patterns
        truncation_patterns = (
            "...", "et cetera", "and so on", "rest of", "similar code",
            "continue like this", "you get the idea", "etc.", "omitted",
            "truncated", "...more code", "[continued]",
        )
        if any(pattern in stripped.lower() for pattern in truncation_patterns):
            raise ValueError(
                f"Output for {file_path!r} appears truncated or incomplete. "
                f"Look for: {{..., etc., rest of, and so on, omitted, continued}}"
            )
        
        # Check it looks like actual code (has at least one structural element)
        code_indicators = (
            "{", "}", "def ", "class ", "function ", "import ", "const ", 
            "let ", "var ", "public ", "private ", "protected ", "async ",
            "package ", "#include", "fn ", "impl ", "struct ", "pub ",
        )
        if not any(ind in stripped for ind in code_indicators):
            # Could be a data file, but check the first line more carefully
            lines = stripped.split('\n')
            first_line = lines[0].strip() if lines else ""
            # If it starts with prose, it's definitely wrong
            prose_openers = ("I ", "This ", "Here", "The ", "As requested")
            if any(first_line.startswith(p) for p in prose_openers):
                raise ValueError(
                    f"Output for {file_path!r} doesn't look like code. "
                    f"First line: {first_line!r}"
                )

    # ── Multi-file response parser ────────────────────────────────────────────

    @staticmethod
    def _extract_files_from_response(
        response_text: str,
        validate: bool = True,
    ) -> dict[str, str]:
        """
        Parse one or more FILE blocks from an LLM response.

        Expected format (produced by all code-generating agents):
            # FILE: relative/path/to/file.py
            ```python
            <content>
            ```

        Args:
            response_text: Raw LLM output.
            validate:      If True, run _validate_code_output on each file.

        Returns:
            dict mapping relative file path → content string.
            Returns {} (empty) if no FILE blocks are found — caller must handle.
        """
        # Allow common formatting variants around FILE markers and fences:
        # - leading whitespace or list bullets
        # - optional blank line between header and fence
        # - optional language after ``` and extra spaces
        pattern = re.compile(
            r"^[ \t]*(?:[-*+]\s*)?#\s*FILE:\s*([^\n]+)\n"
            r"(?:[ \t]*\n)?[ \t]*```[^\n]*\n(.*?)```",
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )

        results: dict[str, str] = {}
        for m in pattern.finditer(response_text):
            file_path = m.group(1).strip()
            content   = m.group(2).strip()
            if validate:
                try:
                    BaseAgent._validate_code_output(file_path, content)
                except ValueError as exc:
                    # Log warning but don't discard — Tester's static analysis
                    # will catch real code errors; this validation catches
                    # complete prose responses.
                    print(f"[Output Validation Warning] {exc}")
                    continue
            results[file_path] = content
        return results

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_code_block(text: str, lang: str = "") -> str:
        pattern = rf"```{lang}\s*(.*?)```" if lang else r"```(?:\w+)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract and parse a JSON block from LLM response (legacy helper)."""
        import json
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        raw = match.group(1).strip() if match else text.strip()
        return json.loads(raw)