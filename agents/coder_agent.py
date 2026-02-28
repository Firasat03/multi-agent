"""
agents/coder_agent.py — Coder Agent

Changes vs original:
  - Uses shared _extract_files_from_response() with validation — guards against
    LLM returning prose instead of code blocks
  - Per-file output validation via _validate_code_output()
  - Returns CoderOutput; Orchestrator calls state.apply(output)
  - Truncates context when generated_files would exceed token budget
  - Cleaner fix-apply path: logs which files were modified
"""

from __future__ import annotations

import os

from agents.base_agent import BaseAgent
from config import CHARS_PER_TOKEN, DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS, Status
from state import CoderOutput, PipelineState
from tools.mcp_client import get_client


# Max chars of file content to include in fix prompts before truncating
_MAX_FILES_CHARS = 40_000


class CoderAgent(BaseAgent):
    name = "Coder"
    system_role = (
        "You are an Expert Backend Developer specializing in production-quality code.\n\n"
        "YOUR ROLE:\n"
        "- Implement exactly what the plan specifies\n"
        "- Write complete, working code with clear structure\n"
        "- Output code in the format specified below\n\n"
        "OUTPUT FORMAT REQUIREMENTS:\n"
        "1. Output code in FILE blocks only. No preamble, no explanations.\n"
        "2. Format for each file:\n"
        "\n"
        "# FILE: relative/path/to/file.ext\n"
        "```language\n"
        "<complete file content>\n"
        "```\n"
        "\n"
        "3. FILE block rules:\n"
        "   - Start '# FILE:' at column 0 (no indent)\n"
        "   - One space after 'FILE:' before the path\n"
        "   - Next line: ```<language> (example: ```python)\n"
        "   - Include complete file content, do not truncate\n"
        "   - Close with ``` on its own line\n"
        "\n"
        "4. Avoid including:\n"
        "   - Prose or explanations outside code blocks\n"
        "   - Comments like 'rest of file' or similar markers\n"
        "   - Incomplete functions or placeholder code\n"
        "   - Extra fence blocks without FILE: headers\n"
        "\n"
        "LANGUAGE MAPPING:\n"
        "- Python: ```python\n"
        "- Java: ```java\n"
        "- TypeScript: ```typescript\n"
        "- JavaScript: ```javascript\n"
        "- Go: ```go\n"
        "\n"
        "Before responding, verify each file follows the format above.\n"
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.CODING
        if state.fix_instructions:
            return self._apply_fix(state)
        return self._generate_from_plan(state)

    # ── Initial generation ────────────────────────────────────────────────

    def _generate_from_plan(self, state: PipelineState) -> PipelineState:
        new_files: dict[str, str] = {}
        total_tokens = 0

        for idx, item in enumerate(state.plan, 1):
            if item.action == "DELETE":
                state.generated_files.pop(item.file, None)
                print(f"  [{idx}/{len(state.plan)}] ❌ DELETE {item.file}")
                continue

            existing = self._read_existing(item, state)
            prompt = self._build_prompt(item, existing, state)
            
            # Log progress
            action_icon = "✏️" if item.action == "MODIFY" else "✨"
            print(f"  [{idx}/{len(state.plan)}] {action_icon} {item.action} {item.file}...", end="", flush=True)
            
            response_text, tokens = self._call_llm(state, prompt)
            total_tokens += tokens

            parsed = self._extract_files_from_response(response_text, validate=True)

            if parsed:
                new_files.update(parsed)
                print(f" ✓ ({tokens} tokens)")
            else:
                # Fall back: try extracting a single code block (no FILE: header)
                lang = _ext_to_lang(item.file)
                code = self._extract_code_block(response_text, lang)
                try:
                    self._validate_code_output(item.file, code)
                    new_files[item.file] = code
                    print(f" ✓ ({tokens} tokens) [fallback parse]")
                except ValueError as exc:
                    print(f" ✗")
                    raise RuntimeError(
                        f"Coder failed to produce valid code for {item.file!r}: {exc}"
                    ) from exc

        print(f"\n📝 Coder complete: {len(new_files)} files generated")
        output = CoderOutput(
            generated_files=new_files,
            modified_files=set(new_files.keys())  # All newly generated files
        )
        state.apply(output)
        state.fix_instructions = None
        state.log(
            self.name,
            tokens=total_tokens,
            notes=f"{len(new_files)} files generated from plan",
        )
        return state

    def _read_existing(self, item, state: PipelineState) -> str:
        """Read existing file content for MODIFY actions."""
        if item.action != "MODIFY":
            return ""
        if item.file in state.generated_files:
            return state.generated_files[item.file]
        if state.project_root:
            try:
                from tools.file_tools import read_file
                return read_file(os.path.join(state.project_root, item.file))
            except Exception:
                pass
        return ""

    def _build_prompt(self, item, existing: str, state: PipelineState) -> str:
        existing_block = (
            f"\nExisting content to modify:\n```\n{existing}\n```" if existing else ""
        )
        return f"""
Implement the following backend file.

File: {item.file}
Action: {item.action}
Description: {item.description}
API Contract: {item.api_contract or 'N/A'}
Scope estimate: {item.scope_estimate or 'N/A'}
{existing_block}

Full task context: {state.task_prompt}

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

EXAMPLE (if implementing src/auth/login.py):

# FILE: src/auth/login.py
```python
import requests
from utils import validate_token

def login(username: str, password: str):
    if not username or not password:
        raise ValueError("Missing credentials")
    user = validate_token(username)
    return {{"user_id": user.id, "token": generate_token(user)}}
```

CRITICAL RULES:
1. First line: '# FILE: {item.file}' starting at column 0
2. Next line: ```<language> (e.g., ```python)
3. Include complete file content, please do not truncate
4. Close with ``` on its own line
5. Output only the FILE block with no additional explanation

VERIFICATION:
  ☐ Starts with "# FILE: {item.file}"
  ☐ Second line is ```<language>
  ☐ Code is complete
  ☐ Ends with ``` on its own line
  ☐ No text outside the code blocks
─────────────────────────────────────────────────────────────────────

Please output the file block now:
"""

    # ── Fix / retry generation ────────────────────────────────────────────

    def _apply_fix(self, state: PipelineState) -> PipelineState:
        files_block = self._format_files_truncated(state.generated_files)

        prompt = f"""
Please fix code failures by regenerating the source files based on the debug analysis.

DEBUGGER'S FIX INSTRUCTIONS:
{state.fix_instructions}

CURRENT SOURCE FILES:
{files_block}

YOUR TASK:
1. Review the fix instructions
2. Identify which specific files need updates
3. Regenerate ONLY those files with the recommended fixes applied
4. Return updated files in the format below

IMPORTANT: Do NOT regenerate files that do not require changes. Files you do not include in your response will be preserved exactly as they are. This is critical to stay within token limits.

─────────────────────────────────────────────────────────────────────
OUTPUT FORMAT:

# FILE: src/auth/login.py
```python
<complete updated content>
```

# FILE: src/config.py
```python
<complete updated content>
```

GUIDELINES:
1. Each file starts with '# FILE: <path>' at column 0
2. Follow immediately with ```<language>
3. Include complete file content
4. Close with ```
5. Separate multiple files with a blank line
6. Provide only FILE blocks, no additional text

FILES CURRENTLY IN PROJECT:
{', '.join(state.generated_files.keys()) or 'applicable files'}

Please provide the updated files (ONLY for those being modified):
"""
        print(f"\n🔧 Coder: Applying fixes from Debugger...")
        response_text, tokens = self._call_llm(state, prompt)
        parsed = self._extract_files_from_response(response_text, validate=True)

        fixed_files: dict[str, str] = {}
        if parsed:
            fixed_files = parsed
            print(f"✓ Extracted {len(parsed)} fixed file(s)")
        else:
            # Try harder: if no explicit FILE blocks, try extracting from prose
            console_msg = (
                "[Coder] Warning: No FILE blocks found in response. "
                "Attempting to extract code from Debugger instructions and regenerate..."
            )
            print(console_msg)
            
            # As a fallback, ask for regeneration again with stricter format
            regenerate_prompt = f"""
Please output the fixed files using the standard format. Here's what's needed:

Files to update: {', '.join(state.generated_files.keys())}

Fixes to apply:
{state.fix_instructions}

FORMAT EXAMPLE:

# FILE: src/auth.py
```python
import hashlib

def validate_password(pwd: str) -> bool:
    return len(pwd) >= 8
```

# FILE: src/config.py
```python
DATABASE_URL = "postgresql://localhost/db"
```

GUIDELINES:
  • Each file: '# FILE: <path>' at column 0
  • Next line: ```<language>
  • Complete code (no truncation)
  • Close with ``` on its own line
  • Nothing outside the ``` blocks
  • Blank line between files

Please provide the files now:
"""
            retry_text, retry_tokens = self._call_llm(state, regenerate_prompt)
            tokens += retry_tokens
            fixed_files = self._extract_files_from_response(retry_text, validate=False)
            
            if not fixed_files:
                print(
                    "[Coder] ERROR: Could not extract fixed files from response. "
                    "The fix may not have been applied. Outputting unchanged files."
                )
                fixed_files = {}
            else:
                print(f"✓ Retry successful: extracted {len(fixed_files)} file(s)")

        # Show which files were updated
        if fixed_files:
            print("\n📄 Updated files:")
            for fpath in fixed_files:
                print(f"   • {fpath}")

        output = CoderOutput(
            generated_files=fixed_files,
            modified_files=set(fixed_files.keys())  # Only these files were fixed
        )
        state.apply(output)
        state.fix_instructions = None
        state.log(
            self.name,
            tokens=tokens,
            notes=f"Fix applied to {len(fixed_files)} file(s): {list(fixed_files.keys())}",
        )
        return state

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_files_truncated(files: dict[str, str]) -> str:
        """
        Format files for inclusion in a fix prompt.
        Truncates large file sets to avoid blowing the context window.
        """
        parts = []
        total_chars = 0
        for path, content in files.items():
            entry = f"# FILE: {path}\n```\n{content}\n```"
            if total_chars + len(entry) > _MAX_FILES_CHARS:
                parts.append(f"# FILE: {path}\n(truncated — file too large to include)")
            else:
                parts.append(entry)
                total_chars += len(entry)
        return "\n\n".join(parts)


def _ext_to_lang(filename: str) -> str:
    mapping = {
        ".py":   "python",  ".java": "java",    ".ts":   "typescript",
        ".js":   "javascript", ".go": "go",     ".rs":   "rust",
        ".kt":   "kotlin",  ".rb":   "ruby",    ".cs":   "csharp",
        ".php":  "php",     ".yaml": "yaml",    ".yml":  "yaml",
        ".json": "json",    ".sql":  "sql",     ".sh":   "bash",
    }
    ext = os.path.splitext(filename)[-1].lower()
    return mapping.get(ext, "")
