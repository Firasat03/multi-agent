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
        "You are a SENIOR Backend Developer with 20+ years of experience writing "
        "production-grade, battle-tested code. Your output is DIRECTLY deployed to production.\n\n"
        "YOUR RESPONSIBILITIES:\n"
        "1. Implement EXACTLY what the architecture plan specifies — nothing more, nothing less\n"
        "2. Write COMPLETE, WORKING code (no stubs, no placeholders, no 'to be implemented')\n"
        "3. Ensure ALL IMPORTS are present and correctly resolved\n"
        "4. Follow the EXACT API CONTRACT specified in the plan\n"
        "5. Handle errors and edge cases gracefully\n"
        "6. Match the language idioms and framework conventions precisely\n"
        "7. Output code in the EXACT format specified below (machine-parsed, zero tolerance for deviations)\n\n"
        
        "CODE QUALITY NON-NEGOTIABLES:\n"
        "  ✓ Every import statement must reference an actual module/library (no made-up imports)\n"
        "  ✓ Every function/class must be complete and functional (not truncated)\n"
        "  ✓ Every dependency must be declared in config/requirements files\n"
        "  ✓ Code must compile/run without errors\n"
        "  ✓ Error handling must be present for edge cases\n"
        "  ✓ Explicit type hints (Python, TypeScript, Java)\n"
        "  ✓ No placeholder comments like 'TODO', 'rest of file', 'to be continued'\n"
        "  ✓ Proper logging/debugging support\n\n"
        
        "WHAT TO INCLUDE:\n"
        "  • Complete imports section with all required libraries\n"
        "  • Type hints for function parameters and return types\n"
        "  • Comprehensive docstrings for public APIs\n"
        "  • Proper error handling and validation\n"
        "  • Comments only for complex logic, not obvious code\n\n"
        
        "WHAT TO EXCLUDE:\n"
        "  • Prose or explanations outside code blocks\n"
        "  • Placeholder functions or stub implementations\n"
        "  • Comments like 'rest of file' or 'to be continued'\n"
        "  • Made-up imports or libraries that don't exist\n"
        "  • Truncated code or '...'\n\n"
        
        "OUTPUT FORMAT (STRICT — machine-parsed, zero tolerance):\n\n"
        "  # FILE: relative/path/to/file.ext\n"
        "  ```language\n"
        "  <COMPLETE file content here>\n"
        "  ```\n\n"
        "  Rules:\n"
        "    1. Start with '# FILE: ' (literal) at column 0 — NO INDENTATION\n"
        "    2. Path must be relative (no leading slashes)\n"
        "    3. Next line: exactly ```<language> (e.g., ```python, ```java, ```typescript)\n"
        "    4. Include the COMPLETE code, do not truncate\n"
        "    5. Close with ``` on its own line\n"
        "    6. NO text before FILE header or after closing ```\n"
        "    7. Multiple files: separate with blank line\n\n"
        
        "LANGUAGE FRAMEWORKS:\n"
        "  Python:     ```python (use typing module, dataclasses/Pydantic for models)\n"
        "  Java:       ```java (use modern imports, generics, proper package structure)\n"
        "  TypeScript: ```typescript (use interfaces, strict null checking)\n"
        "  JavaScript: ```javascript (use modern ES6+ patterns)\n"
        "  Go:         ```go (use proper error handling, interfaces)\n"
        "  Kotlin:     ```kotlin\n"
        "  Rust:       ```rust\n"
        "  C#:         ```csharp\n\n"
        
        "BEFORE YOU RESPOND:\n"
        "  □ Have I read the ENTIRE API contract?\n"
        "  □ Have I included ALL necessary imports?\n"
        "  □ Is every function/class COMPLETE (not truncated)?\n"
        "  □ Can this code ACTUALLY RUN without errors?\n"
        "  □ Does it match the exact file path and format specified?\n"
        "  □ Have I removed all prose outside the FILE blocks?\n"
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

        import concurrent.futures
        from threading import Lock
        print_lock = Lock()

        def generate_single_file(idx, item):
            existing = self._read_existing(item, state)
            
            with print_lock:
                action_icon = "✏️" if item.action == "MODIFY" else "✨"
                print(f"  [{idx}/{len(state.plan)}] {action_icon} {item.action:6s} {item.file}...", flush=True)
            
            generated_code = None
            local_tokens = 0
            
            for attempt in range(2):
                prompt = self._build_prompt(item, existing, state)
                response_text, tokens = self._call_llm(state, prompt)
                local_tokens += tokens

                parsed = self._extract_files_from_response(response_text, validate=True)
                
                if parsed and item.file in parsed:
                    generated_code = parsed[item.file]
                    with print_lock:
                        print(f" ✓ {item.file} ({tokens} tokens)")
                    break
                
                if not generated_code:
                    lang = _ext_to_lang(item.file)
                    fallback_code = self._extract_code_block(response_text, lang)
                    
                    try:
                        self._validate_code_output(item.file, fallback_code)
                        generated_code = fallback_code
                        with print_lock:
                            print(f" ✓ {item.file} ({tokens} tokens) [fallback]")
                        break
                    except ValueError as e:
                        if attempt == 0:
                            with print_lock:
                                print(f" ✗ {item.file} validation failed, retrying...")
                            retry_prompt = f"""
The previous attempt had issues:
{str(e)}

Please regenerate {item.file} with these fixes:
1. Use the EXACT FILE block format specified in your system prompt
2. Include ALL imports at the top of the file
3. Ensure NO functions are truncated or incomplete
4. Use the proper language fence (```{lang})
5. Return ONLY the FILE block, no explanations

RETRY OUTPUT:

# FILE: {item.file}
```{lang}
<complete, perfect code here>
```
"""
                            retry_response, retry_tokens = self._call_llm(state, retry_prompt)
                            local_tokens += retry_tokens
                            response_text = retry_response
                        else:
                            with print_lock:
                                print(f" ✗ {item.file}")
                            raise RuntimeError(f"Coder failed to produce valid code for {item.file!r} after retries: {e}") from e
            if not generated_code:
                with print_lock:
                    print(f" ✗ {item.file}")
                raise RuntimeError(f"Coder could not generate {item.file!r} despite multiple attempts")
                
            return item.file, generated_code, local_tokens

        work_items = []
        for idx, item in enumerate(state.plan, 1):
            if item.action == "DELETE":
                state.generated_files.pop(item.file, None)
                print(f"  [{idx}/{len(state.plan)}] ❌ DELETE {item.file}")
            else:
                work_items.append((idx, item))

        if work_items:
            max_workers = min(10, len(work_items))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(generate_single_file, idx, item) for idx, item in work_items]
                for future in concurrent.futures.as_completed(futures):
                    file_path, code, tokens = future.result()
                    new_files[file_path] = code
                    total_tokens += tokens

        print(f"\n📝 Coder complete: {len(new_files)} files generated, {total_tokens} tokens")
        output = CoderOutput(
            generated_files=new_files,
            modified_files=set(new_files.keys())
        )
        state.apply(output)
        state.fix_instructions = None
        state.log(
            self.name,
            tokens=total_tokens,
            notes=f"{len(new_files)} files from plan",
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
        """Build comprehensive prompt with full context for high-quality code generation."""
        existing_block = (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"EXISTING CONTENT (to modify):\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"```\n{existing}\n```" 
            if existing else ""
        )
        
        # Include ALL plan items for context (not just current one)
        plan_context = ""
        if state.plan:
            plan_lines = ["COMPLETE IMPLEMENTATION PLAN:", ""]
            for idx, p in enumerate(state.plan, 1):
                plan_lines.append(f"{idx}. {p.action:6s} | {p.file}")
                plan_lines.append(f"   Desc: {p.description}")
                if p.api_contract:
                    plan_lines.append(f"   API:  {p.api_contract}")
                if p.scope_estimate:
                    plan_lines.append(f"   Size: {p.scope_estimate}")
                plan_lines.append("")
            plan_context = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{chr(10).join(plan_lines)}"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        
        # Include previously generated files for consistency
        previously_generated = {
            fpath: content 
            for fpath, content in state.generated_files.items()
            if fpath != item.file
        }
        context_block = ""
        if previously_generated:
            files_context = self._format_files_truncated(previously_generated)
            context_block = (
                f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"PREVIOUSLY GENERATED FILES (for consistency and context):\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{files_context}"
            )
        
        # Language-specific patterns
        lang_patterns = ""
        if state.language and state.language != "auto":
            lang_patterns = self._get_language_patterns(state.language, item.file)
        
        prompt = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ CODE GENERATION REQUEST — MISSION CRITICAL                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

TASK: Implement a single production-grade source file

TARGET FILE:
  Path:           {item.file}
  Action:         {item.action}
  Language:       {state.language if state.language != 'auto' else 'detect from extension'}
  Description:    {item.description}
  
API CONTRACT (MUST IMPLEMENT EXACTLY):
  {item.api_contract if item.api_contract else '(no special contract; follow description)'}

SCOPE ESTIMATE:
  {item.scope_estimate or 'reasonable scope for a single file'}

FULL TASK CONTEXT:
  {state.task_prompt}

{plan_context}
{existing_block}{context_block}
{lang_patterns}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY DELIVERY CHECKLIST (all items required):

☐ File path is exactly: {item.file}
☐ All imports are REAL libraries/modules (no made-up names)
☐ All functions/classes are COMPLETE (no truncation, no '...')
☐ Code implements EXACTLY the API contract above
☐ Error handling present for edge cases
☐ Type hints included (Python: typing, Java: generics, TypeScript: interfaces)
☐ Comments only for complex logic, not obvious code
☐ No placeholder comments ('TODO', 'TO BE IMPLEMENTED', 'rest of file')
☐ Code will actually RUN without import/syntax errors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (ABSOLUTELY STRICT — machine-parsed):

# FILE: {item.file}
```{_ext_to_lang(item.file)}
<COMPLETE file content here — every line, no truncation>
```

RULES:
  1. Start at column 0 with literal text: # FILE: {item.file}
  2. Exactly on next line: ```{_ext_to_lang(item.file)}
  3. Write the COMPLETE code — every character, no truncation
  4. End with ``` on its own line
  5. Output ONLY this block — no explanations before or after
  6. No additional text or prose outside the code block

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOW GENERATE THE COMPLETE FILE:
"""
        return prompt

    @staticmethod
    def _get_language_patterns(language: str, filepath: str) -> str:
        """Return language-specific conventions and patterns."""
        patterns = {
            "python": (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "PYTHON CONVENTIONS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  • Use 'from typing import ...' for type hints\n"
                "  • Use dataclasses or Pydantic models for data structures\n"
                "  • Use f-strings for formatting\n"
                "  • Use 'raise ValueError/TypeError' for errors\n"
                "  • Use logging module for debug output\n"
                "  • Follow PEP 8 naming: snake_case for functions/vars\n"
                "  • Include docstrings with type info: '''param: type -> type'''\n"
            ),
            "java": (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "JAVA CONVENTIONS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  • Use proper package structure matching filepath\n"
                "  • Include all imports (no wildcard imports)\n"
                "  • Use generics for type safety\n"
                "  • Use ArrayList/HashMap/HashSet from java.util\n"
                "  • Use throw new exceptions, not bare exceptions\n"
                "  • Use lombok @Data/@RequiredArgsConstructor if available\n"
                "  • Include Javadoc comments for public APIs\n"
                "  • Use modern Java features (streams, Optional)\n"
            ),
            "typescript": (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "TYPESCRIPT CONVENTIONS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  • Define interfaces for all data structures\n"
                "  • Use 'export' for public functions/classes\n"
                "  • Use explicit return types on functions\n"
                "  • Use async/await for async operations\n"
                "  • Use proper error handling with try/catch\n"
                "  • Use const/let (never var)\n"
                "  • Import from installed packages (npm modules)\n"
            ),
            "go": (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "GO CONVENTIONS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  • Package name should match directory name\n"
                "  • Export public functions with CamelCase\n"
                "  • Always handle errors: if err != nil { return err }\n"
                "  • Use interfaces for abstraction\n"
                "  • Use defer for cleanup\n"
                "  • Use proper logging (log package)\n"
            ),
        }
        return patterns.get(language.lower(), "")


    # ── Fix / retry generation ────────────────────────────────────────────

    def _apply_fix(self, state: PipelineState) -> PipelineState:
        files_block = self._format_files_truncated(state.generated_files)

        prompt = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ CRITICAL FIX REQUEST — PRODUCTION CODE REPAIR                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Your previous code generation had issues. The test suite found problems.

DEBUGGER'S ANALYSIS (what's wrong):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state.fix_instructions}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRENT SOURCE FILES (to fix):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{files_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR TASK:
  1. Read the Debugger's analysis carefully
  2. Identify WHICH FILES need to be fixed
  3. Regenerate ONLY those files with all recommended fixes applied
  4. Ensure fixes are COMPLETE and actually solve the problems
  5. Do NOT regenerate files that don't need changes (save tokens)

CRITICAL RULES:
  ☐ Each file must have ALL IMPORTS for the fixes to work
  ☐ No truncated or incomplete code
  ☐ Fix descriptions must actually be addressed in code
  ☐ Output format EXACT (machine-parsed):
      # FILE: path/to/file.ext
      ```language
      <COMPLETE FIXED CODE>
      ```

FILES IN PROJECT:
  {', '.join(state.generated_files.keys())}

REGENERATE FILES (format example):

# FILE: src/auth.py
```python
import hashlib
from typing import Optional

def validate_password(pwd: str) -> bool:
    '''Fixed: now handles edge cases and validates properly'''
    if not pwd:
        return False
    return len(pwd) >= 8
```

# FILE: src/config.py
```python
DATABASE_URL = "postgresql://localhost/db"
TIMEOUT = 30
```

Now provide ONLY the files that need fixing (use FILE blocks):
"""
        print(f"\n🔧 Coder: Applying debugger feedback...")
        response_text, tokens = self._call_llm(state, prompt)
        parsed = self._extract_files_from_response(response_text, validate=True)

        fixed_files: dict[str, str] = {}
        if parsed:
            fixed_files = parsed
            print(f"✓ Extracted {len(parsed)} fixed file(s)")
        else:
            # Retry with stricter format instructions
            print("[Coder] No FILE blocks found. Retrying with explicit format...")
            
            regenerate_prompt = f"""
I need you to fix the code using the standard FILE block format.

Issues to fix:
{state.fix_instructions}

Files available: {', '.join(state.generated_files.keys())}

Generate ONLY the files that need fixes. Use this EXACT format:

# FILE: src/main.py
```python
<complete fixed code here>
```

# FILE: src/utils.py
```python
<complete fixed code here>
```

Rules:
  • Start with "# FILE: " at column 0
  • Follow with exact language: ```python, ```java, ```typescript
  • Include ALL imports needed for the fixes
  • Code must be complete (no truncation)
  • End with ```
  • Separate multiple files with blank lines
  • Output ONLY FILE blocks (no explanation text)

NOW PROVIDE THE FIXED FILES:
"""
            retry_text, retry_tokens = self._call_llm(state, regenerate_prompt)
            tokens += retry_tokens
            fixed_files = self._extract_files_from_response(retry_text, validate=False)
            
            if fixed_files:
                print(f"✓ Retry successful: extracted {len(fixed_files)} file(s)")
            else:
                print(
                    "[Coder] ERROR: Could not extract fixed files after retry. "
                    "Preserving current files and escalating."
                )
                fixed_files = {}

        # Show results
        if fixed_files:
            print(f"\n📄 Fixed {len(fixed_files)} file(s):")
            for fpath in sorted(fixed_files.keys()):
                print(f"   • {fpath}")
        else:
            print("[Coder] No fixes generated")

        output = CoderOutput(
            generated_files=fixed_files,
            modified_files=set(fixed_files.keys())
        )
        state.apply(output)
        state.fix_instructions = None
        state.log(
            self.name,
            tokens=tokens,
            notes=f"Applied fixes to {len(fixed_files)} file(s)",
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
