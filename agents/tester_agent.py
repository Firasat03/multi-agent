"""
agents/tester_agent.py — Tester Agent

Changes vs original:
  - Returns typed TesterOutput; Orchestrator calls state.apply(output)
  - Static analysis result is validated before being stored (no silent empty-string storage)
  - Auto-fix pyflakes now re-runs static analysis after patching to confirm fixes
  - Language resolution is separated and logged explicitly
  - Language configurations moved to config.py (single source of truth)
"""

from __future__ import annotations

import os

from agents.base_agent import BaseAgent
from config import Status, LANGUAGE_CONFIG
from state import PipelineState, TesterOutput
from tools.file_tools import write_file
from tools.shell_tools import (
    auto_fix_pyflakes,
    detect_language,
    run_static_analysis,
    run_tests,
)


class TesterAgent(BaseAgent):
    name = "Tester"

    @property
    def system_role(self) -> str:  # type: ignore[override]
        return (
            "You are a meticulous QA Engineer specialising in production-grade backend testing. "
            "You write comprehensive, idiomatic test suites using the test framework native "
            "to the project's language. Your tests cover:\n"
            "  - Happy paths for every public function / endpoint\n"
            "  - Edge cases: empty inputs, boundary values, null/None, empty collections\n"
            "  - Error conditions: invalid data, unauthorized, not-found, server errors\n"
            "  - Idempotency: verify repeated calls produce the same result\n"
            "  - Contract tests: response shape matches the API contract in the plan\n\n"
            "Rules:\n"
            "  - Mock ALL external dependencies (DB, HTTP, filesystem, time, env vars)\n"
            "  - Never depend on a live database, network, or real filesystem\n"
            "  - Each test must document the scenario it covers (docstring or comment)\n"
            "  - Use the idiomatic setup/teardown mechanism for the language\n"
            "  - If there are many files, prioritize testing core business logic and critical edges.\n"
            "  - Check if the project needs new test-scoped dependencies (e.g., pytest-mock, Mockito) and note them.\n"
            "Output each file inside a fenced code block preceded by: # FILE: <relative/path>"
        )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.TESTING

        print(f"\n🧪 Tester: Starting test suite generation and validation...")
        language = self._resolve_language(state)
        print(f"   Language: {language}")
        output = TesterOutput()

        # ── Step 1: Static Analysis ───────────────────────────────────────
        print(f"\n   📊 Step 1/4: Running static analysis...")
        output = self._run_static_analysis(state, language, output)

        # Python-only: auto-fix simple pyflakes errors
        if output.static_analysis_output and language == "python":
            output, state = self._try_auto_fix_pyflakes(state, language, output)

        # Static errors remain → skip test runner
        if output.static_analysis_output:
            error_count = len(output.static_analysis_output.split('\n'))
            print(f"   ❌ {error_count} static error(s) found - test run skipped")
            state.apply(output)
            state.log(
                self.name,
                notes=f"STATIC ERRORS — test run skipped (attempt {state.retry_count + 1})",
            )
            return state

        # ── Step 2: Generate test files ───────────────────────────────────
        print(f"\n   📊 Step 2/4: Generating test files...")
        # On first run: generate tests for all files
        # On retry: only regenerate tests for files that were modified by Coder
        should_regenerate = not state.test_files or (
            state.retry_count > 0 and state.modified_source_files
        )
        if should_regenerate:
            # Determine which source files to generate tests for
            files_to_test = (
                state.modified_source_files
                if (state.retry_count > 0 and state.modified_source_files)
                else None  # None = all files
            )
            output = self._generate_tests(state, language, output, files_to_test=files_to_test)

        # ── Step 3: Flush to disk ─────────────────────────────────────────
        print(f"\n   📊 Step 3/4: Writing test files to disk...")
        # Always flush to disk — if project_root is empty, use "."
        flush_root = state.project_root or "."
        self._flush_to_disk(state, output, language, project_root=flush_root)
        print(f"   ✓ Test files written")

        # ── Step 4: Run tests ─────────────────────────────────────────────
        print(f"\n   📊 Step 4/4: Running test suite...")
        result = run_tests(project_root=flush_root, language=language)
        output.test_output = result

        if result.get("returncode", 1) != 0:
            output.error_log = (
                f"STDOUT:\n{result.get('stdout', '')}\n\nSTDERR:\n{result.get('stderr', '')}"
            )
            state.apply(output)
            # Extract error message for console (first 500 chars to show more context)
            error_msg = result.get('stderr', '') or result.get('stdout', '')
            error_msg = error_msg[:500] if error_msg else ""
            print(f"\n   ❌ Tests FAILED")
            if error_msg:
                # Show first line, or first 500 chars if it's all one line
                first_line = error_msg.split('\n')[0] if '\n' in error_msg else error_msg
                if len(first_line) > 100:
                    # If first line is too long, truncate it
                    print(f"   Error: {first_line[:100]}...")
                else:
                    print(f"   Error: {first_line}")
            state.log(self.name, notes=f"RUNTIME FAIL (attempt {state.retry_count + 1})")
        else:
            output.error_log = None
            state.apply(output)
            print(f"   ✅ All tests PASSED")
            state.log(self.name, notes="PASS — all tests green")

        return state

    # ── Language resolution ────────────────────────────────────────────────

    def _resolve_language(self, state: PipelineState) -> str:
        if state.language and state.language != "auto":
            return state.language.lower().strip()
        detected = detect_language(state.generated_files)
        print(f"[Tester] Auto-detected language: {detected!r}")
        state.language = detected
        return detected

    # ── Static analysis ───────────────────────────────────────────────────

    def _run_static_analysis(
        self, state: PipelineState, language: str, output: TesterOutput
    ) -> TesterOutput:
        from rich.console import Console
        console = Console()
        console.rule("[bold yellow]🔬 Static Analysis[/bold yellow]")

        result = run_static_analysis(state.generated_files, language=language)

        if result["has_errors"]:
            error_block = "\n".join(result["errors"])
            output.static_analysis_output = error_block
            output.error_log = f"STATIC ANALYSIS ERRORS:\n{error_block}"
            console.print(
                f"[red]❌ Static analysis: {len(result['errors'])} error(s)[/red]\n{error_block}"
            )
        else:
            output.static_analysis_output = None
            console.print("[green]✅ Static analysis passed[/green]")

        return output

    # ── Pyflakes auto-fix ─────────────────────────────────────────────────

    def _try_auto_fix_pyflakes(
        self, state: PipelineState, language: str, output: TesterOutput
    ) -> tuple[TesterOutput, PipelineState]:
        from rich.console import Console
        console = Console()

        errors = output.static_analysis_output.splitlines() if output.static_analysis_output else []
        patched, remaining = auto_fix_pyflakes(state.generated_files, errors)

        fixed_count = len(errors) - len(remaining)
        if fixed_count > 0:
            # Update state files immediately so re-analysis sees the patch
            state.generated_files.update(patched)
            console.print(
                f"[cyan]🔧 Auto-fixed {fixed_count} pyflakes issue(s) "
                f"({len(remaining)} remain)[/cyan]"
            )
            if remaining:
                output.static_analysis_output = "\n".join(remaining)
                output.error_log = f"STATIC ANALYSIS ERRORS:\n{output.static_analysis_output}"
            else:
                output.static_analysis_output = None
                output.error_log = None

        return output, state

    # ── Test generation ───────────────────────────────────────────────────

    def _generate_tests(
        self, state: PipelineState, language: str, output: TesterOutput,
        files_to_test: set[str] | None = None
    ) -> TesterOutput:
        """
        Generate test files for the given source files.
        """
        lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["unknown"])
        framework   = lang_config["test_framework"]
        test_folder = lang_config["test_folder"]
        test_ext    = lang_config["test_extension"]

        source_files = state.generated_files.items()
        if files_to_test:
            source_files = [
                (path, content) for path, content in source_files
                if path in files_to_test
            ]
            print(f"   ℹ️  Regenerating tests for {len(files_to_test)} modified file(s)")
        else:
            print(f"   ℹ️  Generating tests for {len(state.generated_files)} file(s)")

        if not source_files:
            output.test_files = {}
            return output

        import concurrent.futures
        from threading import Lock
        print_lock = Lock()
        
        total_tokens = 0
        new_test_files = {}

        def resolve_test_filename(source_path: str) -> str:
            import os
            base = os.path.basename(source_path)
            name, _ = os.path.splitext(base)
            if language == "python":
                return f"{test_folder}test_{name}{test_ext}"
            elif language in ["java", "kotlin"]:
                return f"{test_folder}{name}Test{test_ext}"
            elif language == "go":
                name = name.rstrip("_test") # just in case
                return f"{name}_test.go"
            elif language == "ruby":
                return f"{test_folder}{name}{test_ext}"
            elif language == "php":
                return f"{test_folder}{name.capitalize()}Test.php"
            elif language == "csharp":
                return f"{test_folder}{name}Tests{test_ext}"
            else:
                return f"{test_folder}{name}{test_ext}"

        def generate_test_for_file(path, content):
            expected_test_file = resolve_test_filename(path)
            
            with print_lock:
                print(f"     🧪 Generating tests for {path}...", flush=True)

            files_block = f"# FILE: {path}\n```\n{content}\n```"
            
            prompt = f"""
Write a comprehensive, production-grade test suite for the following specific backend file.

LANGUAGE: {language}
TEST FRAMEWORK: {framework}
TEST FOLDER: {test_folder}
TEST FILE EXTENSION: {test_ext}

TASK CONTEXT: {state.task_prompt}

TARGET FILE TO TEST:
{files_block}

INSTRUCTIONS:
- Generate a comprehensive but token-efficient test suite specifically for this file.
- Ensure all external mocks are correctly implemented.
- If you need new test dependencies that aren't in the project, mention them in a comment.

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

EXAMPLE (if writing Python tests with pytest):

# FILE: {expected_test_file}
```{language}
import pytest
from unittest.mock import Mock, patch
from auth.login import login

class TestLogin:
    def test_login_with_valid_credentials_returns_token(self):
        result = login("user", "pass")
        assert "token" in result
        assert "user_id" in result
    
    def test_login_with_empty_username_raises_error(self):
        with pytest.raises(ValueError):
            login("", "pass")
```

CRITICAL RULES:
1. Each file: '# FILE: {expected_test_file}' at column 0 (no indentation)
2. Next line: ```{language}
3. Write COMPLETE test code (no truncation, no comments like "rest of tests")
4. Close with ``` on its own line
5. Output ONLY FILE blocks — NO prose, NO explanations before/after
6. Every test must have a clear name and test ONE scenario
7. Mock ALL external dependencies

FALLBACK (if you cannot generate valid tests):
# FILE: {expected_test_file}
```{language}
//placeholder test file
```

─────────────────────────────────────────────────────────────────────

NOW OUTPUT ONLY FILE BLOCKS:
"""
            response_text, local_tokens = self._call_llm(state, prompt)
            
            parsed = self._extract_files_from_response(response_text, validate=False)
            if not parsed:
                with print_lock:
                    print(f"[Tester] Warning: No test FILE blocks found for {path}. Retrying...")
                retry_prompt = f"Output test files in EXACT format. ONLY FILE blocks, NO prose.\n\nEXPECTED OUTPUT:\n# FILE: {expected_test_file}\n```{language}\n<your test code here>\n```\n\nTARGET FILE:\n{files_block}"
                retry_text, retry_tokens = self._call_llm(state, retry_prompt)
                local_tokens += retry_tokens
                parsed = self._extract_files_from_response(retry_text, validate=False)
                
            if not parsed:
                with print_lock:
                    print(f"[Tester] ERROR: Could not generate valid test files for {path}. Creating placeholder.")
                parsed = {expected_test_file: "//placeholder test file\n"}
                
            return parsed, local_tokens
            
        if source_files:
            max_workers = min(10, len(source_files))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(generate_test_for_file, path, content) for path, content in source_files]
                for future in concurrent.futures.as_completed(futures):
                    parsed_result, tokens = future.result()
                    new_test_files.update(parsed_result)
                    total_tokens += tokens
                
        output.test_files.update(new_test_files)
        state.log(
            self.name,
            tokens=total_tokens,
            notes=f"{len(output.test_files)} test file(s) generated",
        )
        return output

    # ── File flush ────────────────────────────────────────────────────────

    def _flush_to_disk(
        self, state: PipelineState, output: TesterOutput, language: str, project_root: str = "."
    ) -> None:
        root = project_root
        for rel_path, content in state.generated_files.items():
            try:
                write_file(os.path.join(root, rel_path), content, project_root=root)
            except ValueError as exc:
                print(f"[Tester] Skipping unsafe source path {rel_path!r}: {exc}")
        for rel_path, content in output.test_files.items():
            try:
                write_file(os.path.join(root, rel_path), content, project_root=root)
            except ValueError as exc:
                print(f"[Tester] Skipping unsafe test path {rel_path!r}: {exc}")

        if language == "python":
            init_path = os.path.join(root, "tests", "__init__.py")
            if not os.path.exists(init_path):
                write_file(init_path, "", project_root=root)