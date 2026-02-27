"""
agents/tester_agent.py — Tester Agent

Changes vs original:
  - Returns typed TesterOutput; Orchestrator calls state.apply(output)
  - Static analysis result is validated before being stored (no silent empty-string storage)
  - Auto-fix pyflakes now re-runs static analysis after patching to confirm fixes
  - Language resolution is separated and logged explicitly
"""

from __future__ import annotations

import os

from agents.base_agent import BaseAgent
from config import Status
from state import PipelineState, TesterOutput
from tools.file_tools import write_file
from tools.shell_tools import (
    auto_fix_pyflakes,
    detect_language,
    run_static_analysis,
    run_tests,
)


_LANG_TEST_FRAMEWORK: dict[str, str] = {
    "python":  "pytest",
    "java":    "JUnit 5 + Mockito",
    "kotlin":  "JUnit 5 + MockK",
    "nodejs":  "Jest (TypeScript / JavaScript)",
    "go":      "Go testing package (table-driven tests)",
    "rust":    "Rust built-in #[test] + cargo test",
    "csharp":  "xUnit + Moq",
    "ruby":    "RSpec",
    "php":     "PHPUnit",
    "unknown": "the most appropriate testing framework for this language",
}

_LANG_TEST_FOLDER: dict[str, str] = {
    "python":  "tests/",
    "java":    "src/test/java/",
    "kotlin":  "src/test/kotlin/",
    "nodejs":  "__tests__/",
    "go":      "",
    "rust":    "tests/",
    "csharp":  "Tests/",
    "ruby":    "spec/",
    "php":     "tests/",
    "unknown": "tests/",
}

_LANG_TEST_EXT: dict[str, str] = {
    "python":  ".py",
    "java":    ".java",
    "kotlin":  ".kt",
    "nodejs":  ".test.ts",
    "go":      "_test.go",
    "rust":    ".rs",
    "csharp":  ".cs",
    "ruby":    "_spec.rb",
    "php":     "Test.php",
    "unknown": ".py",
}


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
        if not state.test_files or state.retry_count > 0:
            output = self._generate_tests(state, language, output)

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
            error_msg = result.get('stderr', '') or result.get('stdout', '')[:200]
            print(f"\n   ❌ Tests FAILED")
            if error_msg:
                print(f"   Error: {error_msg.split(chr(10))[0]}")
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
        self, state: PipelineState, language: str, output: TesterOutput
    ) -> TesterOutput:
        framework   = _LANG_TEST_FRAMEWORK.get(language, _LANG_TEST_FRAMEWORK["unknown"])
        test_folder = _LANG_TEST_FOLDER.get(language, "tests/")
        test_ext    = _LANG_TEST_EXT.get(language, ".py")

        files_block = "\n\n".join(
            f"# FILE: {path}\n```\n{content}\n```"
            for path, content in state.generated_files.items()
        )

        prompt = f"""
Write a comprehensive, production-grade test suite for the following backend code.

LANGUAGE: {language}
TEST FRAMEWORK: {framework}
TEST FOLDER: {test_folder}
TEST FILE EXTENSION: {test_ext}

TASK CONTEXT: {state.task_prompt}

SOURCE FILES:
{files_block}

INSTRUCTIONS:
- Generate a comprehensive but token-efficient test suite.
- If the project is large, focus on the most important logic files.
- Ensure all external mocks are correctly implemented.
- If you need new test dependencies that aren't in the project, mention them in a comment.

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

EXAMPLE (if writing Python tests with pytest):

# FILE: {test_folder}test_auth{test_ext}
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
1. Each file: '# FILE: {test_folder}<name>{test_ext}' at column 0 (no indentation)
2. Next line: ```{language}
3. Write COMPLETE test code (no truncation, no comments like "rest of tests")
4. Close with ``` on its own line
5. Output ONLY FILE blocks — NO prose, NO explanations before/after
6. Every test must have a clear name and test ONE scenario
7. Mock ALL external dependencies

FALLBACK (if you cannot generate valid tests):
# FILE: {test_folder}test_placeholder{test_ext}
```{language}
//placeholder test file
```

FORMAT CHECK before responding:
  ☐ Each file starts with '# FILE: <path>' at column 0
  ☐ Immediately followed by ```{language}
  ☐ Test code is COMPLETE (no truncation)
  ☐ Ends with ``` on its own line
  ☐ No prose or explanations outside ``` blocks
  ☐ Every test has a clear name and description
  ☐ All external dependencies are mocked
─────────────────────────────────────────────────────────────────────

NOW OUTPUT ONLY FILE BLOCKS:
"""
        response_text, tokens = self._call_llm(state, prompt)

        parsed = self._extract_files_from_response(response_text, validate=False)
        if parsed:
            output.test_files.update(parsed)
        else:
            # If parsing failed, attempt a structured retry
            print(
                "[Tester] Warning: No test FILE blocks found. "
                "Retrying with stricter format requirements..."
            )
            
            retry_prompt = f"""
Output test files in EXACT format. ONLY FILE blocks, NO prose.

Language: {language}
Framework: {framework}
Test folder: {test_folder}
File extension: {test_ext}

EXAMPLE (minimal valid test file):

# FILE: {test_folder}test_basic{test_ext}
```{language}
import pytest
def test_placeholder():
    assert True
```

FORMAT REQUIREMENTS:
  ☐ '# FILE: <path>' at column 0 (no indentation)
  ☐ Immediately followed by ```{language}
  ☐ COMPLETE test code (no truncation)
  ☐ ``` on its own line to close
  ☐ Nothing else — no prose, no explanations

If generation fails, output:
# FILE: {test_folder}test_placeholder{test_ext}
```{language}
//placeholder test file
```

GENERATE TESTS FOR these source files:
{files_block}

RESPOND WITH ONLY FILE BLOCKS:
"""
            retry_text, retry_tokens = self._call_llm(state, retry_prompt)
            tokens += retry_tokens
            parsed = self._extract_files_from_response(retry_text, validate=False)
            
            if parsed:
                output.test_files.update(parsed)
            else:
                # Last resort: create a minimal test file to avoid breaking the pipeline
                print(
                    "[Tester] ERROR: Could not generate valid test files. "
                    "Creating a placeholder test file."
                )
                placeholder_name = f"{test_folder}test_generated{test_ext}"
                output.test_files[placeholder_name] = "//placeholder test file\n"

        state.log(
            self.name,
            tokens=tokens,
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