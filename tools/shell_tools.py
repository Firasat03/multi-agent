"""
tools/shell_tools.py — Shell / subprocess helpers for BE Multi-Agent Workflow

Provides:
    run_command()           — subprocess wrapper with timeout and encoding safety
    detect_language()       — infer language from file-extension distribution
    run_static_analysis()   — pyflakes (Python), basic checks for other languages
    auto_fix_pyflakes()     — automatically patch simple pyflakes errors in Python
    run_tests()             — invoke language-native test runner
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ─── Caching for static analysis (avoid re-checking unchanged files) ──────────
_ANALYSIS_CACHE: dict[str, dict] = {}  # hash(files) -> result


def _compute_files_hash(files: dict[str, str]) -> str:
    """Compute a hash of file contents for caching purposes."""
    content = json.dumps(files, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


# ─── run_command ──────────────────────────────────────────────────────────────

def run_command(
    cmd: list[str] | str,
    cwd: Optional[str] = None,
    timeout: int = 120,
    env: Optional[dict] = None,
) -> dict:
    """
    Run a shell command and return a result dict.

    Args:
        cmd:     Command as a list of strings or a shell string.
        cwd:     Working directory for the command.
        timeout: Seconds before raising subprocess.TimeoutExpired.
        env:     Optional environment variables (merged with os.environ).

    Returns:
        {returncode: int, stdout: str, stderr: str}
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    use_shell = isinstance(cmd, str)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=use_shell,
            env=merged_env,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s: {cmd}",
        }
    except FileNotFoundError as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {exc}",
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Unexpected error running command: {exc}",
        }


# ─── detect_language ──────────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py":   "python",
    ".java": "java",
    ".kt":   "kotlin",
    ".kts":  "kotlin",
    ".ts":   "nodejs",
    ".js":   "nodejs",
    ".mjs":  "nodejs",
    ".cjs":  "nodejs",
    ".go":   "go",
    ".rs":   "rust",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".php":  "php",
}

# Presence of certain config/build files disambiguates ties (checked in order)
_MARKER_LANG: list[tuple[str, str]] = [
    ("pom.xml",            "java"),
    ("build.gradle",       "java"),
    ("build.gradle.kts",   "kotlin"),
    ("package.json",       "nodejs"),
    ("go.mod",             "go"),
    ("Cargo.toml",         "rust"),
    ("Gemfile",            "ruby"),
    ("composer.json",      "php"),
    ("requirements.txt",   "python"),
    ("setup.py",           "python"),
    ("pyproject.toml",     "python"),
]


def detect_language(files: dict[str, str]) -> str:
    """
    Infer the primary programming language from a dict of {relative_path: content}.

    Strategy:
      1. Check well-known build/config filenames (highest confidence).
      2. Count source file extensions — majority vote wins.
      3. Return 'unknown' if detection fails.

    Returns one of:
        "python" | "java" | "kotlin" | "nodejs" | "go" |
        "rust"   | "csharp" | "ruby" | "php" | "unknown"
    """
    if not files:
        return "unknown"

    paths = list(files.keys())

    # Phase 1: build file markers (checked by filename only, not glob)
    for marker, lang in _MARKER_LANG:
        if any(Path(p).name == marker for p in paths):
            return lang

    # Phase 2: extension vote — csproj handled separately (glob pattern)
    if any(p.endswith(".csproj") for p in paths):
        return "csharp"

    counts: Counter = Counter()
    for p in paths:
        ext = Path(p).suffix.lower()
        lang = _EXT_TO_LANG.get(ext)
        if lang:
            counts[lang] += 1

    if counts:
        return counts.most_common(1)[0][0]

    return "unknown"


# ─── run_static_analysis ──────────────────────────────────────────────────────

def run_static_analysis(
    files: dict[str, str],
    language: str = "unknown",
) -> dict:
    """
    Run lightweight static analysis on the supplied in-memory files.

    For Python: uses stdlib ``ast`` (always available) to catch syntax errors,
    then pyflakes for undefined names / unused imports if installed.
    For other languages: basic structural sanity check (non-empty, no placeholders).

    Results are cached by file content hash to avoid re-analysis of unchanged code.

    Returns:
        {has_errors: bool, errors: list[str]}
    """
    # Check cache first
    cache_key = _compute_files_hash(files)
    if cache_key in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[cache_key]
    
    errors: list[str] = []

    if language == "python":
        errors.extend(_python_ast_check(files))
        # Only run pyflakes when there are no syntax errors (it would crash otherwise)
        if not errors:
            errors.extend(_python_pyflakes_check(files))
    else:
        errors.extend(_generic_sanity_check(files, language))

    result = {"has_errors": bool(errors), "errors": errors}
    _ANALYSIS_CACHE[cache_key] = result
    return result


def _python_ast_check(files: dict[str, str]) -> list[str]:
    """Parse every .py file with ast.parse — catches syntax errors with zero deps."""
    errors: list[str] = []
    for path, content in files.items():
        if not path.endswith(".py"):
            continue
        try:
            ast.parse(content, filename=path)
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}: SyntaxError: {exc.msg}")
    return errors


def _python_pyflakes_check(files: dict[str, str]) -> list[str]:
    """
    Run pyflakes on Python source files if it is installed.
    Falls back gracefully (empty list) when pyflakes is not available.
    """
    try:
        from pyflakes.checker import Checker
    except ImportError:
        return []

    errors: list[str] = []
    for path, content in files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(content, filename=path)
            checker = Checker(tree, filename=path)
            for msg in checker.messages:
                errors.append(
                    f"{path}:{msg.lineno}: {msg.__class__.__name__}: "
                    f"{msg.message % msg.message_args}"
                )
        except SyntaxError:
            pass  # Already caught by _python_ast_check
        except Exception:
            pass  # pyflakes internal error — skip silently

    return errors


def _generic_sanity_check(files: dict[str, str], language: str) -> list[str]:
    """
    Minimal sanity checks for non-Python languages:
      - File must not be empty
      - File must not consist only of placeholder comment text
    """
    errors: list[str] = []
    placeholder_re = re.compile(
        r"(TODO|FIXME|PLACEHOLDER|rest of code here|implement this)",
        re.IGNORECASE,
    )

    for path, content in files.items():
        stripped = content.strip()
        if len(stripped) < 10:
            errors.append(f"{path}: File is empty or nearly empty ({len(stripped)} chars).")
            continue

        lines = stripped.splitlines()
        code_lines = [
            l for l in lines
            if l.strip() and not re.match(r"^\s*(//|#|/\*|\*|-->|<!)", l)
        ]
        if not code_lines:
            errors.append(f"{path}: File contains only comments — no implementation found.")
            continue

        if placeholder_re.search(content):
            errors.append(f"{path}: File contains unresolved placeholder text.")

    return errors


# ─── auto_fix_pyflakes ────────────────────────────────────────────────────────

def auto_fix_pyflakes(
    files: dict[str, str],
    errors: list[str],
) -> tuple[dict[str, str], list[str]]:
    """
    Attempt to auto-fix simple pyflakes errors in Python files:
      - 'imported but unused'  → remove the offending import line
      - 'redefinition of unused name' (import only) → remove the import line

    Args:
        files:  Dict of {relative_path: content}.
        errors: List of error strings formatted as
                "<path>:<lineno>: <ErrorClass>: <message>"

    Returns:
        (patched_files, remaining_errors)
        ``patched_files`` contains only files that were actually modified.
    """
    # Group fixable errors by file
    file_errors: dict[str, list[tuple[int, str, str]]] = {}
    unparseable: list[str] = []

    for err in errors:
        m = re.match(r"^([^:]+):(\d+):\s*(\w+):\s*(.+)$", err)
        if not m:
            unparseable.append(err)
            continue
        path, lineno_str, cls, msg = (
            m.group(1), m.group(2), m.group(3), m.group(4)
        )
        file_errors.setdefault(path, []).append((int(lineno_str), cls, msg))

    patched: dict[str, str] = {}
    remaining: list[str] = list(unparseable)

    _FIXABLE_CLASSES = {"UnusedImport", "ImportationFrom", "RedefinedWhileUnused"}
    _FIXABLE_PHRASES = ("imported but unused", "redefinition of unused")

    for path, errs in file_errors.items():
        if path not in files:
            remaining.extend(
                f"{path}:{lineno}: {cls}: {msg}" for lineno, cls, msg in errs
            )
            continue

        lines = files[path].splitlines(keepends=True)
        lines_to_remove: set[int] = set()
        still_broken: list[str] = []

        for lineno, cls, msg in errs:
            idx = lineno - 1  # Convert to 0-based index
            if idx < 0 or idx >= len(lines):
                still_broken.append(f"{path}:{lineno}: {cls}: {msg}")
                continue

            line = lines[idx]
            is_import = bool(re.match(r"^\s*(import |from )\S", line))
            is_fixable_class = cls in _FIXABLE_CLASSES
            is_fixable_phrase = any(phrase in msg for phrase in _FIXABLE_PHRASES)

            if is_import and (is_fixable_class or is_fixable_phrase):
                lines_to_remove.add(idx)
            else:
                still_broken.append(f"{path}:{lineno}: {cls}: {msg}")

        if lines_to_remove:
            patched[path] = "".join(
                l for i, l in enumerate(lines) if i not in lines_to_remove
            )

        remaining.extend(still_broken)

    return patched, remaining


# ─── run_tests ────────────────────────────────────────────────────────────────

def run_tests(project_root: str, language: str) -> dict:
    """
    Invoke the language-native test runner inside project_root.

    Each language lists one or more candidate commands in priority order.
    The first command that does NOT return "command not found" is used.

    Supported runners:
        python  → pytest tests/  (falls back to pytest .)
        java    → mvn test -q
        kotlin  → mvn test -q  (falls back to gradle test)
        nodejs  → npm test
        go      → go test ./...
        rust    → cargo test
        csharp  → dotnet test
        ruby    → bundle exec rspec  (falls back to rspec)
        php     → ./vendor/bin/phpunit  (falls back to phpunit)

    Returns:
        {returncode: int, stdout: str, stderr: str}
    """
    lang = (language or "unknown").lower().strip()

    # Each entry is a list of candidate command lists — tried in order until one works
    runners: dict[str, list[list[str]]] = {
        "python": [
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            [sys.executable, "-m", "pytest", ".",       "-v", "--tb=short", "-q"],
        ],
        "java":   [["mvn", "test", "-q"]],
        "kotlin": [
            ["mvn",    "test", "-q"],
            ["gradle", "test", "--quiet"],
        ],
        "nodejs": [["npm", "test", "--", "--forceExit"]],
        "go":     [["go", "test", "./...", "-v"]],
        "rust":   [["cargo", "test"]],
        "csharp": [["dotnet", "test", "--nologo", "--verbosity", "minimal"]],
        "ruby":   [
            ["bundle", "exec", "rspec", "--format", "progress"],
            ["rspec", "--format", "progress"],
        ],
        "php":    [
            ["./vendor/bin/phpunit", "--colors=never"],
            ["phpunit", "--colors=never"],
        ],
    }

    candidates = runners.get(lang)
    if not candidates:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": (
                f"No test runner configured for language '{lang}'. "
                "Supported: python, java, kotlin, nodejs, go, rust, csharp, ruby, php. "
                "To add support for this language, edit tools/shell_tools.py:run_tests()"
            ),
        }

    # Ensure dependencies are installed before running tests
    if lang == "python":
        req_file = Path(project_root) / "requirements.txt"
        if req_file.exists():
            print(f"[Shell] Installing Python dependencies from requirements.txt ...")
            run_command(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=project_root,
                timeout=120,
            )
    elif lang == "nodejs":
        pkg_file = Path(project_root) / "package.json"
        if pkg_file.exists():
            print(f"[Shell] Installing Node.js dependencies via npm install ...")
            run_command(["npm", "install", "--quiet"], cwd=project_root, timeout=120)

    last_result: dict = {"returncode": 1, "stdout": "", "stderr": ""}
    for cmd in candidates:
        result = run_command(cmd, cwd=project_root, timeout=300)
        last_result = result
        # A returncode of -1 with "not found" means the binary is absent — try next
        if not (
            result["returncode"] == -1
            and (
                "not found" in result["stderr"].lower()
                or "no such file" in result["stderr"].lower()
            )
        ):
            break

    return last_result