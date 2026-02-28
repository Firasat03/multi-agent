"""
agents/writer_agent.py — Writer Agent

Changes vs original:
  - Returns typed WriterOutput; state.apply(output) merges it
  - Uses _extract_files_from_response for docstring updates
"""

from __future__ import annotations

import os

from agents.base_agent import BaseAgent
from config import Status
from state import PipelineState, WriterOutput
from tools.file_tools import write_file, read_file, file_exists
from tools.git_tools import is_git_repo, git_stage_all, git_commit


class WriterAgent(BaseAgent):
    name = "Writer"
    system_role = (
        "You are a Technical Writer and Documentation Specialist for backend systems. "
        "You write clear, concise, and accurate documentation. "
        "For docstrings, follow Google-style format. "
        "For README sections, use clean Markdown. "
        "For CHANGELOG entries, follow Keep a Changelog format."
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.WRITING
        total_tokens = 0

        # Step 1: Add language-specific docstrings (Python only for now)
        py_files = {k: v for k, v in state.generated_files.items() if k.endswith(".py")}
        if py_files:
            state, tokens = self._add_docstrings(state, py_files)
            total_tokens += tokens

        # Step 2: Generate README (language-agnostic, always runs)
        print(f"\n   📝 Generating README.md...")
        readme_tokens = self._update_readme(state)
        total_tokens += readme_tokens

        # Step 3: Generate CHANGELOG (language-agnostic, always runs)
        print(f"   📝 Generating CHANGELOG.md...")
        changelog_tokens = self._update_changelog(state)
        total_tokens += changelog_tokens

        # Step 4: Git commit (if project is a git repo)
        if state.project_root and is_git_repo(state.project_root):
            print(f"   💾 Committing changes to git...")
            git_stage_all(state.project_root)
            git_commit(state.project_root, f"feat: {state.task_prompt[:72]} [auto-generated]")

        output = WriterOutput(docs_updated=True)
        state.apply(output)
        state.status = Status.WRITING  # Keep as WRITING; Orchestrator will advance to DEVOPS or DONE
        state.log(self.name, tokens=total_tokens, notes="docs written, status=WRITING")
        return state

    def _add_docstrings(self, state: PipelineState, py_files: dict) -> tuple[PipelineState, int]:
        """
        Add or improve Google-style docstrings to Python files only.
        This is Python-specific; for other languages, docstring addition is skipped.
        """
        print(f"   📚 Adding docstrings to {len(py_files)} Python file(s)...")
        
        files_block = "\n\n".join(
            f"# FILE: {path}\n```python\n{content}\n```"
            for path, content in py_files.items()
        )
        prompt = f"""
Add or improve Google-style docstrings to every public class, method, and function.
Do not change any logic — only add/update docstrings.

Python files to document:
{files_block}

Output each updated file as:
# FILE: <path>
```python
<complete updated content with improved docstrings>
```
"""
        response_text, tokens = self._call_llm(state, prompt)
        parsed = self._extract_files_from_response(response_text, validate=False)
        if parsed:
            for file_path, content in parsed.items():
                state.generated_files[file_path] = content
                if state.project_root:
                    try:
                        write_file(
                            os.path.join(state.project_root, file_path),
                            content,
                            project_root=state.project_root,
                        )
                    except ValueError as exc:
                        print(f"[Writer] Skipping unsafe path {file_path!r}: {exc}")
            print(f"   ✓ Docstrings added to {len(parsed)} file(s)")
        else:
            print(f"   ⚠ No docstring updates parsed from LLM response")
        return state, tokens

    def _update_readme(self, state: PipelineState) -> int:
        readme_path = os.path.join(state.project_root, "README.md") if state.project_root else "README.md"
        existing = read_file(readme_path) if file_exists(readme_path) else ""
        
        # Format the plan for the prompt to give the LLM details about endpoints and files
        plan_details = "\n".join([
            f"- {item.file}: {item.description} (API: {item.api_contract or 'N/A'})"
            for item in state.plan
        ])

        # Detect project language(s) from generated files
        file_extensions = {}
        for file_path in state.generated_files.keys():
            ext = os.path.splitext(file_path)[1].lstrip('.')
            if ext:
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
        
        detected_language = state.language if state.language != "auto" else "unknown"
        file_types_info = ", ".join([f"{ext} ({count})" for ext, count in sorted(file_extensions.items())])

        prompt = f"""
Update (or create) a professional, production-grade README.md for this project.

PROJECT CONTEXT:
TASK: {state.task_prompt}
LANGUAGE/TECH: {detected_language}
FILE TYPES: {file_types_info}
IMPLEMENTED FEATURES:
{plan_details}

YOUR TASK:
Generate a complete README.md that includes:
1. # Project Title (from the task)
2. ## Overview: What this project does.
3. ## Tech Stack: What technologies/languages are used (e.g., {detected_language}).
4. ## Requirements: Necessary runtimes (e.g., Python 3.9+, Node 18, Java 17, C# .NET 6+) and dependencies.
5. ## Setup & Installation:
   - Step-by-step commands to install dependencies (specific to {detected_language}).
   - Specific commands to start the application.
   - Specific commands to run tests (if applicable).
6. ## API Documentation / Usage:
   - A table or list of all endpoints, their methods, and expected request/response shapes (based on the implemented features listed above).
   - Show example requests/responses if applicable.
7. ## Project Structure: Brief description of key files and directories.
8. ## Testing: How to run tests and what test coverage is expected.

IMPORTANT: Make the README specific to the {detected_language} project. Use appropriate package managers, runtimes, and conventions for {detected_language}.

EXISTING README (if any):
{existing or "(empty)"}

Output ONLY the complete updated README inside a ```markdown block.
"""
        response_text, tokens = self._call_llm(state, prompt)
        content = self._extract_code_block(response_text, "markdown") or response_text
        if state.project_root:
            try:
                write_file(readme_path, content, project_root=state.project_root)
                print(f"   ✓ README.md written to {readme_path}")
            except ValueError as exc:
                print(f"[Writer] Skipping unsafe path for README: {exc}")
        else:
            print(f"[Writer] Warning: project_root not set, README not written")
        return tokens

    def _update_changelog(self, state: PipelineState) -> int:
        from datetime import date
        changelog_path = os.path.join(state.project_root, "CHANGELOG.md") if state.project_root else "CHANGELOG.md"
        existing = read_file(changelog_path) if file_exists(changelog_path) else ""
        
        prompt = f"""
Add a Keep-a-Changelog entry for today ({date.today().isoformat()}):

PROJECT: {state.task_prompt}
LANGUAGE: {state.language if state.language != "auto" else "auto-detected"}
WHAT WAS DONE: {state.plan_summary}

EXISTING CHANGELOG:
{existing or "(empty)"}

Add a new [Unreleased] or [VERSION] section at the top with the changes made.
Format your response as:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features from today

### Changed
- Any modifications to existing functionality

### Fixed
- Bug fixes (if any)

{existing or ""}
```

Output ONLY the complete updated CHANGELOG.md inside a ```markdown block.
"""
        response_text, tokens = self._call_llm(state, prompt)
        content = self._extract_code_block(response_text, "markdown") or response_text
        if state.project_root:
            try:
                write_file(changelog_path, content, project_root=state.project_root)
                print(f"   ✓ CHANGELOG.md written to {changelog_path}")
            except ValueError as exc:
                print(f"[Writer] Skipping unsafe path for CHANGELOG: {exc}")
        else:
            print(f"[Writer] Warning: project_root not set, CHANGELOG not written")
        return tokens