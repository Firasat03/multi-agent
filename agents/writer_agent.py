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

        py_files = {k: v for k, v in state.generated_files.items() if k.endswith(".py")}
        if py_files:
            state, tokens = self._add_docstrings(state, py_files)
            total_tokens += tokens

        readme_tokens = self._update_readme(state)
        total_tokens += readme_tokens

        changelog_tokens = self._update_changelog(state)
        total_tokens += changelog_tokens

        if state.project_root and is_git_repo(state.project_root):
            git_stage_all(state.project_root)
            git_commit(state.project_root, f"feat: {state.task_prompt[:72]} [auto-generated]")

        output = WriterOutput(docs_updated=True)
        state.apply(output)
        state.status = Status.DONE
        state.log(self.name, tokens=total_tokens, notes="docs written, status=DONE")
        return state

    def _add_docstrings(self, state: PipelineState, py_files: dict) -> tuple[PipelineState, int]:
        files_block = "\n\n".join(
            f"# FILE: {path}\n```python\n{content}\n```"
            for path, content in py_files.items()
        )
        prompt = f"""
Add or improve Google-style docstrings to every public class, method, and function.
Do not change any logic — only add/update docstrings.

{files_block}

Output each updated file as:
# FILE: <path>
```python
<complete updated content>
```
"""
        response_text, tokens = self._call_llm(state, prompt)
        parsed = self._extract_files_from_response(response_text, validate=False)
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
        return state, tokens

    def _update_readme(self, state: PipelineState) -> int:
        readme_path = os.path.join(state.project_root, "README.md") if state.project_root else "README.md"
        existing = read_file(readme_path) if file_exists(readme_path) else ""
        
        # Format the plan for the prompt to give the LLM details about endpoints and files
        plan_details = "\n".join([
            f"- {item.file}: {item.description} (API: {item.api_contract or 'N/A'})"
            for item in state.plan
        ])

        prompt = f"""
Update (or create) a professional, production-grade README.md for this project.

CONEXT:
TASK: {state.task_prompt}
IMPLEMENTED FEATURES:
{plan_details}

YOUR TASK:
Generate a complete README.md that includes:
1. # Project Title (from the task)
2. ## Overview: What this project does.
3. ## Requirements: Necessary runtimes (e.g., Python 3.9+, Node 18, Java 17) and dependencies.
4. ## Setup & Run:
   - Specific commands to install dependencies.
   - Specific commands to start the application.
   - Specific commands to run tests.
5. ## API Documentation:
   - A table or list of all endpoints, their methods, and expected request/response shapes (based on the implemented features listed above).
6. ## Project Structure: Brief description of key files.

EXISTING README (if any):
{existing or "(empty)"}

Output ONLY the complete updated README inside a ```markdown block.
"""
        response_text, tokens = self._call_llm(state, prompt)
        content = self._extract_code_block(response_text, "markdown") or response_text
        if state.project_root:
            try:
                write_file(readme_path, content, project_root=state.project_root)
            except ValueError as exc:
                print(f"[Writer] Skipping unsafe path for README: {exc}")
        return tokens

    def _update_changelog(self, state: PipelineState) -> int:
        from datetime import date
        changelog_path = os.path.join(state.project_root, "CHANGELOG.md") if state.project_root else "CHANGELOG.md"
        existing = read_file(changelog_path) if file_exists(changelog_path) else ""
        prompt = f"""
Add a Keep-a-Changelog entry for today ({date.today().isoformat()}):
TASK: {state.task_prompt}
WHAT WAS DONE: {state.plan_summary}
EXISTING CHANGELOG: {existing or "(empty)"}

Output ONLY the complete updated CHANGELOG.md inside a ```markdown block.
"""
        response_text, tokens = self._call_llm(state, prompt)
        content = self._extract_code_block(response_text, "markdown") or response_text
        if state.project_root:
            try:
                write_file(changelog_path, content, project_root=state.project_root)
            except ValueError as exc:
                print(f"[Writer] Skipping unsafe path for CHANGELOG: {exc}")
        return tokens