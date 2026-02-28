"""
agents/architect_agent.py — Architect Agent

Changes vs original:
  - Uses _call_llm_structured() for the JSON plan — eliminates regex-based
    JSON extraction and the silent parse-failure fallback
  - Returns typed ArchitectOutput; Orchestrator calls state.apply(output)
  - Validates that every PlanItem has required fields before returning
  - KB unavailability is now logged explicitly rather than silently swallowed
"""

from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from config import Status
from state import ArchitectOutput, PipelineState, PlanItem
from tools.file_tools import file_tree
from tools.mcp_client import get_client
from tools.shell_tools import detect_language


class ArchitectAgent(BaseAgent):
    name = "Architect"
    system_role = (
        "You are a Senior Backend Architect with 15+ years of experience designing "
        "production-grade, cloud-native backend systems in Python, Java, Node.js/TypeScript, "
        "Go, Kotlin, Rust, C#, Ruby, PHP, and any other backend language or framework.\n\n"
        "Your job is to analyse a task and produce:\n"
        "  1. A precise implementation plan (JSON array)\n"
        "  2. A numbered Task Checklist\n"
        "  3. A concise human-readable summary\n\n"
        "Think carefully about: files to create/modify, API contracts, DB schema changes, "
        "auth/security considerations, error handling strategy, and scope of work.\n\n"
        "CRITICAL — CONFIGURATION FILES:\n"
        "  Every backend project needs configuration and tooling files to compile, run, "
        "and test. Always include ALL required config files for the detected language and "
        "framework — compiler configs, build descriptors, dependency manifests, "
        "test runner configs, linter configs, and environment templates.\n"
        "  A project that is missing its build or compiler config file CANNOT run. "
        "If you are unsure whether a config file is needed, include it.\n\n"
        "Never skip error handling or validation. "
        "Always output all three parts in the exact format requested."
    )

    _PLAN_SCHEMA = (
        "Array of objects, each with: "
        "file (string), action (CREATE|MODIFY|DELETE), description (string), "
        "api_contract (string or \"\"), scope_estimate (string or \"\")"
    )

    _AUDIT_SCHEMA = (
        "Array of objects, each with: "
        "file (string — relative path), "
        "description (string — what this config file does and why it is required)"
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.ARCHITECT

        # ── Gather context ────────────────────────────────────────────────
        tree = ""
        if state.project_root:
            try:
                tree = file_tree(state.project_root)
            except Exception:
                tree = "(could not read project tree)"

        # ── MCP: optional knowledge base query ────────────────────────────
        kb_context = ""
        try:
            mcp = get_client("architect")
            if "knowledge-base" in mcp.list_allowed_servers():
                result = mcp.call("knowledge-base", "query", query=state.task_prompt)
                if result.get("results"):
                    kb_context = "\nRelevant past patterns:\n" + "\n".join(
                        str(r) for r in result["results"]
                    )
                else:
                    print("[Architect] Knowledge base returned no results (not configured).")
        except Exception as exc:
            print(f"[Architect] Knowledge base unavailable: {exc}")

        # ── Feedback context ──────────────────────────────────────────────
        feedback_block = ""
        if state.user_feedback:
            feedback_block = (
                f"\n\nUSER FEEDBACK ON PREVIOUS PLAN:\n{state.user_feedback}\n"
                "Incorporate ALL feedback. Do not re-produce rejected items."
            )

        lang_hint = (
            f"\nTARGET LANGUAGE: {state.language}  "
            "(Use idiomatic conventions, file layout, and dependency management for this language.)"
            if state.language and state.language != "auto"
            else ""
        )

        # ── Step 1: JSON plan via structured output ───────────────────────
        plan_prompt = f"""
Task: {state.task_prompt}
{lang_hint}
Project file tree:
{tree or "(empty / new project)"}
{kb_context}
{feedback_block}

Produce a JSON ARRAY (and nothing else) — a list of plan items, each with:
  - "file":           relative path (e.g. "src/auth/login.ts")
  - "action":         "CREATE" | "MODIFY" | "DELETE"
  - "description":    detailed description of what this file does / what to change
  - "api_contract":   full API signature if applicable, else ""
  - "scope_estimate": approximate lines of code, else ""

IMPORTANT: Include ALL required configuration files for the target language and framework.
This includes but is not limited to: compiler configs, build descriptors, dependency manifests,
test runner configs, linter/formatter configs, and environment variable templates.
"""
        try:
            raw_plan, plan_tokens = self._call_llm_structured(
                state, plan_prompt, schema_hint=self._PLAN_SCHEMA
            )
            # raw_plan may be a dict with a wrapper key or a list directly
            if isinstance(raw_plan, dict):
                # e.g. {"plan": [...]} or {"items": [...]}
                items_list = next(
                    (v for v in raw_plan.values() if isinstance(v, list)), []
                )
            else:
                items_list = raw_plan

            plan = [PlanItem(**item) for item in items_list]
        except Exception as exc:
            state.record_failure(
                stage="ARCHITECT",
                agent=self.name,
                error_summary=f"Failed to parse JSON plan: {exc}",
            )
            raise RuntimeError(f"Architect could not produce a valid plan: {exc}") from exc

        if not plan:
            raise RuntimeError("Architect produced an empty plan — nothing to implement.")

        # ── Step 2: LLM config-file audit — no hardcoding needed ─────────
        plan = self._audit_missing_configs(state, plan)

        # ── Step 3: Checklist + summary in one prose call ─────────────────
        prose_prompt = f"""
You have already planned these files:
{chr(10).join(f'  {i.action} {i.file}: {i.description[:60]}' for i in plan)}

Now produce TWO sections:

CHECKLIST_START
1. <First concrete implementation step>
2. <Second step>
...
N. <Final step>
CHECKLIST_END

Then, after CHECKLIST_END, write a concise bullet-point plan summary a developer can
approve or reject. Include: files list, API shape, DB changes, dependencies, security
considerations, error handling strategy, estimated scope.
"""
        prose_text, prose_tokens = self._call_llm(state, prose_prompt)

        checklist_match = re.search(
            r"CHECKLIST_START\s*(.+?)\s*CHECKLIST_END", prose_text, re.DOTALL
        )
        task_checklist = checklist_match.group(1).strip() if checklist_match else ""

        summary_match = re.search(r"CHECKLIST_END\s*(.+)$", prose_text, re.DOTALL)
        plan_summary = summary_match.group(1).strip() if summary_match else prose_text

        output = ArchitectOutput(
            plan=plan,
            plan_summary=plan_summary,
            task_checklist=task_checklist,
        )
        state.apply(output)
        state.replan_count += 1
        state.plan_approved = False

        # ── Cache language detection once — downstream agents read state.language ─
        if state.language in (None, "", "auto"):
            # Infer from the plan's file extensions (no generated_files yet at this stage)
            plan_files = {item.file: "" for item in plan}
            detected = detect_language(plan_files)
            if detected != "unknown":
                state.language = detected
                print(f"[Architect] Detected language from plan: {detected!r}")

        state.log(
            self.name,
            notes=f"{len(plan)} plan items (replan #{state.replan_count})",
            tokens=plan_tokens + prose_tokens,
        )
        return state

    # ── Config-file audit ─────────────────────────────────────────────────────

    def _audit_missing_configs(
        self, state: PipelineState, plan: list[PlanItem]
    ) -> list[PlanItem]:
        """
        Ask the LLM to inspect the current plan and return any required
        configuration or tooling files that are absent.

        WHY this beats a hardcoded registry:
          - The LLM knows what every language/framework/version requires
          - Works for mixed stacks (TypeScript + Docker + Prisma + ESLint)
          - Works for monorepos with per-service configs
          - Handles emerging frameworks with zero code changes
          - Failure is non-fatal: original plan is preserved on any error

        Returns the original plan extended with injected CREATE items.
        """
        planned_files = "\n".join(f"  - {item.file}" for item in plan)

        stack_hint = (
            f"TARGET LANGUAGE/STACK: {state.language}"
            if state.language and state.language not in ("auto", "unknown", "")
            else f"Infer the stack from these planned source files: "
                 f"{[item.file for item in plan[:12]]}"
        )

        audit_prompt = f"""
{stack_hint}

A backend service is being created. The following files are already in the plan:
{planned_files}

Your task: identify any REQUIRED configuration or tooling files that are MISSING.

Only include files that meet ALL of these criteria:
  1. The project CANNOT compile, run, or be tested without them
  2. They are NOT already listed above (exact path or functional equivalent)
  3. They are standard/idiomatic for this language and framework

Common files that are often forgotten:
  - TypeScript:    tsconfig.json (compiler will not run without it)
  - Node.js/TS:    jest.config.js or jest.config.ts (tests cannot run without it)
  - Node.js/TS:    .eslintrc.json or eslint.config.js
  - Python:        pyproject.toml or setup.cfg (pytest, black, mypy config)
  - Java/Maven:    pom.xml (build will not run without it)
  - Java/Gradle:   build.gradle or build.gradle.kts
  - Go:            go.mod (module resolution will fail without it)
  - Rust:          Cargo.toml
  - All languages: .env.example (documents required environment variables)
  - All languages: .gitignore (prevents secrets and build artefacts being committed)

Return a JSON ARRAY. Each object must have:
  - "file":        relative path (e.g. "tsconfig.json")
  - "description": one sentence — what this config does and why it is required

Return [] if nothing is missing. Do NOT re-list files already in the plan.
Do NOT include optional or nice-to-have files.
"""
        try:
            raw_audit, _ = self._call_llm_structured(
                state, audit_prompt, schema_hint=self._AUDIT_SCHEMA
            )
            if isinstance(raw_audit, dict):
                missing_list = next(
                    (v for v in raw_audit.values() if isinstance(v, list)), []
                )
            else:
                missing_list = raw_audit if isinstance(raw_audit, list) else []

        except Exception as exc:
            print(f"[Architect] Config audit failed (non-fatal): {exc}")
            return plan

        if not missing_list:
            print("[Architect] Config audit: plan is complete — no missing config files.")
            return plan

        injected: list[PlanItem] = []
        for entry in missing_list:
            if not isinstance(entry, dict) or "file" not in entry:
                continue
            file_path = entry["file"].strip()
            description = entry.get("description", "Required configuration file")
            print(f"[Architect] Injecting missing config: {file_path} — {description}")
            injected.append(PlanItem(
                file=file_path,
                action="CREATE",
                description=description,
                api_contract="",
                scope_estimate="5-40",
            ))

        if injected:
            print(f"[Architect] Config audit added {len(injected)} missing file(s) to plan.")
        return plan + injected