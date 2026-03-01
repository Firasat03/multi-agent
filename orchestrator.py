"""
orchestrator.py — Central Pipeline Controller

Changes vs original:
  - Global _step counter replaced by run-scoped _RunContext (concurrency-safe)
  - Every agent.run() wrapped with a configurable timeout (AGENT_TIMEOUT_SECS)
  - Cost budget enforcement: MAX_RUN_COST_USD halts the pipeline if exceeded
  - SecurityAgent stage added between Reviewer and Tester
  - Security findings with fix_instructions route back through Coder
  - state.apply(output) pattern throughout — Orchestrator owns all state writes
  - Dead-letter: final failure state written to .workflow/<run-id>/FAILURE.json
  - Debugger escalation (output.escalate) handled here, not inside the agent
  - review_verdict read from state.review_verdict (not re-parsed from review_notes)
  - Skip agent functionality: Optional agents can be disabled via config

Agent execution order:
  Architect → [Human Gate] → Coder → Reviewer → Security (skip: SKIP_SECURITY)
  → Tester → Debugger (loop, skip: SKIP_DEBUGGER)
  → Integration (loop, skip: SKIP_INTEGRATION)
  → Writer (skip: SKIP_WRITER)
  → DevOps (skip: SKIP_DEVOPS)

MANDATORY agents (CANNOT be skipped):
  - Architect: Creates the implementation plan
  - Coder: Generates code from the plan
  - Reviewer: Reviews code for quality and correctness

OPTIONAL agents that can be disabled via config or env vars:
  - Security: Security scanning (env: SKIP_SECURITY=true)
  - Tester: Unit and integration tests (env: SKIP_TESTER=true)
  - Debugger: Fixes failed tests (env: SKIP_DEBUGGER=true)
  - Integration: Integration tests (env: SKIP_INTEGRATION=true)
  - Writer: Documentation generation (env: SKIP_WRITER=true)
  - DevOps: Docker & deployment setup (env: SKIP_DEVOPS=true)
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    AGENT_TIMEOUT_SECS,
    MAX_DEBUG_RETRIES,
    MAX_RUN_COST_USD,
    MAX_REVIEW_RETRIES,
    SKIP_DEBUGGER,
    SKIP_DEVOPS,
    SKIP_INTEGRATION,
    SKIP_SECURITY,
    SKIP_TESTER,
    SKIP_WRITER,
    Status,
)
from state import PipelineState
from tools.checkpoint_tools import save_checkpoint
from tools.rules_loader import load_rules

from agents.architect_agent import ArchitectAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from agents.security_agent import SecurityAgent
from agents.tester_agent import TesterAgent
from agents.debugger_agent import DebuggerAgent
from agents.integration_agent import IntegrationAgent
from agents.writer_agent import WriterAgent
from agents.devops_agent import DevOpsAgent

console = Console()


# ─── Run-scoped context (replaces module-level _step global) ─────────────────

class _RunContext:
    """Encapsulates per-run mutable state for the Orchestrator."""

    def __init__(self, state: PipelineState) -> None:
        self._step = 0
        self._run_id = state.run_id

    def checkpoint(self, state: PipelineState, label: str) -> None:
        self._step += 1
        path = save_checkpoint(state, label, self._step)
        console.log(f"[dim]💾 Checkpoint: {path}[/dim]")

    def run_agent(
        self,
        agent,
        state: PipelineState,
        label: str,
        timeout: int = AGENT_TIMEOUT_SECS,
    ) -> PipelineState:
        """
        Run agent.run(state) with a hard timeout.
        Saves a checkpoint after completion.
        Raises TimeoutError if the agent exceeds the timeout.
        """
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(agent.run, state)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                elapsed = int(time.time() - start)
                raise TimeoutError(
                    f"Agent {agent.name!r} exceeded timeout of {timeout}s "
                    f"(ran for {elapsed}s)"
                )
        self.checkpoint(result, label)
        return result

    def check_cost_budget(self, state: PipelineState) -> None:
        """Halt the pipeline if estimated cost exceeds MAX_RUN_COST_USD."""
        if MAX_RUN_COST_USD > 0 and state.estimated_cost_usd >= MAX_RUN_COST_USD:
            raise RuntimeError(
                f"Cost budget exceeded: ${state.estimated_cost_usd:.4f} >= "
                f"${MAX_RUN_COST_USD:.2f} (MAX_RUN_COST_USD). "
                "Increase MAX_RUN_COST_USD or reduce the number of retries."
            )


# ─── Human Plan Approval Gate ─────────────────────────────────────────────────

def _human_plan_approval(state: PipelineState) -> PipelineState:
    if state.task_checklist:
        console.print(Panel(
            f"[bold cyan]📋 IMPLEMENTATION TASK CHECKLIST[/bold cyan]\n\n{state.task_checklist}",
            title=f"[bold]Architect's Task Breakdown — {state.run_id}[/bold]",
            border_style="cyan",
        ))

    console.print(Panel(
        f"[bold yellow]🗺️  ARCHITECT'S PLAN SUMMARY[/bold yellow]\n\n{state.plan_summary}",
        title="[bold]Plan Summary[/bold]",
        border_style="yellow",
    ))

    table = Table(title="Files to be created / modified", show_lines=True, min_width=80)
    table.add_column("Action",       style="cyan",  width=8)
    table.add_column("File",         style="white")
    table.add_column("API Contract", style="green")
    table.add_column("Scope",        style="dim",   width=10)
    table.add_column("Description",  style="dim")
    for item in state.plan:
        table.add_row(
            item.action, item.file,
            item.api_contract or "—",
            item.scope_estimate or "—",
            item.description[:80] + ("..." if len(item.description) > 80 else ""),
        )
    console.print(table)

    while True:
        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  [green][A][/green] Approve plan and start coding")
        console.print("  [yellow][C][/yellow] Request changes to the plan")
        console.print("  [red][X][/red]   Abort workflow")
        choice = input("\nYour choice (A/C/X): ").strip().upper()

        if choice == "A":
            state.plan_approved = True
            state.user_feedback = None
            state.status = Status.PLAN_REVIEW
            console.print("[green]✅ Plan approved![/green]")
            return state
        elif choice == "C":
            console.print("\nDescribe the changes (press Enter twice to submit):")
            lines = []
            while True:
                line = input("> ")
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            state.user_feedback = "\n".join(lines).strip()
            state.plan_approved = False
            state.status = Status.ARCHITECT
            console.print("[yellow]✏️  Feedback recorded. Re-running Architect...[/yellow]")
            return state
        elif choice == "X":
            console.print("[red]🛑 Workflow aborted by user.[/red]")
            state.status = Status.ABORTED
            return state
        else:
            console.print("[red]❌ Invalid choice.[/red]")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run(
    task_prompt: str,
    project_root: str,
    rules_file: str | None = None,
    existing_state: PipelineState | None = None,
    devops_mode: str | None = None,
    language: str = "auto",
) -> PipelineState:

    # ── Init state ────────────────────────────────────────────────────────
    if existing_state:
        state = existing_state
        if devops_mode and not state.devops_mode:
            state.devops_mode = devops_mode
        if language and language != "auto" and state.language == "auto":
            state.language = language
        console.print(f"[cyan]▶ Resuming {state.run_id} from {state.status}[/cyan]")
    else:
        state = PipelineState(
            task_prompt=task_prompt,
            project_root=project_root,
            devops_mode=devops_mode,
            language=language,
        )
        state.user_rules = load_rules(rules_file)
        state.active_rules_file = str(rules_file or "rules/RULES.md")

    ctx = _RunContext(state)

    console.print(Panel(
        f"[bold cyan]🚀 Multi-Agent BE Workflow[/bold cyan]\n"
        f"Run ID:   [bold]{state.run_id}[/bold]\n"
        f"Task:     {task_prompt[:120]}\n"
        f"Language: {state.language}\n"
        f"Rules:    {state.active_rules_file}\n"
        f"DevOps:   {state.devops_mode or 'disabled'}\n"
        f"Budget:   {'$' + str(MAX_RUN_COST_USD) if MAX_RUN_COST_USD > 0 else 'unlimited'}",
        border_style="cyan",
    ))

    # ── Instantiate agents ────────────────────────────────────────────────
    architect  = ArchitectAgent()
    coder      = CoderAgent()
    reviewer   = ReviewerAgent()
    security   = SecurityAgent()
    tester     = TesterAgent()
    debugger   = DebuggerAgent()
    integrator = IntegrationAgent()
    writer     = WriterAgent()
    devops     = DevOpsAgent()

    try:
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1 — Architect → Human Approval Gate
        # ═══════════════════════════════════════════════════════════════════
        if state.status in (Status.INIT, Status.ARCHITECT, Status.PLAN_REVIEW):
            while True:
                console.rule("[bold blue]🏛️  Stage 1 — Architect[/bold blue]")
                state = ctx.run_agent(architect, state, "architect")
                ctx.check_cost_budget(state)

                # Cache detected language so downstream agents don't re-detect
                if state.language == "auto" and state.generated_files:
                    from tools.shell_tools import detect_language
                    detected = detect_language(state.generated_files)
                    if detected != "unknown":
                        state.language = detected
                        console.log(f"[dim]🔍 Language detected and cached: {detected}[/dim]")

                state = _human_plan_approval(state)
                ctx.checkpoint(state, "plan_review")

                if state.status == Status.ABORTED:
                    return _finalize(state, ctx)
                if state.plan_approved:
                    break

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2 — Coder → Reviewer (with retry)
        # ═══════════════════════════════════════════════════════════════════
        if state.status in (Status.PLAN_REVIEW, Status.CODING, Status.REVIEWING):
            review_attempts = state.review_retry_count

            while True:
                console.rule("[bold green]💻 Stage 2a — Coder[/bold green]")
                state = ctx.run_agent(coder, state, "coder")
                ctx.check_cost_budget(state)

                console.rule("[bold magenta]🔍 Stage 2b — Reviewer[/bold magenta]")
                state = ctx.run_agent(reviewer, state, "reviewer")
                ctx.check_cost_budget(state)

                if state.review_verdict == "PASS":
                    console.print("[green]✅ Code review passed![/green]")
                    state.files_with_issues = set()  # Clear files_with_issues since code passed review
                    break
                elif review_attempts >= MAX_REVIEW_RETRIES:
                    console.print(
                        f"[yellow]⚠ Reviewer rejected {review_attempts + 1}x — "
                        "proceeding to security scan.[/yellow]"
                    )
                    break
                else:
                    review_attempts += 1
                    state.review_retry_count = review_attempts
                    console.print(
                        f"[yellow]🔄 Reviewer REJECT #{review_attempts} — "
                        "sending back to Coder...[/yellow]"
                    )
                    # Build fix instructions with explicit file list
                    files_list = ", ".join(sorted(state.files_with_issues)) if state.files_with_issues else "All files"
                    state.fix_instructions = (
                        f"The code reviewer rejected the code.\n\n"
                        f"FILES NEEDING FIXES: {files_list}\n\n"
                        f"REVIEWER'S DETAILED FEEDBACK:\n"
                        + (state.review_notes or "")
                    )

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 3 — Security scan (SAST + dep vulnerabilities)
        # Routes back to Coder if HIGH findings exist
        # ═══════════════════════════════════════════════════════════════════
        if SKIP_SECURITY:
            console.print("[yellow]⏭️  Skipping Stage 3 — Security Scanner (disabled)[/yellow]")
        elif state.status not in (Status.TESTING, Status.DEBUGGING,
                                Status.INTEGRATION, Status.WRITING,
                                Status.DEVOPS, Status.DONE,
                                Status.FAILED, Status.ABORTED):
            security_attempts = 0
            MAX_SECURITY_RETRIES = 2

            while True:
                console.rule("[bold red]🔒 Stage 3 — Security Scanner[/bold red]")
                state = ctx.run_agent(security, state, f"security_{security_attempts + 1}")
                ctx.check_cost_budget(state)

                # SecurityAgent sets fix_instructions when HIGH findings block pipeline
                if state.fix_instructions and "SECURITY SCAN FAILED" in (state.fix_instructions or ""):
                    if security_attempts >= MAX_SECURITY_RETRIES:
                        console.print(
                            f"[red]Security scan still failing after {MAX_SECURITY_RETRIES} "
                            "fix attempts. Escalating to human.[/red]"
                        )
                        state.status = Status.FAILED
                        return _finalize(state, ctx)

                    console.print(
                        f"[yellow]🔄 Security fix #{security_attempts + 1} — "
                        "sending to Coder...[/yellow]"
                    )
                    # Include files_with_issues in fix instructions for Coder focus
                    files_list = ", ".join(sorted(state.files_with_issues)) if state.files_with_issues else "All files"
                    if state.fix_instructions and "FILES NEEDING FIXES:" not in state.fix_instructions:
                        state.fix_instructions = f"FILES NEEDING FIXES: {files_list}\n\n" + state.fix_instructions
                    state = ctx.run_agent(coder, state, f"coder_security_fix_{security_attempts + 1}")
                    ctx.check_cost_budget(state)
                    security_attempts += 1
                else:
                    console.print("[green]✅ Security scan passed![/green]")
                    state.files_with_issues = set()  # Clear files_with_issues since security passed
                    break

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 4 — Tester (unit tests) → Debugger loop
        # ═══════════════════════════════════════════════════════════════════
        if SKIP_TESTER:
            console.print("[yellow]⏭️  Skipping Stage 4 — Tester (disabled)[/yellow]")
        elif state.status in (Status.REVIEWING, Status.CODING, Status.SECURITY,
                            Status.TESTING, Status.DEBUGGING):
            attempts = state.retry_count

            while True:
                # ── 4a: Unit / static tests ───────────────────────────────
                console.rule(
                    f"[bold blue]🧪 Unit Tests — attempt {attempts + 1}/{MAX_DEBUG_RETRIES + 1}[/bold blue]"
                )
                state = ctx.run_agent(tester, state, f"tester_{attempts + 1}")
                ctx.check_cost_budget(state)

                if not state.test_passed():
                    if attempts >= MAX_DEBUG_RETRIES:
                        console.print(
                            f"[red]Max debug retries ({MAX_DEBUG_RETRIES}) reached — "
                            "escalating to human.[/red]"
                        )
                        state.status = Status.FAILED
                        state.record_failure(
                            stage="TESTING",
                            agent="Orchestrator",
                            error_summary=f"Max debug retries exceeded after {attempts} cycles",
                            error_detail=state.error_log or "",
                        )
                        return _finalize(state, ctx)

                    console.print("[red]Unit tests failed — invoking Debugger...[/red]")
                    
                    if SKIP_DEBUGGER:
                        console.print("[yellow]⏭️  Debugger disabled — escalating to human instead[/yellow]")
                        state.status = Status.FAILED
                        state.record_failure(
                            stage="TESTING",
                            agent="Orchestrator",
                            error_summary="Unit tests failed and Debugger is disabled",
                            error_detail=state.error_log or "",
                        )
                        return _finalize(state, ctx)
                    
                    console.rule(f"[bold red]🐛 Debugger — cycle {attempts + 1}[/bold red]")
                    state = ctx.run_agent(debugger, state, f"debugger_unit_{attempts + 1}")
                    ctx.check_cost_budget(state)

                    # Check for low-confidence escalation
                    if _debugger_escalated(state):
                        state.status = Status.FAILED
                        return _finalize(state, ctx)

                    # Include files_with_issues in fix instructions for Coder focus
                    files_list = ", ".join(sorted(state.files_with_issues)) if state.files_with_issues else "All files"
                    if state.fix_instructions and "FILES NEEDING FIXES:" not in state.fix_instructions:
                        state.fix_instructions = f"FILES NEEDING FIXES: {files_list}\n\n" + state.fix_instructions
                    console.print("[yellow]Applying fix via Coder...[/yellow]")
                    state = ctx.run_agent(coder, state, f"coder_fix_unit_{attempts + 1}")
                    ctx.check_cost_budget(state)
                    attempts = state.retry_count
                    continue

                console.print("[green]✅ Unit / static tests passed.[/green]")
                state.files_with_issues = set()  # Clear files_with_issues since tests passed
                break

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 4b — Integration Tests → Debugger loop (independent of Tester)
        # ═══════════════════════════════════════════════════════════════════
        if SKIP_INTEGRATION:
            console.print("[yellow]⏭️  Skipping Stage 4b — Integration Tests (disabled)[/yellow]")
        elif state.status not in (Status.FAILED, Status.ABORTED):
            # Integration stage can run after unit tests pass, or independently if tester is skipped
            if state.status in (Status.TESTING, Status.DEBUGGING, Status.INTEGRATION, 
                              Status.REVIEWING, Status.CODING, Status.SECURITY):
                attempts = state.retry_count

                while True:
                    # ── 4b: Integration tests ─────────────────────────────────
                    console.rule("[bold cyan]🔗 Integration Tests[/bold cyan]")
                    state = ctx.run_agent(integrator, state, f"integration_{attempts + 1}")
                    ctx.check_cost_budget(state)

                    if state.integration_passed:
                        console.print("[green bold]✅ Integration tests passed.[/green bold]")
                        state.files_with_issues = set()  # Clear files_with_issues since integration tests passed
                        break

                    if attempts >= MAX_DEBUG_RETRIES:
                        console.print(
                            f"[red]Max retries reached after integration failures. "
                            "Escalating to human.[/red]"
                        )
                        state.status = Status.FAILED
                        state.record_failure(
                            stage="INTEGRATION",
                            agent="Orchestrator",
                            error_summary=f"Integration tests failed after {attempts} retries",
                            error_detail=state.integration_test_output or "",
                        )
                        return _finalize(state, ctx)

                    console.print("[red]Integration tests failed — invoking Debugger...[/red]")
                    
                    if SKIP_DEBUGGER:
                        console.print("[yellow]⏭️  Debugger disabled — escalating to human instead[/yellow]")
                        state.status = Status.FAILED
                        state.record_failure(
                            stage="INTEGRATION",
                            agent="Orchestrator",
                            error_summary="Integration tests failed and Debugger is disabled",
                            error_detail=state.integration_test_output or "",
                        )
                        return _finalize(state, ctx)
                    
                    console.rule(f"[bold red]🐛 Debugger — integration cycle {attempts + 1}[/bold red]")
                    state = ctx.run_agent(debugger, state, f"debugger_integration_{attempts + 1}")
                    ctx.check_cost_budget(state)

                    if _debugger_escalated(state):
                        state.status = Status.FAILED
                        return _finalize(state, ctx)

                    # Include files_with_issues in fix instructions for Coder focus
                    files_list = ", ".join(sorted(state.files_with_issues)) if state.files_with_issues else "All files"
                    if state.fix_instructions and "FILES NEEDING FIXES:" not in state.fix_instructions:
                        state.fix_instructions = f"FILES NEEDING FIXES: {files_list}\n\n" + state.fix_instructions
                    console.print("[yellow]Applying integration fix via Coder...[/yellow]")
                    state = ctx.run_agent(coder, state, f"coder_fix_integration_{attempts + 1}")
                    ctx.check_cost_budget(state)
                    attempts = state.retry_count
                    state.integration_passed = None  # reset so integration re-runs

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 5 — Writer
        # ═══════════════════════════════════════════════════════════════════
        if SKIP_WRITER:
            console.print("[yellow]⏭️  Skipping Stage 5 — Writer (disabled)[/yellow]")
        elif state.status not in (Status.WRITING, Status.DEVOPS, Status.DONE,
                                Status.FAILED, Status.ABORTED):
            console.rule("[bold]📝 Stage 5 — Writer[/bold]")
            state = ctx.run_agent(writer, state, "writer")
            ctx.check_cost_budget(state)

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 6 — DevOps (opt-in)
        # ═══════════════════════════════════════════════════════════════════
        if SKIP_DEVOPS or not state.devops_mode:
            if SKIP_DEVOPS:
                console.print("[yellow]⏭️  Skipping Stage 6 — DevOps (disabled)[/yellow]")
        elif state.status not in (Status.DONE, Status.FAILED, Status.ABORTED):
            console.rule(
                f"[bold cyan]🐳 Stage 6 — DevOps (mode: {state.devops_mode})[/bold cyan]"
            )
            state = ctx.run_agent(devops, state, "devops")

    except TimeoutError as exc:
        console.print(f"[red]⏱ Agent timeout: {exc}[/red]")
        state.status = Status.FAILED
        state.record_failure(
            stage=state.status,
            agent="Orchestrator",
            error_summary=str(exc),
        )
    except RuntimeError as exc:
        console.print(f"[red]🔥 Pipeline error: {exc}[/red]")
        state.status = Status.FAILED
        state.record_failure(
            stage=state.status,
            agent="Orchestrator",
            error_summary=str(exc),
        )

    # ── Mark done ─────────────────────────────────────────────────────────
    if state.status not in (Status.FAILED, Status.ABORTED):
        state.status = Status.DONE

    return _finalize(state, ctx)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _debugger_escalated(state: PipelineState) -> bool:
    """
    Return True if the Debugger flagged low confidence and wants human escalation.
    The Debugger signals this by leaving fix_instructions=None.
    """
    if state.fix_instructions is None and state.status == Status.DEBUGGING:
        console.print(
            "[red]🚨 Debugger escalating to human — low confidence fix.[/red]"
        )
        return True
    return False


def _finalize(state: PipelineState, ctx: _RunContext) -> PipelineState:
    """Save final checkpoint, write failure report if needed, print summary."""
    ctx.checkpoint(state, "final")

    if state.pipeline_errors:
        _write_failure_report(state)

    _print_summary(state)
    return state


def _write_failure_report(state: PipelineState) -> None:
    """Write a structured JSON failure report for post-mortem analysis."""
    from config import WORKFLOW_DIR
    report_path = WORKFLOW_DIR / state.run_id / "FAILURE.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id":         state.run_id,
        "task_prompt":    state.task_prompt,
        "final_status":   state.status,
        "total_tokens":   state.total_tokens_used,
        "estimated_cost": state.estimated_cost_usd,
        "pipeline_errors": [e.model_dump() for e in state.pipeline_errors],
        "audit_trail":    [a.model_dump() for a in state.audit_trail],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]📋 Failure report: {report_path}[/dim]")


def _print_summary(state: PipelineState) -> None:
    colour = {"DONE": "green", "FAILED": "red", "ABORTED": "yellow"}.get(
        state.status, "white"
    )

    devops_info = f"DevOps files: {len(state.devops_files)}\n" if state.devops_mode else ""
    cost_info = (
        f"Est. cost:    ${state.estimated_cost_usd:.4f} USD\n"
        f"Total tokens: {state.total_tokens_used:,}\n"
    )

    console.print(Panel(
        f"[bold {colour}]Pipeline {state.status}[/bold {colour}]\n\n"
        f"Run ID:       {state.run_id}\n"
        f"Task:         {state.task_prompt[:80]}\n"
        f"Files made:   {len(state.generated_files)}\n"
        f"{devops_info}"
        f"{cost_info}"
        f"Debug cycles: {state.retry_count}\n"
        f"Review retries: {state.review_retry_count}\n"
        f"Agents ran:   {len(state.audit_trail)}",
        title="[bold]Workflow Summary[/bold]",
        border_style=colour,
    ))

    table = Table(title="Agent Audit Trail", show_lines=True)
    table.add_column("#",        style="dim",   width=3)
    table.add_column("Agent",    style="cyan")
    table.add_column("Status",   style="white")
    table.add_column("Tokens",   style="green")
    table.add_column("Cost",     style="yellow")
    table.add_column("Duration", style="dim")
    table.add_column("Notes",    style="dim")
    for i, entry in enumerate(state.audit_trail, 1):
        table.add_row(
            str(i), entry.agent, entry.status,
            str(entry.tokens_used),
            f"${entry.estimated_cost_usd:.4f}",
            f"{entry.duration_ms}ms",
            entry.notes[:60],
        )
    console.print(table)

    if state.devops_files:
        dt = Table(title="DevOps Files", show_lines=True)
        dt.add_column("File", style="cyan")
        dt.add_column("Size", style="dim")
        for path, content in state.devops_files.items():
            dt.add_row(path, f"{len(content)} chars")
        console.print(dt)

    if state.pipeline_errors:
        et = Table(title="[red]Pipeline Errors[/red]", show_lines=True)
        et.add_column("Stage",   style="red")
        et.add_column("Agent",   style="cyan")
        et.add_column("Summary", style="white")
        for err in state.pipeline_errors:
            et.add_row(err.stage, err.agent, err.error_summary)
        console.print(et)