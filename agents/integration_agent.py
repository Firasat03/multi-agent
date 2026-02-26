"""
agents/integration_agent.py — Live Integration Testing Agent

Changes vs original:
  - Uses a random free port instead of hardcoded 8080 (no port conflict failures)
  - Captures server stdout/stderr on health-check timeout so the Debugger
    gets the real startup error rather than a generic timeout message
  - Response body schema assertion: checks that POST/GET responses contain
    the fields declared in the plan's api_contract
  - Returns typed IntegrationOutput; Orchestrator calls state.apply(output)
  - Server teardown is more robust (SIGTERM → wait → SIGKILL fallback)
  - Fixed: class now inherits BaseAgent directly (no monkey-patching)
"""

from __future__ import annotations

import socket
import time
from rich.console import Console
from rich.table import Table

from agents.base_agent import BaseAgent
from config import Status
from state import IntegrationOutput, PipelineState
from tools.integration_tools import run_integration_tests

console = Console()


def _find_free_port() -> int:
    """Find a random available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class IntegrationAgent(BaseAgent):
    name = "Integration"
    system_role = ""  # IntegrationAgent delegates to integration_tools; no LLM calls

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.INTEGRATION
        console.rule("[bold cyan]🔗 Integration Tests — Build → Start → Curl[/bold cyan]")

        lang = (state.language or "auto").lower().strip()
        if lang in ("auto", "unknown"):
            from tools.shell_tools import detect_language
            lang = detect_language(state.generated_files)

        port = _find_free_port()
        console.print(f"[dim]Using port {port} for integration server[/dim]")

        t0 = time.time()
        result = run_integration_tests(
            project_root=state.project_root,
            language=lang,
            plan_items=state.plan,
            generated_files=state.generated_files,
            port=port,
        )
        elapsed = int((time.time() - t0) * 1000)

        if result.get("build_output"):
            console.print(f"[dim]Build:[/dim]\n{result['build_output'][:800]}")

        if result.get("results"):
            self._print_results_table(result["results"])

        if result.get("server_startup_log"):
            console.print(
                "[dim]Server startup log (last 500 chars):[/dim]\n"
                + result["server_startup_log"][-500:]
            )

        if result.get("error"):
            console.print(f"[red]Integration error: {result['error']}[/red]")

        output = IntegrationOutput()

        if result["passed"]:
            console.print("[green bold]✅ All integration tests passed![/green bold]")
            output.integration_test_output = self._format_results(result)
            output.integration_passed = True
            state.apply(output)
            state.log(self.name, notes="All integration tests passed", duration_ms=elapsed)
        else:
            console.print("[red bold]❌ Integration tests FAILED[/red bold]")
            report = self._build_failure_report(result)
            output.integration_test_output = report
            output.integration_passed = False
            output.error_log = f"INTEGRATION TEST FAILURES:\n{report}"
            state.apply(output)
            state.log(self.name, notes="Integration tests failed", duration_ms=elapsed)

        return state

    def _print_results_table(self, results: list[dict]) -> None:
        table = Table(title="Integration Test Results", show_lines=True)
        table.add_column("Method",   style="cyan",  width=7)
        table.add_column("Path",     style="white")
        table.add_column("Expected", style="dim",   width=8)
        table.add_column("Actual",   style="white", width=8)
        table.add_column("Schema",   style="dim",   width=6)
        table.add_column("Status",   style="white", width=6)
        table.add_column("Response (preview)", style="dim")

        for r in results:
            ok = r.get("passed", False)
            status_icon = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            actual_col = (
                f"[green]{r['actual_status']}[/green]" if ok
                else f"[red]{r['actual_status']}[/red]"
            )
            schema_ok = r.get("schema_ok")
            schema_col = (
                "[green]✓[/green]" if schema_ok is True else
                "[red]✗[/red]"    if schema_ok is False else
                "[dim]—[/dim]"
            )
            table.add_row(
                r["method"], r["path"],
                str(r["expected_status"]), actual_col,
                schema_col, status_icon,
                (r.get("response_body") or "")[:80],
            )
        console.print(table)

    def _format_results(self, result: dict) -> str:
        lines = []
        for r in result.get("results", []):
            flag = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"[{flag}] {r['method']} {r['path']} "
                f"→ expected={r['expected_status']} got={r['actual_status']}"
            )
        return "\n".join(lines)

    def _build_failure_report(self, result: dict) -> str:
        lines = []
        if result.get("error"):
            lines.append(f"ERROR: {result['error']}")
        if result.get("build_output"):
            lines.append("BUILD OUTPUT:")
            lines.append(result["build_output"][:2000])
        if result.get("server_startup_log"):
            lines.append("SERVER STARTUP LOG (may show real cause):")
            lines.append(result["server_startup_log"][-1000:])
        failed = [r for r in result.get("results", []) if not r["passed"]]
        if failed:
            lines.append("FAILED ENDPOINTS:")
            for r in failed:
                lines.append(
                    f"  {r['method']} {r['path']} — "
                    f"expected HTTP {r['expected_status']}, got HTTP {r['actual_status']}\n"
                    f"  Response: {(r.get('response_body') or '')[:300]}"
                )
                if r.get("schema_ok") is False:
                    lines.append(
                        f"  Schema check: FAILED — "
                        f"expected fields {r.get('expected_fields', [])} not all present"
                    )
        return "\n".join(lines)