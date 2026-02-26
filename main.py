"""
main.py — CLI Entry Point for BE Multi-Agent Workflow

Usage:
  python main.py --task "Add POST /login endpoint"
                 --project-root ./my_project
                 [--language python|java|nodejs|go|kotlin|rust|csharp|ruby|php]
                 [--rules rules/spring-boot.md]
                 [--max-retries 3]
                 [--model gemini-2.0-flash]
                 [--devops docker|k8s|all]
                 [--max-cost 2.00]
                 [--no-security]
                 [--resume <run-id>]
                 [--list-runs]
"""
import os
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="be-agent-workflow",
        description="Multi-Agent Backend Code Workflow powered by LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "Add POST /login endpoint" --project-root ./my_api
  python main.py --task "Add POST /register" --project-root ./my_api --language java
  python main.py --task "Add auth service" --project-root ./api --devops docker
  python main.py --task "Add payment service" --max-cost 1.50
  python main.py --resume a3f2-20260223-0452
  python main.py --list-runs
        """,
    )

    parser.add_argument("--task",         type=str, help="Task description")
    parser.add_argument("--project-root", type=str, default=".", help="Target project path")
    parser.add_argument("--language",     type=str, default="auto",
                        choices=["auto", "python", "java", "nodejs", "go",
                                 "kotlin", "rust", "csharp", "ruby", "php"])
    parser.add_argument("--rules",        type=str, default=None)
    parser.add_argument("--max-retries",  type=int, default=None,
                        help="Max debug retry cycles (overrides MAX_DEBUG_RETRIES env)")
    parser.add_argument("--model",        type=str, default=None,
                        help="LLM model name (overrides LLM_MODEL env)")
    parser.add_argument("--max-cost",     type=float, default=None,
                        help="Max USD to spend per run (0 = unlimited). Example: --max-cost 2.00")
    parser.add_argument("--no-security",  action="store_true",
                        help="Skip the security scanning stage (SAST + dep audit)")
    parser.add_argument("--resume",       type=str, default=None)
    parser.add_argument("--list-runs",    action="store_true")
    parser.add_argument(
        "--devops",
        nargs="?", const="all", default=None,
        choices=["docker", "k8s", "all"],
        metavar="MODE",
        help="Enable DevOps agent: docker | k8s | all (default when flag present: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── API key validation ────────────────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    _key_map = {
        "gemini":        ("GEMINI_API_KEY",    "set GEMINI_API_KEY=<key>"),
        "openai":        ("OPENAI_API_KEY",     "set OPENAI_API_KEY=<key>"),
        "openai_compat": ("OPENAI_API_KEY",     "set OPENAI_API_KEY=<key>"),
        "anthropic":     ("ANTHROPIC_API_KEY",  "set ANTHROPIC_API_KEY=<key>"),
        "ollama":        (None, None),
    }
    key_name, key_hint = _key_map.get(provider, ("GEMINI_API_KEY", "set GEMINI_API_KEY=<key>"))
    if key_name and not os.getenv(key_name):
        console.print(f"[red]❌ {key_name} is not set (provider: {provider})[/red]")
        console.print(f"Set it with:  {key_hint}")
        sys.exit(1)

    # ── Apply CLI overrides to config ─────────────────────────────────────
    import config
    if args.max_retries is not None:
        config.MAX_DEBUG_RETRIES = args.max_retries
    if args.model is not None:
        config.LLM_MODEL = args.model
    if args.max_cost is not None:
        config.MAX_RUN_COST_USD = args.max_cost
    if args.no_security:
        config.ENABLE_BANDIT   = False
        config.ENABLE_PIP_AUDIT = False
        os.environ["SECURITY_BLOCK_ON_HIGH"] = "false"

    # ── --list-runs ───────────────────────────────────────────────────────
    if args.list_runs:
        from tools.checkpoint_tools import list_runs
        runs = list_runs()
        if not runs:
            console.print("[yellow]No past runs found.[/yellow]")
            return
        table = Table(title="Past Workflow Runs", show_lines=True)
        table.add_column("Run ID",          style="cyan")
        table.add_column("Status",          style="white")
        table.add_column("Checkpoints",     style="green")
        table.add_column("Last Checkpoint", style="dim")
        table.add_column("Task",            style="dim")
        for r in runs:
            table.add_row(
                r["run_id"], r["status"], str(r["checkpoints"]),
                r["last_checkpoint"], r["task_prompt"],
            )
        console.print(table)
        return

    # ── --resume ──────────────────────────────────────────────────────────
    existing_state = None
    if args.resume:
        from tools.checkpoint_tools import load_latest_checkpoint
        existing_state = load_latest_checkpoint(args.resume)
        if existing_state is None:
            console.print(f"[red]❌ No checkpoint found for run-id: {args.resume}[/red]")
            sys.exit(1)

    if not existing_state and not args.task:
        console.print("[red]❌ --task is required for new runs.[/red]")
        sys.exit(1)

    # ── Info banners ──────────────────────────────────────────────────────
    if args.devops:
        console.print(f"[cyan]🐳 DevOps: [bold]{args.devops}[/bold][/cyan]")
    if args.language != "auto":
        console.print(f"[cyan]>> Language: [bold]{args.language}[/bold][/cyan]")
    if args.no_security:
        console.print("[yellow]⚠ Security scanning disabled (--no-security)[/yellow]")
    if config.MAX_RUN_COST_USD > 0:
        console.print(f"[cyan]💰 Cost budget: ${config.MAX_RUN_COST_USD:.2f}[/cyan]")

    # ── Run pipeline ──────────────────────────────────────────────────────
    from orchestrator import run
    state = run(
        task_prompt=args.task or (existing_state.task_prompt if existing_state else ""),
        project_root=os.path.abspath(args.project_root),
        rules_file=args.rules,
        existing_state=existing_state,
        devops_mode=args.devops,
        language=args.language,
    )

    if state.status in ("FAILED", "ABORTED"):
        sys.exit(1)


if __name__ == "__main__":
    main()
