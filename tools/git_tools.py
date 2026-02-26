"""tools/git_tools.py — unchanged from original"""
from __future__ import annotations
from tools.shell_tools import run_command

def git_diff(cwd: str) -> str:
    result = run_command("git diff HEAD", cwd=cwd)
    return result["stdout"] or "(no diff)"

def git_stage_all(cwd: str) -> dict:
    return run_command("git add -A", cwd=cwd)

def git_commit(cwd: str, message: str) -> dict:
    return run_command(["git", "commit", "-m", message], cwd=cwd)

def git_status(cwd: str) -> str:
    result = run_command("git status --short", cwd=cwd)
    return result["stdout"] or "(clean)"

def git_current_branch(cwd: str) -> str:
    result = run_command("git rev-parse --abbrev-ref HEAD", cwd=cwd)
    return result["stdout"].strip()

def is_git_repo(cwd: str) -> bool:
    result = run_command("git rev-parse --is-inside-work-tree", cwd=cwd)
    return result["returncode"] == 0
