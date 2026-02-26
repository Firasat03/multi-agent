"""tools/rules_loader.py — unchanged from original"""
from __future__ import annotations
from pathlib import Path
from config import DEFAULT_RULES_FILE

def load_rules(rules_file: str | Path | None = None) -> str:
    path = Path(rules_file) if rules_file else DEFAULT_RULES_FILE
    if not path.exists():
        print(f"[Rules] No rules file at {path}. Running without custom rules.")
        return ""
    content = path.read_text(encoding="utf-8").strip()
    print(f"[Rules] Loaded: {path} ({len(content)} chars)")
    return content

def validate_rules(rules_content: str) -> list[str]:
    warnings = []
    if not rules_content:
        warnings.append("Rules content is empty.")
    if len(rules_content) > 8000:
        warnings.append(f"Rules content very long ({len(rules_content)} chars).")
    return warnings

def build_rules_block(user_rules: str) -> str:
    if not user_rules:
        return ""
    return (
        "\n\n[USER CODING RULES — YOU MUST FOLLOW THESE IN ALL OUTPUT]\n"
        "═══════════════════════════════════════════════════════════\n"
        + user_rules
        + "\n═══════════════════════════════════════════════════════════\n"
    )
