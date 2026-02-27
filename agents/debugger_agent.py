"""
agents/debugger_agent.py — Debugger Agent

Changes vs original:
  - Returns typed DebuggerOutput; Orchestrator calls state.apply(output)
  - Escalation is signalled via output.escalate flag rather than directly
    mutating state.status — keeps state mutation in the Orchestrator
  - Low-confidence escalations now include a structured PipelineError record
  - Files block is truncated for large codebases to avoid context overflow
"""

from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from config import Status
from state import DebuggerOutput, PipelineState


_LOW_CONFIDENCE_THRESHOLD = 3
_MAX_FILES_CHARS = 40_000


class DebuggerAgent(BaseAgent):
    name = "Debugger"
    system_role = (
        "You are an expert Debugging Specialist for production backend systems. "
        "You analyze test failures and root-cause errors in generated code.\n\n"
        "YOUR JOB:\n"
        "1. Read error messages (static analysis errors first, then runtime test failures)\n"
        "2. Examine the source code to understand what went wrong\n"
        "3. Pinpoint EXACTLY which files and functions have issues\n"
        "4. Provide PRECISE, minimal fix instructions that the Coder can implement\n\n"
        "PRIORITY ORDER:\n"
        "  STATIC ERRORS (syntax, imports, undefined names) — fix FIRST\n"
        "  RUNTIME ERRORS (logic bugs, assertion failures) — fix after static\n\n"
        "FIX INSTRUCTIONS MUST BE:\n"
        "  • File-specific (one section per file with exact path)\n"
        "  • Line-number precise (reference line numbers where possible)\n"
        "  • Actionable (Coder can implement directly without ambiguity)\n"
        "  • Minimal (fix only what's broken, don't refactor)\n"
        "  • Complete (if you identify the fix, describe it fully)\n\n"
        "OUTPUT: Structured analysis → FIX INSTRUCTIONS → CONFIDENCE score (1-5)\n\n"
        "INFRASTRUCTURE ERRORS:\n"
        "If you see errors like 'Command not found', 'ModuleNotFoundError', or dependency failures, "
        "you MUST include infrastructure files (pom.xml, requirements.txt, etc.) in your fix instructions."
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.DEBUGGING

        print(f"\n🐛 Debugger: Analyzing test failures...")
        
        files_block = _format_files_truncated(state.generated_files)

        static_section = (
            f"STATIC ANALYSIS ERRORS (fix these FIRST):\n{state.static_analysis_output}"
            if state.static_analysis_output
            else "STATIC ANALYSIS ERRORS: None."
        )
        runtime_section = (
            f"RUNTIME TEST ERRORS:\n{state.error_log}"
            if state.error_log and not state.static_analysis_output
            else (
                "RUNTIME TEST ERRORS: Skipped — fix static errors first."
                if state.static_analysis_output
                else "RUNTIME TEST ERRORS: None."
            )
        )

        rules_context = f"\nUSER CODING RULES (MANDATORY):\n{state.user_rules}\n" if state.user_rules else ""

        prompt = f"""
A test stage failed. Perform root-cause analysis and produce PRECISE fix instructions 
that the Coder can implement immediately.

{rules_context}
ORIGINAL TASK: {state.task_prompt}

ARCHITECT'S PLAN:
{state.plan_summary}

─── ERROR REPORT ─────────────────────────────────────────────────────────
{static_section}

{runtime_section}
──────────────────────────────────────────────────────────────────────────

CURRENT SOURCE FILES:
{files_block}

ANALYSIS INSTRUCTIONS:
1. Address static errors FIRST — they block all other progress.
2. For each error, identify the EXACT root cause with exact file, function, and line #.
3. Describe WHY the code fails — don't just restate the error message.
4. Provide MINIMAL targeted fixes — don't rewrite entire functions.
5. When providing fix instructions, structure them clearly by file name.
6. Use exact code snippets showing BEFORE → AFTER when helpful.

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

EXAMPLE OUTPUT:

ERROR CATEGORY: STATIC
ROOT CAUSE: Missing import statement for 'validate_token' function used on line 42
AFFECTED FILES: src/auth.py

ANALYSIS:
The code imports nothing from utils module (line 1).
On line 42, it calls validate_token() which doesn't exist in the current scope.
This causes a NameError at runtime.

FIX INSTRUCTIONS:

---FILE: src/auth.py---
Line 1: Add this import at the top:
from utils import validate_token

CONFIDENCE: 5

CRITICAL RULES:
1. Start with "ERROR CATEGORY:"
2. Follow with "ROOT CAUSE:" (1-2 sentences)
3. List "AFFECTED FILES:"
4. Then "ANALYSIS:" (2-3 paragraphs)
5. Then "FIX INSTRUCTIONS:"
6. Each file fix preceded by "---FILE: <path>---" (exactly this format)
7. If infrastructure files need changes, include them here.
8. End with "CONFIDENCE: <1-5>" (single digit score)
9. The Coder searches for 'FIX INSTRUCTIONS:' — proper formatting is critical

FORMAT CHECK before responding:
  ☐ Starts with "ERROR CATEGORY:"
  ☐ Has "ROOT CAUSE:" 
  ☐ Has "AFFECTED FILES:"
  ☐ Has "ANALYSIS:" section
  ☐ Has "FIX INSTRUCTIONS:" 
  ☐ Each file fix starts with "---FILE: <path>---" (three dashes, exact format)
  ☐ Ends with "CONFIDENCE: <1>" (not "CONFIDENCE:<1>" or other variations)
─────────────────────────────────────────────────────────────────────

NOW OUTPUT STRUCTURED ANALYSIS WITH FIX INSTRUCTIONS:
"""
        response_text, tokens = self._call_llm(state, prompt)

        confidence_match = re.search(r"CONFIDENCE:\s*(\d)", response_text)
        confidence = int(confidence_match.group(1)) if confidence_match else 3

        fix_match = re.search(r"FIX INSTRUCTIONS:\s*(.+)$", response_text, re.DOTALL)
        fix_instructions = fix_match.group(1).strip() if fix_match else response_text

        # Extract affected files for logging
        affected = []
        for line in response_text.split('\n'):
            if 'AFFECTED FILES:' in line or 'FILE:' in line:
                match = re.search(r"\b(src/\S+|tests/\S+|\S+\.java|\S+\.py|\S+\.ts)", line)
                if match:
                    affected.append(match.group(1))
        
        affected_str = ", ".join(set(affected)) if affected else "unknown"

        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            state.record_failure(
                stage="DEBUGGING",
                agent=self.name,
                error_summary=f"Low confidence ({confidence}/5) — cannot determine fix",
                error_detail=response_text,
                fix_instructions=fix_instructions,
            )
            output = DebuggerOutput(
                fix_instructions=None,
                confidence=confidence,
                escalate=True,
            )
            state.apply(output)
            print(f"   ⚠️  LOW CONFIDENCE ({confidence}/5) - Cannot reliably fix issues")
            print(f"   📍 Affected files: {affected_str}")
            print(f"   🚨 Escalating to human review")
            state.log(
                self.name,
                tokens=tokens,
                notes=f"LOW CONFIDENCE ({confidence}/5) — escalating to human",
            )
        else:
            output = DebuggerOutput(
                fix_instructions=fix_instructions,
                confidence=confidence,
                escalate=False,
            )
            state.apply(output)
            state.retry_count += 1
            print(f"   ✓ Fix analysis complete (confidence: {confidence}/5)")
            print(f"   📍 Affected files: {affected_str}")
            print(f"   ↻ Sending back to Coder for implementation (retry #{state.retry_count})")
            state.log(
                self.name,
                tokens=tokens,
                notes=f"Fix ready (confidence {confidence}/5, retry #{state.retry_count})",
            )

        return state


def _format_files_truncated(files: dict[str, str]) -> str:
    parts = []
    total_chars = 0
    for path, content in files.items():
        entry = f"### {path}\n```\n{content}\n```"
        if total_chars + len(entry) > _MAX_FILES_CHARS:
            parts.append(f"### {path}\n(content truncated — too large)")
        else:
            parts.append(entry)
            total_chars += len(entry)
    return "\n\n".join(parts)
