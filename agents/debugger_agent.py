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
AFFECTED FILES: src/auth.py, src/utils.py

ANALYSIS:
The code imports nothing from utils module (line 1).
On line 42, it calls validate_token() which doesn't exist in the current scope.
This causes a NameError at runtime.

FIX INSTRUCTIONS:

---FILE: src/auth.py---
Line 1: Add this import at the top:
from utils import validate_token

---FILE: src/utils.py---
Line 10: Remove the old implementation and replace with:
def validate_token(token: str) -> bool:
    # implementation...

CONFIDENCE: 5

CRITICAL RULES:
1. Start with "ERROR CATEGORY:" (STATIC or RUNTIME)
2. Follow with "ROOT CAUSE:" (1-2 sentences, be specific)
3. List "AFFECTED FILES:" — MANDATORY — comma-separated on ONE LINE (e.g., "file1.py, file2.py, file3.py")
   - MUST identify at least one file, or write "AFFECTED FILES: File identification failed"
   - Do NOT skip this line under any circumstances
4. Then "ANALYSIS:" (2-3 paragraphs explaining the root cause)
5. Then "FIX INSTRUCTIONS:" (empty line after this, then structured fixes below)
6. Each file fix preceded by "---FILE: <path>---" (exactly this format, three dashes)
7. If infrastructure files need changes (pom.xml, package.json, requirements.txt, etc.), include them
8. End with "CONFIDENCE: <1-5>" (single digit score only)
9. The Coder searches for 'AFFECTED FILES:', 'FIX INSTRUCTIONS:', and '---FILE:' markers — all are critical

IMPORTANT: Even if you cannot be certain about which files to fix, you MUST attempt to list them
in the AFFECTED FILES line. The parser will extract this regardless of format. Examples:
  ✓ AFFECTED FILES: pom.xml, src/main/java/com/example/auth/service/AuthService.java
  ✓ AFFECTED FILES: looks like SecurityConfig.java, maybe also JwtTokenProvider.java
  ✗ (skipping AFFECTED FILES line entirely — will fail parsing)

FORMAT CHECK before responding:
  ☐ Starts with "ERROR CATEGORY:"
  ☐ Has "ROOT CAUSE:" 
  ☐ Has "AFFECTED FILES:" on ONE line, comma-separated (no line breaks within list)
  ☐ Has "ANALYSIS:" section
  ☐ Has "FIX INSTRUCTIONS:" 
  ☐ Each file fix starts with "---FILE: <path>---" (three dashes, exact format)
  ☐ Ends with "CONFIDENCE: <digit>" (not "CONFIDENCE: <digit>/5" or other variations)
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
        
        # Strategy 1: Extract from "AFFECTED FILES: ..." line
        for line in response_text.split('\n'):
            if 'AFFECTED FILES:' in line:
                # Split on comma and extract all file paths
                parts = line.split('AFFECTED FILES:')[1]
                for item in parts.split(','):
                    item = item.strip()
                    if item and len(item) > 2 and item.lower() != 'none':  # exclude empty or single-char items
                        affected.append(item)
        
        # Strategy 2: Extract from "---FILE: <path> ---" format lines
        for line in response_text.split('\n'):
            if '---FILE:' in line and '---' in line:
                # Extract path from: ---FILE: src/auth.py---
                try:
                    file_part = line.split('---FILE:')[1].split('---')[0].strip()
                    if file_part and len(file_part) > 2:
                        affected.append(file_part)
                except (IndexError, ValueError):
                    pass
        
        # Strategy 3: Extract file paths from the ANALYSIS section (fallback)
        # Look for common patterns like "src/auth/...", "tests/...", etc.
        if not affected:
            import re as regex_mod
            # Match common Java/Python file paths in the response
            file_patterns = regex_mod.findall(
                r'(?:src|tests)/[a-zA-Z0-9/_\-\.]+\.(?:java|py|ts|go)',
                response_text,
                regex_mod.IGNORECASE
            )
            affected.extend([f for f in file_patterns if len(f) > 2])
        
        # Deduplicate and format
        affected_str = ", ".join(sorted(set(affected))) if affected else "unknown"

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
    """
    Format files for the debugger context window, with smart truncation.
    
    Strategy:
    1. Always include small files (< 1KB) — likely where bugs are
    2. Truncate large files first (preserve imports/decls, remove middle/end)
    3. Stop adding files once budget exhausted
    4. Show summary of what was truncated
    """
    parts = []
    total_chars = 0
    truncated_files = []
    
    # Sort: small files first (high signal), then by size
    sorted_files = sorted(files.items(), key=lambda x: (len(x[1]), x[0]))
    
    for path, content in sorted_files:
        content_len = len(content)
        
        # Small files — always include
        if content_len < 1_000:
            entry = f"### {path}\n```\n{content}\n```"
            if total_chars + len(entry) <= _MAX_FILES_CHARS:
                parts.append(entry)
                total_chars += len(entry)
            else:
                truncated_files.append(path)
            continue
        
        # Large files — truncate imports + first 500 chars + last 300 chars
        if content_len > 1_000:
            lines = content.split('\n')
            
            # Keep imports/declarations (first ~10 lines)
            import_lines = []
            code_lines = []
            for i, line in enumerate(lines[:15]):
                if any(kw in line for kw in ('import', 'from ', 'require', 'def ', 'class ', 'package')):
                    import_lines.append(line)
                else:
                    code_lines.append(line)
            
            # Construct truncated version
            truncated_content = '\n'.join(import_lines + code_lines[:20])
            truncated_content += f"\n\n... ({content_len - len(truncated_content)} chars omitted) ...\n\n"
            truncated_content += '\n'.join(lines[-10:])  # Last 10 lines often have error hints
            
            entry = f"### {path} (truncated from {content_len} chars)\n```\n{truncated_content}\n```"
            if total_chars + len(entry) <= _MAX_FILES_CHARS:
                parts.append(entry)
                total_chars += len(entry)
            else:
                truncated_files.append(path)
    
    result = "\n\n".join(parts)
    
    if truncated_files:
        result += f"\n\n### Files not included (context limit reached):\n"
        result += "\n".join(f"- {f}" for f in truncated_files[:5])
        if len(truncated_files) > 5:
            result += f"\n- ... and {len(truncated_files) - 5} more files"
    
    return result
