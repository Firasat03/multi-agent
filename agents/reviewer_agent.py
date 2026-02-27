"""
agents/reviewer_agent.py — Reviewer Agent

Changes vs original:
  - Verdict parsing now defaults to REJECT (fail-safe) instead of PASS
    — a malformed LLM response that omits the verdict line no longer
    silently passes code review
  - Returns typed ReviewerOutput; Orchestrator calls state.apply(output)
  - Verdict is stored separately from review_notes on state for cleaner access
"""

from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from config import Status
from state import PipelineState, ReviewerOutput


class ReviewerAgent(BaseAgent):
    name = "Reviewer"
    system_role = (
        "You are a Senior Code Reviewer and Security Specialist with 15+ years of backend experience. "
        "You enforce production-grade standards. Your review MUST cover:\n\n"
        "1. Correctness — does the code implement the plan accurately?\n"
        "2. API contract alignment — do all endpoints/signatures exactly match the plan?\n"
        "3. Security:\n"
        "   - SQL injection / NoSQL injection risks\n"
        "   - Unvalidated / unsanitised inputs\n"
        "   - Secret / credential leaks in code or logs\n"
        "   - Missing authentication / authorisation guards\n"
        "   - Insecure defaults (debug mode, wide CORS, open endpoints)\n"
        "4. Error handling:\n"
        "   - All exceptions caught and handled gracefully\n"
        "   - No bare 'except' clauses\n"
        "   - HTTP responses always include a meaningful error message\n"
        "5. Observability — structured logging on all significant operations; no PII logged\n"
        "6. Data integrity — correct transaction boundaries; no N+1 query risks\n"
        "7. Code quality — DRY, consistent naming, complex logic documented\n"
        "8. User coding rules (RULES.md) — any violation is an automatic REJECT\n\n"
        "Be strict but constructive. Quote the specific line or function causing the issue.\n\n"
        "CRITICAL — MANDATORY OUTPUT FORMAT:\n"
        "You MUST end your review with EXACTLY this format (one section per line):\n\n"
        "FILES_WITH_ISSUES: <Comma-separated list of relative paths needing fixes, or 'None'>\n"
        "VERDICT: PASS\n\n"
        "OR if there are issues:\n\n"
        "FILES_WITH_ISSUES: src/auth/login.py, src/config.py\n"
        "VERDICT: REJECT\n"
        "REASON: <Clear, specific explanation of the critical issue that must be fixed>\n\n"
        "The machine parser looks for exactly 'VERDICT: PASS' or 'VERDICT: REJECT'.\n"
        "The 'FILES_WITH_ISSUES' line is used to tell the Coder which files to regenerate.\n"
        "If you cannot decide, default to REJECT.\n"
        "DO NOT output any other text after the VERDICT line."
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.REVIEWING

        if not state.generated_files:
            output = ReviewerOutput(review_notes="No files to review.", verdict="PASS")
            state.apply(output)
            state.log(self.name, notes="skip — no files")
            return state

        files_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in state.generated_files.items()
        )

        print(f"\n📋 Reviewer: Analyzing {len(state.generated_files)} file(s)...")
        print(f"   Files to review: {', '.join(state.generated_files.keys())}")

        rules_context = f"\nUSER CODING RULES (MANDATORY):\n{state.user_rules}\n" if state.user_rules else ""

        prompt = f"""
Review the following generated backend code strictly.

ORIGINAL TASK: {state.task_prompt}
{rules_context}
ARCHITECT'S PLAN SUMMARY:
{state.plan_summary}

GENERATED FILES:
{files_block}

Perform a thorough review covering all 8 dimensions in your role description.

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

You MUST output review analysis, then END with EXACTLY ONE of these:

Example 1 (if code is good):
  FILES_WITH_ISSUES: None
  VERDICT: PASS

Example 2 (if code has issues):
  FILES_WITH_ISSUES: src/auth/login.py
  VERDICT: REJECT
  REASON: Missing authentication guard in login handler (line 42). Allows unauthorized access.

RULES:
  • "VERDICT:" must be on its own line
  • After VERDICT: line, add nothing else EXCEPT optional REASON: line for REJECT
  • Do NOT use other formats like "VERDICT:PASS" or "VEREDICT:" (typos will fail)
  • If uncertain, always use VERDICT: REJECT

FORMAT CHECK before responding:
  ☐ Review covers all 8 dimensions
  ☐ Ends with "VERDICT: PASS" OR ("VERDICT: REJECT" + REASON: line)
  ☐ No text after the REASON: line
  ☐ VERDICT: word is spelled correctly
─────────────────────────────────────────────────────────────────────

START REVIEW:
"""
        response_text, tokens = self._call_llm(state, prompt)
        verdict = _parse_verdict(response_text)

        output = ReviewerOutput(review_notes=response_text, verdict=verdict)
        state.apply(output)

        # Parse and display review feedback
        if verdict == "REJECT":
            state.review_retry_count += 1
            print(f"\n❌ Review REJECTED")
            # Extract reason from response
            reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()
                print(f"\n🔴 Issues Found:")
                print(f"   {reason}")
            else:
                # Extract what went wrong from the review notes
                lines = response_text.split('\n')
                issues = [line for line in lines if any(
                    keyword in line.lower() for keyword in 
                    ['error', 'issue', 'missing', 'vulnerable', 'incorrect', 'failed', 'problem']
                )]
                if issues:
                    print(f"\n🔴 Issues Found:")
                    for issue in issues[:5]:  # Show top 5 issues
                        if issue.strip():
                            print(f"   • {issue.strip()}")
            print(f"\n💡 Action: Sending back to Coder for fixes...")
            state.log(self.name, tokens=tokens, notes="REJECTED - needs rework")
        else:
            print(f"✅ Review PASSED")
            print(f"   ✓ All files meet quality standards")
            state.log(self.name, tokens=tokens, notes="PASSED")

        return state


def _parse_verdict(text: str) -> str:
    """
    Parse reviewer verdict with fallback strategies. Defaults to REJECT if no verdict found.
    
    Tries multiple patterns:
    1. Explicit "VERDICT: PASS|REJECT" line
    2. Files_with_issues: None pattern
    3. Summary context clues (No issues, All tests pass, etc.)
    
    Fail-safe: ambiguous response → REJECT and re-review
    """
    # Strategy 1: Explicit VERDICT line (primary)
    match = re.search(r"VERDICT:\s*(PASS|REJECT)", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()
    
    # Strategy 2: FILES_WITH_ISSUES pattern
    if "FILES_WITH_ISSUES: None" in text or "FILES_WITH_ISSUES:None" in text:
        return "PASS"
    
    # Strategy 3: Check for explicit rejection patterns
    if re.search(r"(REJECT|needs fixing|must fix|critical issue)", text, re.IGNORECASE):
        return "REJECT"
    
    # Strategy 4: Check for explicit pass patterns
    if re.search(r"(code is good|all standards|no issues|passes all|looks good)", text, re.IGNORECASE):
        return "PASS"
    
    # Fail-safe: ambiguous response → reject and re-review
    print(
        "[Reviewer] ⚠️  Warning: Could not determine verdict from response. "
        "Defaulting to REJECT (fail-safe). Response snippet: "
        + text[:200].replace('\n', ' ')
    )
    return "REJECT"
