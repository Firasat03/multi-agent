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

        is_retry = state.review_retry_count > 0
        
        target_files = state.generated_files
        if is_retry and hasattr(state, 'modified_source_files') and state.modified_source_files:
            target_files = {
                path: content for path, content in state.generated_files.items()
                if path in state.modified_source_files
            }
            print(f"\n📋 Reviewer (Incremental Mode): Analyzing {len(target_files)} modified file(s)...")
            print(f"   Files to review: {', '.join(target_files.keys())}")
        else:
            print(f"\n📋 Reviewer: Analyzing {len(target_files)} file(s)...")
            print(f"   Files to review: {', '.join(target_files.keys())}")

        if not target_files:
            output = ReviewerOutput(review_notes="No modified files to review.", verdict="PASS")
            state.apply(output)
            state.log(self.name, notes="skip — no files modified in retry")
            return state

        files_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in target_files.items()
        )

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

STEP 1: Write your detailed review (covering all 8 dimensions)

STEP 2: At the END, output EXACTLY ONE of these patterns on its own line:

PATTERN A (if code is good, no issues):
═════════════════════════════════════
FILES_WITH_ISSUES: None
VERDICT: PASS
═════════════════════════════════════

PATTERN B (if code has issues that need fixing):
═════════════════════════════════════
FILES_WITH_ISSUES: src/auth/login.py, src/config.py
VERDICT: REJECT
REASON: <one sentence: what is the critical issue>
═════════════════════════════════════

CRITICAL RULES FOR THIS SECTION:
  1. ONLY output one of the two patterns above
  2. Do NOT output both patterns
  3. VERDICT: must be exactly "VERDICT: PASS" or "VERDICT: REJECT" (no typos)
  4. FILES_WITH_ISSUES line must come BEFORE VERDICT line
  5. For REJECT cases, add a REASON: line that is ONE sentence max
  6. Do NOT output any text AFTER the REASON: line
  7. If unsure, always choose VERDICT: REJECT (fail-safe)
  8. The dashes (═══) on the lines before/after help parsing

EXAMPLE 1 (PASS case):
═════════════════════════════════════
FILES_WITH_ISSUES: None
VERDICT: PASS
═════════════════════════════════════

EXAMPLE 2 (REJECT case):
═════════════════════════════════════
FILES_WITH_ISSUES: src/auth/service/AuthService.java, src/auth/security/SecurityConfig.java
VERDICT: REJECT
REASON: Missing authentication on /login endpoint allows unauthorized token generation.
═════════════════════════════════════

DO NOT include any other text after the dashed lines.

START YOUR REVIEW:
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
    # Strategy 1: Look for dashed boundary format (new format)
    # Pattern: ═════════...
    #         FILES_WITH_ISSUES: ...
    #         VERDICT: PASS|REJECT
    #         [REASON: ...]
    #         ═════════...
    dashed_match = re.search(
        r"═+\s*\n\s*FILES_WITH_ISSUES:\s*(.+?)\n\s*VERDICT:\s*(PASS|REJECT)",
        text,
        re.IGNORECASE | re.MULTILINE
    )
    if dashed_match:
        verdict = dashed_match.group(2).upper()
        return verdict
    
    # Strategy 2: Explicit VERDICT line (primary fallback)
    match = re.search(r"VERDICT:\s*(PASS|REJECT)", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()
    
    # Strategy 3: FILES_WITH_ISSUES pattern
    if "FILES_WITH_ISSUES: None" in text or "FILES_WITH_ISSUES:None" in text:
        return "PASS"
    
    # Strategy 4: Check for explicit rejection patterns
    if re.search(r"(REJECT|needs fixing|must fix|critical issue|cannot pass|fails)", text, re.IGNORECASE):
        return "REJECT"
    
    # Strategy 5: Check for explicit pass patterns
    if re.search(r"(code is good|all standards|no issues|passes all|looks good|all files|acceptable|approved)", text, re.IGNORECASE):
        return "PASS"
    
    # Fail-safe: ambiguous response → reject and re-review
    print(
        "[Reviewer] ⚠️  Warning: Could not determine verdict from response. "
        "Defaulting to REJECT (fail-safe). Response snippet: "
        + text[:200].replace('\n', ' ')
    )
    return "REJECT"
