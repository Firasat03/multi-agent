"""
agents/reviewer_agent.py — Reviewer Agent

Changes vs original:
  - Refocused on practical code functionality (imports, function calls, cross-file consistency)
  - Removed production-level concerns (PII, observability patterns, security details)
  - Verdict parsing defaults to REJECT (fail-safe)
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
        "You are a Code Reviewer focused on ensuring code will actually run. "
        "Your review MUST verify:\n\n"
        "CODE FILES:\n"
        "1. IMPORTS — All imported modules/functions actually exist in standard libraries or are defined in project files\n"
        "2. FUNCTION CALLS — Every function/method called is actually defined somewhere (in the file or imported)\n"
        "3. CROSS-FILE CONSISTENCY — Functions called in one file are properly defined in other files\n"
        "4. PARAMETER MATCHING — Function calls match the defined signatures (right number/types of parameters)\n"
        "5. CLASS/OBJECT USAGE — Classes are properly instantiated before use; methods exist on objects\n"
        "6. VARIABLE SCOPE — Variables are defined before use; no undefined variable references\n"
        "7. SYNTAX CORRECTNESS — Code has valid syntax for the language (no obvious typos or malformed code)\n"
        "8. USER CODING RULES (RULES.md) — any violation is an automatic REJECT\n\n"
        "DEPENDENCY & CONFIGURATION FILES (pom.xml, requirements.txt, package.json, application.properties, config.py, etc.):\n"
        "9. DEPENDENCY VALIDITY — All listed dependencies are real packages (correct names, versions exist)\n"
        "10. DEPENDENCY CONFLICTS — No version conflicts or incompatible dependency combinations\n"
        "11. REQUIRED DEPENDENCIES — All dependencies imported/used in code are listed in dependency files\n"
        "12. CONFIGURATION SYNTAX — Property files have valid syntax (no malformed entries, proper delimiters)\n"
        "13. CONFIGURATION REFERENCES — Config values referenced in code are actually defined\n"
        "14. PROPERTY VALUE VALIDITY — Configuration values have valid formats for their context\n"
        "15. APPLICATION SETTINGS — application.properties/config.py values are properly formatted and complete\n\n"
        "Focus ONLY on whether the code will actually execute. Ignore:\n"
        "  - Security vulnerabilities (that's a separate security scan)\n"
        "  - Performance optimization\n"
        "  - Code style or naming conventions (unless breaking imports)\n"
        "  - PII in logs or observability patterns\n"
        "  - Transaction boundaries\n"
        "  - Production-grade quality standards\n\n"
        "Be strict and specific. Quote the exact line, dependency name, or config key that has the problem.\n\n"
        "CRITICAL — MANDATORY OUTPUT FORMAT:\n"
        "You MUST end your review with EXACTLY this format:\n\n"
        "FILES_WITH_ISSUES: <Comma-separated list of relative paths needing fixes, or 'None'>\n"
        "VERDICT: PASS\n\n"
        "OR if there are issues:\n\n"
        "FILES_WITH_ISSUES: src/auth/login.py, src/config.py\n"
        "VERDICT: REJECT\n"
        "REASON: <One sentence: what function/import/dependency/variable is broken>\n\n"
        "The machine parser looks for exactly 'VERDICT: PASS' or 'VERDICT: REJECT'.\n"
        "The 'FILES_WITH_ISSUES' line tells the Coder which files to regenerate.\n"
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
Review the following generated backend code. Focus ONLY on whether it will actually run.

ORIGINAL TASK: {state.task_prompt}
{rules_context}
ARCHITECT'S PLAN SUMMARY:
{state.plan_summary}

GENERATED FILES:
{files_block}

Your review MUST check CODE FILES:
  1. IMPORTS: Do all imported modules/functions actually exist or are they defined in the code?
  2. FUNCTION CALLS: Is every function/method actually defined before it's called?
  3. CROSS-FILE REFERENCES: If file A calls a function from file B, is it properly exported/available?
  4. PARAMETER MATCHING: Do function calls have the right number and types of arguments?
  5. VARIABLE SCOPE: Are all variables defined before use? No undefined references?
  6. SYNTAX: Is the code syntactically correct for the language?
  7. CLASS INSTANTIATION: Are objects created before methods are called on them?

Your review MUST check DEPENDENCY & CONFIGURATION FILES (pom.xml, requirements.txt, package.json, application.properties, config.py, etc.):
  8. DEPENDENCY VALIDITY: Are all listed dependencies real packages with correct names and valid versions?
  9. DEPENDENCY CONFLICTS: Are there any version conflicts or incompatible dependency combinations?
  10. REQUIRED DEPENDENCIES: Are all dependencies imported or used in code actually listed in dependency files?
  11. CONFIGURATION SYNTAX: Do property files have valid syntax (proper formatting, no malformed entries)?
  12. CONFIGURATION REFERENCES: Are all config values referenced in code actually defined in config files?
  13. PROPERTY VALUE VALIDITY: Are configuration values properly formatted for their context?
  14. APPLICATION SETTINGS: Are application.properties/config.py values complete and properly formatted?

IGNORE: security, performance, style, logging patterns, PII handling, transaction boundaries, production standards.

─────────────────────────────────────────────────────────────────────
MANDATORY OUTPUT FORMAT (machine-parsed, no exceptions)

STEP 1: Write your detailed review (checking the 7 dimensions above)
Write the specific issues found (missing imports, undefined functions, etc.)

STEP 2: At the END, output EXACTLY ONE of these patterns:

PATTERN A (if code is runnable, no issues):
═════════════════════════════════════
FILES_WITH_ISSUES: None
VERDICT: PASS
═════════════════════════════════════

PATTERN B (if code will not run):
═════════════════════════════════════
FILES_WITH_ISSUES: src/auth/login.py, src/models/user.py
VERDICT: REJECT
REASON: login.py calls hash_password() but it's not imported from utils; also User class not defined in models/user.py.
═════════════════════════════════════

CRITICAL RULES:
  1. Output ONLY one pattern above
  2. VERDICT must be exactly "VERDICT: PASS" or "VERDICT: REJECT"
  3. FILES_WITH_ISSUES line must come BEFORE VERDICT line
  4. For REJECT: REASON must be ONE sentence max, specific about what's broken
  5. Do NOT output any text AFTER the dashed lines
  6. If unsure, choose VERDICT: REJECT (fail-safe)
  7. The dashes (═══) help parsing

EXAMPLES (to clarify what to look for):

BAD - REJECT (Code Issues): 
  - Function calls processor.process(data) but processor is not imported
  - Class User defined in models.py but not imported in auth.py where it's used
  - Function signature: def create_user(name) but called with create_user(name, email, age)
  - Variable user referenced before it's assigned

BAD - REJECT (Dependency/Config Issues):
  - requirements.txt lists 'django==4.0.5' but code imports 'djangorestframework' which is not listed
  - package.json missing 'express' dependency but code has const express = require('express')
  - pom.xml has version conflict: one module needs spring-boot 2.5 but another requires 3.0
  - application.properties references 'database.url' but code looks for 'db.host' and 'db.port'
  - config.py references environment variable '${DB_PASSWORD}' but it's malformed (should be ${DB_PASSWORD})
  - requirements.txt lists 'psycopg2' with invalid version string like 'psycopg2==abc' (not a real version)

GOOD - PASS:
  - All imports exist (standard library or defined in project)
  - All functions are defined before calling
  - Cross-file calls use proper imports
  - Function signatures match their calls
  - All imported dependencies are listed in dependency files with valid versions
  - Config values referenced in code are defined in configuration files
  - No dependency version conflicts
  - Property syntax is valid and complete

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
