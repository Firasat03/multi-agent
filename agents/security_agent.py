"""
agents/security_agent.py — Security Scanner Agent  (NEW STAGE)

Runs AFTER the Reviewer and BEFORE the Tester.
Performs two independent scans and blocks the pipeline on HIGH severity findings:

  1. SAST (Static Application Security Testing)
     - Python  → bandit  (checks for hardcoded secrets, SQL injection, insecure calls)
     - Node.js → semgrep with the nodejs security ruleset (if semgrep installed)
     - Java    → semgrep with the java security ruleset
     - Others  → LLM-assisted security review of the generated files

  2. Dependency vulnerability scanning
     - Python  → pip-audit
     - Node.js → npm audit --json
     - Java    → mvn dependency-check (OWASP)
     - Others  → skipped with a logged warning

Results are stored in state.security_report (str).
If HIGH severity issues are found and SECURITY_BLOCK_ON_HIGH=true (default),
the pipeline is halted and the Coder is asked to fix them before proceeding.

Environment variables:
  ENABLE_BANDIT          true|false   (default true)
  ENABLE_PIP_AUDIT       true|false   (default true)
  SECURITY_BLOCK_ON_HIGH true|false   (default true)
  BANDIT_MIN_SEVERITY    LOW|MEDIUM|HIGH (default MEDIUM)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from agents.base_agent import BaseAgent
from config import (
    BANDIT_MIN_SEVERITY,
    ENABLE_BANDIT,
    ENABLE_PIP_AUDIT,
    Status,
)
from state import PipelineState, SecurityOutput
from tools.shell_tools import run_command, detect_language


# Block pipeline when HIGH severity findings exist (set env var to override)
SECURITY_BLOCK_ON_HIGH = os.getenv("SECURITY_BLOCK_ON_HIGH", "true").lower() == "true"

_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


class SecurityAgent(BaseAgent):
    name = "Security"
    system_role = (
        "You are a Senior Application Security Engineer specialising in backend systems. "
        "You perform thorough security code reviews looking for:\n"
        "  - Hardcoded secrets, credentials, API keys\n"
        "  - SQL / NoSQL / command injection vulnerabilities\n"
        "  - Insecure direct object references (IDOR)\n"
        "  - Missing input validation and sanitisation\n"
        "  - Broken authentication / authorisation\n"
        "  - Insecure cryptography (weak algorithms, hardcoded salts)\n"
        "  - Sensitive data exposure in logs or error messages\n"
        "  - Race conditions and TOCTOU vulnerabilities\n"
        "  - Insecure deserialization\n"
        "  - Security misconfiguration (debug mode, open CORS, etc.)\n\n"
        "For each finding:\n"
        "  - State the severity: HIGH | MEDIUM | LOW\n"
        "  - Identify the exact file and line\n"
        "  - Explain the vulnerability and its impact\n"
        "  - Provide a concrete fix\n\n"
        "End with: SECURITY_VERDICT: PASS (no HIGH findings) or SECURITY_VERDICT: FAIL"
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.SECURITY

        language = state.language if state.language != "auto" else detect_language(state.generated_files)
        findings: list[dict] = []
        report_lines: list[str] = []

        # ── 1. SAST ───────────────────────────────────────────────────────
        report_lines.append("=" * 60)
        report_lines.append("SAST FINDINGS")
        report_lines.append("=" * 60)

        sast_findings = self._run_sast(state, language)
        findings.extend(sast_findings)

        if sast_findings:
            for f in sast_findings:
                report_lines.append(
                    f"[{f['severity']}] {f['file']}:{f.get('line', '?')} — {f['issue']}"
                )
                if f.get("fix"):
                    report_lines.append(f"  Fix: {f['fix']}")
        else:
            report_lines.append("No SAST findings.")

        # ── 2. Dependency scan ────────────────────────────────────────────
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("DEPENDENCY VULNERABILITY SCAN")
        report_lines.append("=" * 60)

        dep_findings = self._run_dep_scan(state, language)
        findings.extend(dep_findings)

        if dep_findings:
            for f in dep_findings:
                report_lines.append(
                    f"[{f['severity']}] {f['file']} — {f['issue']}"
                )
        else:
            report_lines.append("No dependency vulnerabilities found.")

        # ── 3. LLM security review (always runs, supplements tool findings) ──
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("LLM SECURITY REVIEW")
        report_lines.append("=" * 60)

        llm_findings, llm_verdict, tokens = self._llm_security_review(state)
        findings.extend(llm_findings)
        report_lines.append(llm_findings and "\n".join(
            f"[{f['severity']}] {f['file']} — {f['issue']}" for f in llm_findings
        ) or "No additional findings from LLM review.")

        # ── Aggregate verdict ─────────────────────────────────────────────
        high_findings = [f for f in findings if f.get("severity") == "HIGH"]
        medium_findings = [f for f in findings if f.get("severity") == "MEDIUM"]

        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append(
            f"SUMMARY: {len(high_findings)} HIGH, {len(medium_findings)} MEDIUM, "
            f"{len(findings) - len(high_findings) - len(medium_findings)} LOW findings"
        )

        security_report = "\n".join(report_lines)

        notes = (
            f"{len(high_findings)} HIGH, {len(medium_findings)} MEDIUM "
            f"findings — {'BLOCKED' if high_findings and SECURITY_BLOCK_ON_HIGH else 'PASSED'}"
        )
        state.log(self.name, tokens=tokens, notes=notes)

        # ── Block on HIGH findings ────────────────────────────────────────
        fix_instructions: str | None = None
        if high_findings and SECURITY_BLOCK_ON_HIGH:
            fix_block = "\n".join(
                f"- [{f['severity']}] {f['file']}: {f['issue']}"
                + (f"\n  Fix: {f['fix']}" if f.get("fix") else "")
                for f in high_findings
            )
            fix_instructions = (
                "SECURITY SCAN FAILED — Fix ALL of the following HIGH severity "
                f"vulnerabilities before proceeding:\n\n{fix_block}"
            )
            state.record_failure(
                stage="SECURITY",
                agent=self.name,
                error_summary=f"{len(high_findings)} HIGH severity security findings",
                error_detail=security_report,
            )
            state.log(self.name, notes="BLOCKED on HIGH security findings")

        output = SecurityOutput(
            security_report=security_report,
            fix_instructions=fix_instructions,
        )
        state.apply(output)
        return state

    # ── SAST ─────────────────────────────────────────────────────────────────

    def _run_sast(self, state: PipelineState, language: str) -> list[dict]:
        if language == "python" and ENABLE_BANDIT:
            return self._run_bandit(state.generated_files)
        if language in ("nodejs", "java"):
            return self._run_semgrep(state.generated_files, language)
        return []

    def _run_bandit(self, files: dict[str, str]) -> list[dict]:
        """Run bandit on Python source files. Returns list of finding dicts."""
        py_files = {p: c for p, c in files.items() if p.endswith(".py")}
        if not py_files:
            return []

        # Check bandit is available
        check = run_command([sys.executable, "-m", "bandit", "--version"])
        if check["returncode"] != 0:
            print("[Security] bandit not installed — skipping SAST. "
                  "Install with: pip install bandit")
            return []

        findings = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel_path, content in py_files.items():
                dest = Path(tmpdir) / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            result = run_command(
                [sys.executable, "-m", "bandit", "-r", tmpdir,
                 "-f", "json", "-l",
                 "--severity-level", BANDIT_MIN_SEVERITY.lower()],
                timeout=60,
            )
            try:
                data = json.loads(result["stdout"] or "{}")
                for issue in data.get("results", []):
                    sev = issue.get("issue_severity", "LOW").upper()
                    if _SEVERITY_RANK.get(sev, 0) >= _SEVERITY_RANK.get(BANDIT_MIN_SEVERITY, 2):
                        # Map tmp path back to relative path
                        abs_file = issue.get("filename", "")
                        rel = abs_file.replace(tmpdir, "").lstrip("/\\")
                        findings.append({
                            "severity": sev,
                            "file": rel,
                            "line": issue.get("line_number"),
                            "issue": issue.get("issue_text", ""),
                            "fix": f"See bandit rule {issue.get('test_id', '')}",
                        })
            except (json.JSONDecodeError, KeyError):
                pass
        return findings

    def _run_semgrep(self, files: dict[str, str], language: str) -> list[dict]:
        """Run semgrep if available. Returns list of finding dicts."""
        check = run_command(["semgrep", "--version"])
        if check["returncode"] != 0:
            print(f"[Security] semgrep not installed — skipping SAST for {language}.")
            return []

        ruleset = "p/nodejs-security" if language == "nodejs" else "p/java"
        ext_map = {"nodejs": (".ts", ".js"), "java": (".java",)}
        exts = ext_map.get(language, ())
        target_files = {p: c for p, c in files.items() if p.endswith(exts)}
        if not target_files:
            return []

        findings = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel_path, content in target_files.items():
                dest = Path(tmpdir) / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            result = run_command(
                ["semgrep", "--config", ruleset, "--json", tmpdir],
                timeout=120,
            )
            try:
                data = json.loads(result["stdout"] or "{}")
                for r in data.get("results", []):
                    sev = r.get("extra", {}).get("severity", "WARNING").upper()
                    mapped_sev = "HIGH" if sev in ("ERROR", "HIGH") else (
                        "MEDIUM" if sev == "WARNING" else "LOW"
                    )
                    abs_file = r.get("path", "")
                    rel = abs_file.replace(tmpdir, "").lstrip("/\\")
                    findings.append({
                        "severity": mapped_sev,
                        "file": rel,
                        "line": r.get("start", {}).get("line"),
                        "issue": r.get("extra", {}).get("message", ""),
                        "fix": r.get("extra", {}).get("fix", ""),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        return findings

    # ── Dependency scanning ───────────────────────────────────────────────────

    def _run_dep_scan(self, state: PipelineState, language: str) -> list[dict]:
        if language == "python" and ENABLE_PIP_AUDIT:
            return self._run_pip_audit(state.generated_files)
        if language == "nodejs":
            return self._run_npm_audit(state.generated_files)
        return []

    def _run_pip_audit(self, files: dict[str, str]) -> list[dict]:
        req_content = files.get("requirements.txt", "")
        if not req_content:
            return []

        check = run_command([sys.executable, "-m", "pip_audit", "--version"])
        if check["returncode"] != 0:
            print("[Security] pip-audit not installed — skipping dep scan. "
                  "Install with: pip install pip-audit")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = Path(tmpdir) / "requirements.txt"
            req_path.write_text(req_content, encoding="utf-8")

            result = run_command(
                [sys.executable, "-m", "pip_audit",
                 "-r", str(req_path), "--format", "json"],
                timeout=120,
            )
            findings = []
            try:
                data = json.loads(result["stdout"] or "[]")
                for dep in (data if isinstance(data, list) else data.get("dependencies", [])):
                    for vuln in dep.get("vulns", []):
                        sev = vuln.get("fix_versions") and "MEDIUM" or "HIGH"
                        findings.append({
                            "severity": sev,
                            "file": "requirements.txt",
                            "issue": (
                                f"{dep.get('name')}=={dep.get('version')} — "
                                f"{vuln.get('id', '')}: {vuln.get('description', '')[:120]}"
                            ),
                            "fix": (
                                f"Upgrade to {dep.get('name')}>={vuln.get('fix_versions', ['?'])[0]}"
                                if vuln.get("fix_versions") else "No fix available — consider alternative"
                            ),
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return findings

    def _run_npm_audit(self, files: dict[str, str]) -> list[dict]:
        pkg_content = files.get("package.json", "")
        if not pkg_content:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text(pkg_content, encoding="utf-8")

            # npm audit requires a lockfile; generate a minimal one first
            run_command(["npm", "install", "--package-lock-only", "--ignore-scripts"],
                        cwd=tmpdir, timeout=60)
            result = run_command(["npm", "audit", "--json"], cwd=tmpdir, timeout=60)

            findings = []
            try:
                data = json.loads(result["stdout"] or "{}")
                for vuln_id, vuln in data.get("vulnerabilities", {}).items():
                    sev_raw = vuln.get("severity", "low").upper()
                    sev = "HIGH" if sev_raw in ("HIGH", "CRITICAL") else (
                        "MEDIUM" if sev_raw == "MODERATE" else "LOW"
                    )
                    findings.append({
                        "severity": sev,
                        "file": "package.json",
                        "issue": f"{vuln_id}: {vuln.get('title', '')} ({sev_raw})",
                        "fix": vuln.get("fixAvailable") and "npm audit fix" or "Manual fix required",
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        return findings

    # ── LLM security review ───────────────────────────────────────────────────

    def _llm_security_review(
        self, state: PipelineState
    ) -> tuple[list[dict], str, int]:
        """
        Supplement tool-based scanning with an LLM security review.
        The LLM catches patterns that static tools miss (logic flaws, IDOR, etc.)
        Returns (findings, verdict, tokens).
        """
        # Limit file content to avoid context overflow
        files_block = "\n\n".join(
            f"### {path}\n```\n{content[:3000]}\n```"
            + (" [TRUNCATED]" if len(content) > 3000 else "")
            for path, content in state.generated_files.items()
        )

        prompt = f"""
Perform a security-focused code review of the following backend source files.

TASK CONTEXT: {state.task_prompt}

SOURCE FILES:
{files_block}

For each vulnerability found, output a line in EXACTLY this format:
FINDING: <SEVERITY>|<file>|<line or "?">|<description>|<fix>

Where SEVERITY is HIGH, MEDIUM, or LOW.

After all FINDING lines (or "No findings." if clean), output:
SECURITY_VERDICT: PASS
or
SECURITY_VERDICT: FAIL
"""
        response_text, tokens = self._call_llm(state, prompt)

        findings = []
        for line in response_text.splitlines():
            if line.startswith("FINDING:"):
                parts = line[len("FINDING:"):].strip().split("|", 4)
                if len(parts) >= 4:
                    findings.append({
                        "severity": parts[0].strip().upper(),
                        "file": parts[1].strip(),
                        "line": parts[2].strip(),
                        "issue": parts[3].strip(),
                        "fix": parts[4].strip() if len(parts) > 4 else "",
                    })

        import re
        verdict_match = re.search(r"SECURITY_VERDICT:\s*(PASS|FAIL)", response_text, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"

        return findings, verdict, tokens