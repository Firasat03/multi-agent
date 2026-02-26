"""
state.py — Shared Pipeline State for BE Multi-Agent Workflow

PipelineState is the single source of truth passed between the Orchestrator
and all agents. No direct agent-to-agent communication; everything goes via
this object.

Changes vs original:
  - Migrated to Pydantic v2 for robust serialization / schema evolution
    (extra fields in checkpoints are silently ignored — no more setdefault hacks)
  - Added typed per-agent output dataclasses so each agent declares exactly
    what it writes; the Orchestrator merges outputs into the central state
  - Added cost tracking fields (estimated_cost_usd)
  - Added pipeline_errors list for structured failure records
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator
from config import Status


# ─── Audit ────────────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    """One log entry per agent run."""
    agent: str
    status: str
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    estimated_cost_usd: float = 0.0


# ─── Plan ─────────────────────────────────────────────────────────────────────

class PlanItem(BaseModel):
    """A single item in the Architect's plan."""
    file: str
    action: str                     # "CREATE" | "MODIFY" | "DELETE"
    description: str
    api_contract: str = ""
    scope_estimate: str = ""

    @model_validator(mode="after")
    def validate_action(self) -> "PlanItem":
        if self.action not in ("CREATE", "MODIFY", "DELETE"):
            raise ValueError(f"PlanItem.action must be CREATE/MODIFY/DELETE, got: {self.action!r}")
        return self


# ─── Structured agent output contracts ───────────────────────────────────────
# Each agent declares exactly what it produces. The Orchestrator merges these
# into PipelineState. This prevents accidental cross-agent field writes.

class ArchitectOutput(BaseModel):
    plan: list[PlanItem] = Field(default_factory=list)
    plan_summary: str = ""
    task_checklist: str = ""


class CoderOutput(BaseModel):
    generated_files: dict[str, str] = Field(default_factory=dict)


class ReviewerOutput(BaseModel):
    review_notes: str = ""
    verdict: str = "PASS"           # "PASS" | "REJECT"


class TesterOutput(BaseModel):
    test_files: dict[str, str] = Field(default_factory=dict)
    test_output: dict[str, Any] = Field(default_factory=dict)
    static_analysis_output: Optional[str] = None
    error_log: Optional[str] = None


class DebuggerOutput(BaseModel):
    fix_instructions: Optional[str] = None
    confidence: int = 3             # 1-5
    escalate: bool = False          # True → human review needed


class IntegrationOutput(BaseModel):
    integration_test_output: Optional[str] = None
    integration_passed: Optional[bool] = None
    error_log: Optional[str] = None


class WriterOutput(BaseModel):
    docs_updated: bool = False


class SecurityOutput(BaseModel):
    security_report: str = ""
    fix_instructions: Optional[str] = None   # Non-None only when HIGH findings block pipeline


class DevOpsOutput(BaseModel):
    devops_files: dict[str, str] = Field(default_factory=dict)


# ─── Pipeline failure record ──────────────────────────────────────────────────

class PipelineError(BaseModel):
    """Structured record written when a pipeline stage fails terminally."""
    stage: str
    agent: str
    error_summary: str
    error_detail: str = ""
    files_at_failure: dict[str, str] = Field(default_factory=dict)
    fix_instructions_attempted: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ─── Central state ────────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Central state for one pipeline run. Pydantic v2 model.

    Key design decisions:
      - model_config extra="ignore" means any unknown fields in a checkpoint
        JSON are silently dropped — schema evolution is safe automatically.
      - All fields have defaults so partial checkpoints always load cleanly.
      - Cost tracking is accumulated by BaseAgent after every LLM call.
    """

    model_config = {"extra": "ignore"}  # schema-evolution safety

    # ── Identity ────────────────────────────────────────────────────────────
    run_id: str = Field(
        default_factory=lambda: (
            f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
    )
    status: str = Status.INIT

    # ── Input ────────────────────────────────────────────────────────────────
    task_prompt: str = ""
    project_root: str = ""
    language: str = "auto"

    # ── User rules ───────────────────────────────────────────────────────────
    user_rules: str = ""
    active_rules_file: str = ""

    # ── Architect ────────────────────────────────────────────────────────────
    plan: list[PlanItem] = Field(default_factory=list)
    plan_summary: str = ""
    task_checklist: str = ""
    replan_count: int = 0

    # ── Human gate ───────────────────────────────────────────────────────────
    plan_approved: bool = False
    user_feedback: Optional[str] = None

    # ── Coder ────────────────────────────────────────────────────────────────
    generated_files: dict[str, str] = Field(default_factory=dict)

    # ── Reviewer ─────────────────────────────────────────────────────────────
    review_notes: Optional[str] = None
    review_verdict: str = "PASS"
    review_retry_count: int = 0

    # ── Tester ───────────────────────────────────────────────────────────────
    test_files: dict[str, str] = Field(default_factory=dict)
    test_output: dict[str, Any] = Field(default_factory=dict)
    static_analysis_output: Optional[str] = None

    # ── Debugger ─────────────────────────────────────────────────────────────
    error_log: Optional[str] = None
    fix_instructions: Optional[str] = None
    retry_count: int = 0

    # ── Integration ──────────────────────────────────────────────────────────
    integration_test_output: Optional[str] = None
    integration_passed: Optional[bool] = None

    # ── Security ─────────────────────────────────────────────────────────────
    security_report: Optional[str] = None

    # ── Writer ───────────────────────────────────────────────────────────────
    docs_updated: bool = False

    # ── DevOps ───────────────────────────────────────────────────────────────
    devops_mode: Optional[str] = None
    devops_files: dict[str, str] = Field(default_factory=dict)

    # ── Cost & audit ─────────────────────────────────────────────────────────
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    audit_trail: list[AuditEntry] = Field(default_factory=list)

    # ── Failure records ───────────────────────────────────────────────────────
    pipeline_errors: list[PipelineError] = Field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def log(
        self,
        agent: str,
        notes: str = "",
        tokens: int = 0,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.audit_trail.append(AuditEntry(
            agent=agent,
            status=self.status,
            tokens_used=tokens,
            duration_ms=duration_ms,
            notes=notes,
            estimated_cost_usd=cost_usd,
        ))
        self.total_tokens_used += tokens
        self.estimated_cost_usd += cost_usd

    def record_failure(
        self,
        stage: str,
        agent: str,
        error_summary: str,
        error_detail: str = "",
        fix_instructions: Optional[str] = None,
    ) -> None:
        """Write a structured failure record for post-mortem analysis."""
        self.pipeline_errors.append(PipelineError(
            stage=stage,
            agent=agent,
            error_summary=error_summary,
            error_detail=error_detail,
            files_at_failure=dict(self.generated_files),
            fix_instructions_attempted=fix_instructions,
        ))

    def test_passed(self) -> bool:
        has_static_errors = bool(self.static_analysis_output)
        has_runtime_errors = self.test_output.get("returncode", 1) != 0
        return not has_static_errors and not has_runtime_errors

    # ── Merge typed agent outputs ─────────────────────────────────────────────

    def apply(self, output: object) -> None:
        """
        Merge a typed agent output object into central state.
        Only fields defined on the output type are written — no accidental
        cross-agent writes.
        """
        if isinstance(output, ArchitectOutput):
            self.plan = output.plan
            self.plan_summary = output.plan_summary
            self.task_checklist = output.task_checklist

        elif isinstance(output, CoderOutput):
            self.generated_files.update(output.generated_files)

        elif isinstance(output, ReviewerOutput):
            self.review_notes = output.review_notes
            self.review_verdict = output.verdict

        elif isinstance(output, TesterOutput):
            self.test_files.update(output.test_files)
            self.test_output = output.test_output
            self.static_analysis_output = output.static_analysis_output
            self.error_log = output.error_log

        elif isinstance(output, DebuggerOutput):
            self.fix_instructions = output.fix_instructions
            self.static_analysis_output = None  # cleared after debug

        elif isinstance(output, IntegrationOutput):
            self.integration_test_output = output.integration_test_output
            self.integration_passed = output.integration_passed
            if output.error_log:
                self.error_log = output.error_log

        elif isinstance(output, WriterOutput):
            self.docs_updated = output.docs_updated

        elif isinstance(output, SecurityOutput):
            self.security_report = output.security_report
            if output.fix_instructions is not None:
                self.fix_instructions = output.fix_instructions

        elif isinstance(output, DevOpsOutput):
            self.devops_files.update(output.devops_files)

        else:
            raise TypeError(f"Unknown agent output type: {type(output)}")

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        """
        Restore from checkpoint. Extra/unknown fields are silently ignored
        (model_config extra='ignore') so old checkpoints always load safely.
        """
        return cls.model_validate(data)