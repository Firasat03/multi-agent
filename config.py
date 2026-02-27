"""
config.py — Central configuration for BE Multi-Agent Workflow
"""

import os
from pathlib import Path

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",   "gemini")
LLM_MODEL      = os.getenv("LLM_MODEL",      "gemini-flash-latest")
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",   "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = LLM_MODEL

# ─── Retry limits ─────────────────────────────────────────────────────────────
MAX_DEBUG_RETRIES        = int(os.getenv("MAX_DEBUG_RETRIES",        "3"))
MAX_REVIEW_RETRIES       = int(os.getenv("MAX_REVIEW_RETRIES",       "1"))
MAX_INTEGRATION_RETRIES  = int(os.getenv("MAX_INTEGRATION_RETRIES",  "2"))
MAX_LLM_RETRIES          = int(os.getenv("MAX_LLM_RETRIES",          "5"))

# ─── Timeouts (seconds) ───────────────────────────────────────────────────────
AGENT_TIMEOUT_SECS       = int(os.getenv("AGENT_TIMEOUT_SECS",  "600"))   # per agent run
LLM_CALL_TIMEOUT_SECS    = int(os.getenv("LLM_CALL_TIMEOUT",    "120"))   # per LLM API call
INTEGRATION_TIMEOUT_SECS = int(os.getenv("INTEGRATION_TIMEOUT", "120"))   # server startup

# ─── Token / context budget ───────────────────────────────────────────────────
# Rough character-to-token ratio for budget estimation (conservative)
CHARS_PER_TOKEN          = 3.5
# Warn when estimated input tokens exceed this fraction of model context
CONTEXT_WARN_THRESHOLD   = 0.75
# Model context window sizes (tokens) — used for budget estimation
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash":            1_000_000,
    "gemini-1.5-pro":              1_000_000,
    "gpt-4o":                        128_000,
    "gpt-4o-mini":                   128_000,
    "claude-3-5-sonnet-20241022":    200_000,
    "claude-3-opus-20240229":        200_000,
    "llama3.1":                        8_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000

# ─── Cost estimation (USD per 1M tokens) ─────────────────────────────────────
# Input cost / output cost pairs. Used only for display; not enforced by API.
MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash":              (0.075,  0.30),
    "gemini-1.5-pro":                (1.25,   5.00),
    "gpt-4o":                        (2.50,  10.00),
    "gpt-4o-mini":                   (0.15,   0.60),
    "claude-3-5-sonnet-20241022":    (3.00,  15.00),
    "claude-3-opus-20240229":       (15.00,  75.00),
}
DEFAULT_COST_PER_1M = (1.00, 4.00)   # fallback

# Max spend per pipeline run (USD). Set to 0.0 to disable.
MAX_RUN_COST_USD = float(os.getenv("MAX_RUN_COST_USD", "0.0"))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent
RULES_DIR          = BASE_DIR / "rules"
DEFAULT_RULES_FILE = RULES_DIR / "RULES.md"
PROMPTS_DIR        = BASE_DIR / "prompts"
WORKFLOW_DIR       = BASE_DIR / ".workflow"
MCP_DIR            = BASE_DIR / "mcp"
MCP_CONFIG_FILE    = MCP_DIR / "agent_mcp_config.json"

# ─── LLM generation settings ──────────────────────────────────────────────────
GENERATION_CONFIG = {
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    "top_p": 0.95,
    "max_output_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192")),
}

# ─── Security scanning ────────────────────────────────────────────────────────
ENABLE_BANDIT       = os.getenv("ENABLE_BANDIT",    "true").lower() == "true"
ENABLE_PIP_AUDIT    = os.getenv("ENABLE_PIP_AUDIT", "true").lower() == "true"
BANDIT_MIN_SEVERITY = os.getenv("BANDIT_MIN_SEVERITY", "MEDIUM")   # LOW | MEDIUM | HIGH

# ─── Pipeline status enum values ──────────────────────────────────────────────
class Status:
    INIT        = "INIT"
    ARCHITECT   = "ARCHITECT"
    PLAN_REVIEW = "PLAN_REVIEW"
    CODING      = "CODING"
    REVIEWING   = "REVIEWING"
    TESTING     = "TESTING"
    DEBUGGING   = "DEBUGGING"
    SECURITY    = "SECURITY"        # new: SAST + dependency scan
    INTEGRATION = "INTEGRATION"
    WRITING     = "WRITING"
    DEVOPS      = "DEVOPS"
    DONE        = "DONE"
    FAILED      = "FAILED"
    ABORTED     = "ABORTED"
