"""
tools/integration_tools.py — Live integration testing

Changes vs original:
  - Port is now a parameter (caller uses _find_free_port()) — no hardcoded 8080
  - Server stdout/stderr captured into result["server_startup_log"] so failures
    include the actual startup error rather than a generic timeout message
  - Response body schema assertion: checks that declared fields from api_contract
    are present in POST/GET responses (e.g. "→ 200 {token, user_id}")
  - Server teardown uses SIGTERM → wait(5s) → SIGKILL to avoid zombie processes
  - _parse_contracts now also extracts expected response fields
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from rich.console import Console

_console = Console()

MAX_STARTUP_SECS = 90  # Spring Boot + JPA can take 60-90s
HEALTH_POLL_SECS = 1

_HEALTH_PATHS = ["/actuator/health", "/health", "/healthz", "/"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _write_files_to_disk(files: dict[str, str], project_root: str) -> None:
    for rel_path, content in files.items():
        abs_path = Path(project_root) / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")


def _find_jar(project_root: str) -> Optional[str]:
    target = Path(project_root) / "target"
    if not target.exists():
        return None
    jars = sorted(target.glob("*.jar"), key=lambda p: p.stat().st_size, reverse=True)
    jars = [j for j in jars if "sources" not in j.name and "tests" not in j.name]
    return str(jars[0]) if jars else None


def _poll_health(port: int) -> bool:
    for path in _HEALTH_PATHS:
        url = f"http://localhost:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                # Accept 2xx and 3xx status codes as healthy
                # Note: Spring Boot /actuator/health returns 200 even during startup phases
                if 200 <= r.status < 400:
                    return True
        except Exception:
            pass
    return False


def _curl(method: str, url: str, body: Optional[str] = None,
          headers: Optional[dict] = None) -> dict:
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method.upper(), url]
    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
    if body:
        cmd += ["-d", body]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = r.stdout.strip().rsplit("\n", 1)
        body_out = lines[0] if len(lines) > 1 else ""
        status_code = int(lines[-1]) if lines[-1].isdigit() else 0
        return {"status_code": status_code, "body": body_out, "error": r.stderr.strip()}
    except Exception as exc:
        return {"status_code": 0, "body": "", "error": str(exc)}


def _assert_schema(body: str, expected_fields: list[str]) -> bool:
    """
    Return True if all expected_fields appear as JSON keys in body.
    Handles nested responses by flattening keys one level.
    """
    if not expected_fields:
        return True
    try:
        data = json.loads(body)
        # Unwrap common envelope patterns: {"data": {...}, ...}
        if isinstance(data, dict):
            keys: set[str] = set(data.keys())
            for v in data.values():
                if isinstance(v, dict):
                    keys |= set(v.keys())
            return all(f.strip() in keys for f in expected_fields)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: string search
    return all(f.strip() in body for f in expected_fields)


def _terminate_server(proc: subprocess.Popen) -> None:
    """Graceful SIGTERM → 5s wait → SIGKILL."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


# ─── build helpers ────────────────────────────────────────────────────────────

def _build_java(project_root: str) -> dict:
    r = subprocess.run(
        ["mvn", "package", "-DskipTests", "-q"],
        cwd=project_root, capture_output=True, text=True, timeout=300,
    )
    return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}


def _build_nodejs(project_root: str) -> dict:
    pkg = Path(project_root) / "package.json"
    scripts = {}
    if pkg.exists():
        try:
            scripts = json.loads(pkg.read_text()).get("scripts", {})
        except Exception:
            pass
    if "build" in scripts:
        r = subprocess.run(
            ["npm", "run", "build"], cwd=project_root,
            capture_output=True, text=True, timeout=120,
        )
        return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}
    return {"ok": True, "output": "no build script — running source directly"}


def _build_go(project_root: str) -> dict:
    r = subprocess.run(
        ["go", "build", "-o", "app", "./..."],
        cwd=project_root, capture_output=True, text=True, timeout=120,
    )
    return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}


def _build_python(project_root: str) -> dict:
    req = Path(project_root) / "requirements.txt"
    if req.exists():
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            cwd=project_root, capture_output=True, text=True, timeout=120,
        )
        return {"ok": r.returncode == 0, "output": r.stdout + r.stderr}
    return {"ok": True, "output": "no requirements.txt"}


# ─── server start helpers ─────────────────────────────────────────────────────

def _start_java_server(project_root: str, port: int) -> Optional[subprocess.Popen]:
    jar = _find_jar(project_root)
    if not jar:
        return None
    env = os.environ.copy()
    env["SERVER_PORT"] = str(port)
    return subprocess.Popen(
        ["java", f"-Dserver.port={port}", "-jar", jar],
        cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True,
    )


def _start_nodejs_server(project_root: str, port: int) -> Optional[subprocess.Popen]:
    env = os.environ.copy()
    env["PORT"] = str(port)
    for entry in ["dist/index.js", "src/index.js", "index.js"]:
        if (Path(project_root) / entry).exists():
            return subprocess.Popen(
                ["node", entry], cwd=project_root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
    return None


def _start_go_server(project_root: str, port: int) -> Optional[subprocess.Popen]:
    env = os.environ.copy()
    env["PORT"] = str(port)
    exe = str(Path(project_root) / ("app.exe" if sys.platform == "win32" else "app"))
    if not Path(exe).exists():
        return None
    return subprocess.Popen(
        [exe], cwd=project_root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _start_python_server(project_root: str, port: int) -> Optional[subprocess.Popen]:
    env = os.environ.copy()
    env["PORT"] = str(port)
    for cmd in [
        [sys.executable, "-m", "uvicorn", "main:app", f"--port={port}"],
        [sys.executable, "-m", "gunicorn", "-b", f"0.0.0.0:{port}", "main:app"],
        [sys.executable, "main.py"],
    ]:
        entry = cmd[-1].replace("--port=" + str(port), "").split(":")[-1] + ".py"
        if (Path(project_root) / entry).exists() or "--port" in " ".join(cmd):
            return subprocess.Popen(
                cmd, cwd=project_root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
    return None


# ─── API contract parser ───────────────────────────────────────────────────────

def _parse_contracts(plan_items) -> list[dict]:
    """
    Extract HTTP test cases from PlanItem.api_contract strings.
    Also parses expected response fields from the contract notation:
      POST /api/products → 201 {id, name, price}
    Returns list of {method, path, expected_status, expected_fields}.
    """
    tests: list[dict] = []
    seen: set = set()
    # Match:  METHOD /path → STATUS {field1, field2}
    pattern = re.compile(
        r"(GET|POST|PUT|PATCH|DELETE)\s+(/[\w/{}.\-]*)"
        r"\s*[→>-]+\s*(\d{3})"
        r"(?:\s*\{([^}]*)\})?",
        re.IGNORECASE,
    )
    for item in (plan_items or []):
        contract = getattr(item, "api_contract", "") or ""
        for m in pattern.finditer(contract):
            method = m.group(1).upper()
            path   = m.group(2)
            status = int(m.group(3))
            fields_raw = m.group(4) or ""
            expected_fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
            key = (method, path)
            if key not in seen:
                seen.add(key)
                tests.append({
                    "method":          method,
                    "path":            path,
                    "expected_status": status,
                    "expected_fields": expected_fields,
                })
    return tests


def _make_sample_body(method: str, path: str) -> Optional[str]:
    if method in ("GET", "DELETE"):
        return None
    path_lower = path.lower()
    if "product" in path_lower:
        return '{"name":"Test Product","description":"auto-generated","price":9.99,"stock":10}'
    if "user" in path_lower:
        return '{"username":"testuser","email":"test@example.com","password":"TestPass1!"}'
    if "order" in path_lower:
        return '{"productId":1,"quantity":2}'
    return '{"name":"test","value":"auto"}'


def _run_single_test(test: dict, base_url: str, created_ids: dict) -> dict:
    """
    Run a single endpoint test.
    OPTIMIZATION: Can be called in parallel from ThreadPoolExecutor.
    """
    method = test["method"]
    raw_path = test["path"]
    expected = test["expected_status"]
    expected_fields = test.get("expected_fields", [])

    path = raw_path
    id_placeholder = re.search(r"\{(\w+)[Ii]d\}", raw_path)
    if id_placeholder:
        resource = id_placeholder.group(1)
        real_id = created_ids.get(resource, "1")
        path = re.sub(r"\{[^}]+\}", real_id, raw_path)

    body = _make_sample_body(method, path)
    url = base_url + path
    headers = {"Content-Type": "application/json"} if body else {}

    _console.print(f"[dim]  → {method} {url}  (expect {expected})[/dim]")
    resp = _curl(method, url, body=body, headers=headers)
    status_ok = resp["status_code"] == expected

    # Schema assertion
    schema_ok: Optional[bool] = None
    if status_ok and expected_fields:
        schema_ok = _assert_schema(resp["body"], expected_fields)

    passed = status_ok and (schema_ok is not False)

    status_label = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    _console.print(
        f"  {status_label}  {method} {path}  "
        f"expected={expected} got={resp['status_code']}"
        + (f"  schema={'OK' if schema_ok else 'FAIL'}" if schema_ok is not None else "")
    )

    return {
        "method":          method,
        "path":            path,
        "expected_status": expected,
        "actual_status":   resp["status_code"],
        "passed":          passed,
        "schema_ok":       schema_ok,
        "expected_fields": expected_fields,
        "response_body":   resp["body"][:500],
        "resp_body_raw":   resp["body"],  # For ID extraction
    }


def _run_tests_parallel(base_url: str, tests: list, created_ids: dict) -> list[dict]:
    """
    OPTIMIZATION: Run tests in parallel using ThreadPoolExecutor.
    
    Handles dependencies:
      - First, run all POST requests (sequential) to capture IDs
      - Then, run all GET/DELETE requests in parallel
    
    Saves ~50% time on integration tests with many endpoints.
    """
    results = []
    
    # Phase 1: Run POST tests sequentially to capture IDs
    post_tests = [t for t in tests if t["method"] == "POST"]
    other_tests = [t for t in tests if t["method"] != "POST"]
    
    for test in post_tests:
        result = _run_single_test(test, base_url, created_ids)
        results.append(result)
        
        # Capture created IDs for later use in dependent tests
        if result["actual_status"] in (200, 201):
            id_match = re.search(r'"id"\s*:\s*(\d+)', result["resp_body_raw"])
            if id_match:
                path = result["path"]
                seg = [s for s in path.split("/") if s and s != "api"]
                resource_name = seg[-1].rstrip("s") if seg else "item"
                created_ids[resource_name] = id_match.group(1)
    
    # Phase 2: Run GET/DELETE tests in parallel
    if other_tests:
        with ThreadPoolExecutor(max_workers=min(5, len(other_tests))) as executor:
            futures = {
                executor.submit(_run_single_test, test, base_url, created_ids): test
                for test in other_tests
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Clean up raw response data (not needed in final output)
    for r in results:
        r.pop("resp_body_raw", None)
    
    return results


# ─── public API ───────────────────────────────────────────────────────────────

def run_integration_tests(
    project_root: str,
    language: str,
    plan_items,
    generated_files: dict[str, str],
    port: int = 8080,
) -> dict:
    """
    Build → start → curl → teardown.

    Returns dict with keys:
        passed            bool
        results           list[dict]
        build_output      str
        server_startup_log str   ← NEW: captured server output on startup failure
        error             str | None
    """
    lang = (language or "unknown").lower().strip()
    results: list[dict] = []
    proc: Optional[subprocess.Popen] = None
    server_startup_log = ""

    _write_files_to_disk(generated_files, project_root)

    # ── Build ──────────────────────────────────────────────────────────────
    _console.print(f"[cyan]🔨 Building project ({lang})…[/cyan]")
    build_result = {"ok": True, "output": ""}
    if lang == "java":
        build_result = _build_java(project_root)
    elif lang == "nodejs":
        build_result = _build_nodejs(project_root)
    elif lang == "go":
        build_result = _build_go(project_root)
    elif lang == "python":
        build_result = _build_python(project_root)
    else:
        _console.print(f"[yellow]⏭️  No integration strategy for language '{lang}' — skipping[/yellow]")
        return {
            "passed": True, "results": [], "build_output": "",
            "server_startup_log": "",
            "error": f"Integration tests skipped — no strategy for language '{lang}'",
        }

    if not build_result["ok"]:
        _console.print("[red bold]❌ Build FAILED[/red bold]")
        return {
            "passed": False, "results": [],
            "build_output": build_result["output"],
            "server_startup_log": "",
            "error": "BUILD FAILED",
        }
    _console.print("[green]✔ Build succeeded[/green]")

    # ── Start server ───────────────────────────────────────────────────────
    _console.print(f"[cyan]🚀 Starting {lang} server on port {port}…[/cyan]")
    starters = {
        "java":   _start_java_server,
        "nodejs": _start_nodejs_server,
        "go":     _start_go_server,
        "python": _start_python_server,
    }
    proc = starters[lang](project_root, port)
    if proc is None:
        _console.print(f"[red]❌ Could not start server for language '{lang}'[/red]")
        return {
            "passed": False, "results": [],
            "build_output": build_result["output"],
            "server_startup_log": "",
            "error": f"Could not start server for language '{lang}'",
        }

    # ── Wait for health ────────────────────────────────────────────────────
    ready = False
    startup_lines: list[str] = []
    _console.print(f"[dim]⏳ Waiting for server on port {port} (up to {MAX_STARTUP_SECS}s)…[/dim]")
    for i in range(int(MAX_STARTUP_SECS / HEALTH_POLL_SECS)):
        time.sleep(HEALTH_POLL_SECS)
        # Drain any available server output and print live
        if proc.stdout:
            try:
                line = proc.stdout.readline()
                if line:
                    stripped = line.rstrip()
                    startup_lines.append(stripped)
                    _console.print(f"[dim]  server │ {stripped}[/dim]")
            except (IOError, OSError):
                pass

        if proc.poll() is not None:
            _console.print("[red]  server │ process exited early — draining output…[/red]")
            # Server exited early — drain remaining output
            if proc.stdout:
                try:
                    for line in proc.stdout.readlines():
                        stripped = line.rstrip()
                        startup_lines.append(stripped)
                        _console.print(f"[dim]  server │ {stripped}[/dim]")
                except (IOError, OSError):
                    pass
            break

        elapsed_s = (i + 1) * HEALTH_POLL_SECS
        if elapsed_s % 5 == 0:
            _console.print(f"[dim]  health-check… {elapsed_s}s elapsed[/dim]")

        if _poll_health(port):
            _console.print(f"[green]  ✔ Server is healthy on port {port} after {elapsed_s}s[/green]")
            ready = True
            break

    server_startup_log = "\n".join(startup_lines)

    if not ready:
        _terminate_server(proc)
        return {
            "passed": False, "results": [],
            "build_output": build_result["output"],
            "server_startup_log": server_startup_log,
            "error": (
                f"Server did not become healthy within {MAX_STARTUP_SECS}s on port {port}. "
                "Check server_startup_log for details."
            ),
        }

    # ── Run curl tests ─────────────────────────────────────────────────────
    base_url = f"http://localhost:{port}"
    tests = _parse_contracts(plan_items)
    if not tests:
        tests = [{"method": "GET", "path": "/actuator/health",
                  "expected_status": 200, "expected_fields": []}]

    _console.print(f"[cyan]🧪 Running {len(tests)} endpoint test(s)…[/cyan]")
    created_ids: dict[str, str] = {}

    # ── Run curl tests (OPTIMIZED: parallel with dependency handling) ─────
    results = _run_tests_parallel(base_url, tests, created_ids)

    # ── Teardown ───────────────────────────────────────────────────────────
    _terminate_server(proc)

    all_passed = all(t["passed"] for t in results)
    return {
        "passed":            all_passed,
        "results":           results,
        "build_output":      build_result["output"],
        "server_startup_log": server_startup_log,
        "error":             None,
    }
