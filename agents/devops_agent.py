"""
agents/devops_agent.py — DevOps Agent (OPT-IN)

Changes vs original:
  - Returns typed DevOpsOutput; state.apply(output) merges it
  - Warns explicitly when project_root is unset (files were previously silently discarded)
  - Uses shared _extract_files_from_response() with the correct multi-file parser
  - _detect_language deduplication: delegates to shell_tools.detect_language
"""

from __future__ import annotations

import os
import re

from agents.base_agent import BaseAgent
from config import Status
from state import DevOpsOutput, PipelineState
from tools.file_tools import write_file
from tools.shell_tools import detect_language


class DevOpsAgent(BaseAgent):
    name = "DevOps"
    system_role = (
        "You are a Senior DevOps Engineer. You produce production-ready infrastructure files.\n\n"
        "Docker:\n"
        "  - Multi-stage builds, non-root user, pinned base image versions\n"
        "  - COPY only needed files; .dockerignore-aware\n\n"
        "Kubernetes:\n"
        "  - Resource requests AND limits on every container\n"
        "  - Liveness and readiness probes\n"
        "  - ConfigMap for env config; HPA targeting 70% CPU\n"
        "  - SecurityContext: runAsNonRoot, readOnlyRootFilesystem\n"
        "  - 2 minimum replicas for HA\n\n"
        "docker-compose:\n"
        "  - Healthchecks, named volumes, .env file reference\n\n"
        "Output every file preceded by: # FILE: <path>\n"
        "Never output placeholder comments — every field must have a real value."
    )

    def run(self, state: PipelineState) -> PipelineState:
        state.status = Status.DEVOPS

        if not state.project_root:
            print(
                "[DevOps] Warning: project_root is not set. "
                "Generated DevOps files will be stored in state but NOT written to disk."
            )

        mode = state.devops_mode or "all"
        lang = detect_language(state.generated_files)
        devops_files: dict[str, str] = {}
        total_tokens = 0

        if mode in ("docker", "all"):
            docker_files, tokens = self._generate_docker(state, lang)
            devops_files.update(docker_files)
            total_tokens += tokens

        if mode in ("k8s", "all"):
            k8s_files, tokens = self._generate_k8s(state, lang)
            devops_files.update(k8s_files)
            total_tokens += tokens

        output = DevOpsOutput(devops_files=devops_files)
        state.apply(output)

        if state.project_root:
            self._flush_to_disk(state)

        state.log(
            self.name,
            tokens=total_tokens,
            notes=f"mode={mode}, lang={lang}, {len(devops_files)} file(s)",
        )
        return state

    def _generate_docker(self, state: PipelineState, lang: str) -> tuple[dict[str, str], int]:
        files_summary = _summarise_files(state.generated_files)
        prompt = f"""
Generate production-ready Docker infrastructure for a {lang} backend application.

TASK: {state.task_prompt}
FILES: {files_summary}
PLAN: {state.plan_summary}

Generate:
1. Dockerfile — multi-stage, non-root user, pinned base image
2. docker-compose.yml — healthchecks, named volumes, .env reference
3. .dockerignore — exclude non-essential files

For each file:
# FILE: <filename>
```<lang>
<complete content>
```
"""
        response_text, tokens = self._call_llm(state, prompt)
        files = self._extract_files_from_response(response_text, validate=False)
        return files, tokens

    def _generate_k8s(self, state: PipelineState, lang: str) -> tuple[dict[str, str], int]:
        files_summary = _summarise_files(state.generated_files)
        app_name = re.sub(r"[^a-z0-9]+", "-", state.task_prompt.lower())[:32].strip("-") or "backend-app"
        prompt = f"""
Generate a complete Kubernetes manifest set for a {lang} backend application.

APP NAME: {app_name}
TASK: {state.task_prompt}
FILES: {files_summary}

Generate under k8s/ directory:
1. k8s/namespace.yaml
2. k8s/configmap.yaml
3. k8s/deployment.yaml — 2 replicas, resource limits, liveness/readiness probes, securityContext
4. k8s/service.yaml — ClusterIP
5. k8s/ingress.yaml — NGINX with TLS placeholder
6. k8s/hpa.yaml — CPU 70%, min=2, max=10

# FILE: k8s/<filename>.yaml
```yaml
<complete content>
```
"""
        response_text, tokens = self._call_llm(state, prompt)
        files = self._extract_files_from_response(response_text, validate=False)
        return files, tokens

    def _flush_to_disk(self, state: PipelineState) -> None:
        root = state.project_root
        for rel_path, content in state.devops_files.items():
            try:
                write_file(os.path.join(root, rel_path), content, project_root=root)
            except ValueError as exc:
                print(f"[DevOps] Skipping unsafe path {rel_path!r}: {exc}")


def _summarise_files(generated_files: dict[str, str]) -> str:
    lines = []
    for path, content in generated_files.items():
        first_line = content.split("\n")[0][:80] if content else ""
        lines.append(f"  {path}  ({first_line}...)")
    return "\n".join(lines) or "(no files)"