"""
tools/file_tools.py — File system helpers for agents
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Characters that are illegal in filenames on Windows (and generally bad practice)
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def _validate_path(path: str, project_root: str | None = None) -> Path:
    """
    Resolve and validate a file path.

    Raises:
        ValueError: if the path contains illegal characters or escapes project_root.
    """
    p = Path(path)

    # Reject illegal filename characters in any component
    for part in p.parts:
        if _ILLEGAL_FILENAME_CHARS.search(part):
            raise ValueError(
                f"Path contains illegal characters: {path!r} (offending part: {part!r})"
            )

    # When project_root is provided, enforce containment to prevent path traversal
    if project_root:
        resolved = (Path(project_root) / p).resolve()
        root_resolved = Path(project_root).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {path!r} resolves outside "
                f"project root {project_root!r}"
            )

    return p


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str, project_root: str | None = None) -> None:
    """
    Write content to path, optionally validating it stays inside project_root.

    Args:
        path:         Relative or absolute path to the file.
        content:      Text content to write.
        project_root: If given, path must resolve inside this directory.
    """
    _validate_path(path, project_root=project_root)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def file_exists(path: str) -> bool:
    return Path(path).exists()


def list_files(root: str, extensions: list[str] | None = None) -> list[str]:
    skip_dirs = {".git", ".workflow", "__pycache__", "node_modules", ".venv", "venv"}
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if extensions is None or any(fname.endswith(ext) for ext in extensions):
                results.append(os.path.join(dirpath, fname))
    return sorted(results)


def file_tree(root: str, max_depth: int = 4) -> str:
    lines: list[str] = []

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        skip = {".git", ".workflow", "__pycache__", "node_modules", ".venv", "venv"}
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, entry in enumerate(entries):
            if entry.name in skip:
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(Path(root).name + "/")
    _walk(Path(root), "", 1)
    return "\n".join(lines)


def delete_file(path: str, project_root: str | None = None) -> None:
    """
    Delete a file, optionally validating it is inside project_root first.
    """
    _validate_path(path, project_root=project_root)
    p = Path(path)
    if p.exists():
        p.unlink()