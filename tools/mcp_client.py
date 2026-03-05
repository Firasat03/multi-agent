"""tools/mcp_client.py — unchanged from original"""
from __future__ import annotations
import json
from config import MCP_CONFIG_FILE

def _load_agent_config() -> dict:
    if not MCP_CONFIG_FILE.exists():
        return {}
    with open(MCP_CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)

class MCPClient:
    def __init__(self, agent_name: str, allowed_servers: list[str]) -> None:
        self.agent_name = agent_name
        self.allowed_servers = allowed_servers

    def call(self, server: str, tool: str, **kwargs) -> dict:
        if server not in self.allowed_servers:
            raise PermissionError(f"Agent '{self.agent_name}' cannot use '{server}'")
        return _get_handler(server)(tool, **kwargs)

    def list_allowed_servers(self) -> list[str]:
        return list(self.allowed_servers)

def get_client(agent_name: str) -> MCPClient:
    config = _load_agent_config()
    servers: list[str] = config.get(agent_name, {}).get("servers", [])
    return MCPClient(agent_name=agent_name, allowed_servers=servers)

def _get_handler(server: str):
    handlers = {
        "filesystem":     _handle_filesystem,
        "knowledge-base": _handle_knowledge_base,
        "postgres":       _handle_postgres,
        "github":         _handle_github,
        "sonarqube":      _handle_sonarqube,
        "swagger":        _handle_swagger,
        "shell":          _handle_shell,
    }
    if server not in handlers:
        raise ValueError(f"Unknown MCP server: '{server}'")
    return handlers[server]

def _handle_filesystem(tool: str, **kwargs) -> dict:
    from tools.file_tools import read_file, write_file, list_files, file_tree
    if tool == "read_file":   return {"content": read_file(kwargs["path"])}
    if tool == "write_file":  write_file(kwargs["path"], kwargs["content"]); return {"success": True}
    if tool == "list_files":  return {"files": list_files(kwargs["root"], kwargs.get("extensions"))}
    if tool == "file_tree":   return {"tree": file_tree(kwargs["root"])}
    raise ValueError(f"Unknown filesystem tool: {tool}")

def _handle_knowledge_base(tool: str, **kwargs) -> dict:
    if tool == "query": return {"results": [], "note": "Knowledge base not configured."}
    raise ValueError(f"Unknown knowledge-base tool: {tool}")

def _handle_postgres(tool: str, **kwargs) -> dict:
    if tool == "get_schema": return {"schema": {}, "note": "Postgres MCP not configured."}
    raise ValueError(f"Unknown postgres tool: {tool}")

def _handle_github(tool: str, **kwargs) -> dict:
    if tool in ("create_pr", "commit"): return {"success": False, "note": "GitHub MCP not configured."}
    raise ValueError(f"Unknown github tool: {tool}")

def _handle_sonarqube(tool: str, **kwargs) -> dict:
    if tool == "analyze": return {"issues": [], "note": "SonarQube MCP not configured."}
    raise ValueError(f"Unknown sonarqube tool: {tool}")

def _handle_swagger(tool: str, **kwargs) -> dict:
    if tool == "validate": return {"valid": True, "note": "Swagger MCP not configured."}
    raise ValueError(f"Unknown swagger tool: {tool}")

def _handle_shell(tool: str, **kwargs) -> dict:
    from tools.shell_tools import run_command
    if tool == "run": return run_command(kwargs["cmd"], cwd=kwargs.get("cwd"))
    raise ValueError(f"Unknown shell tool: {tool}")
