"""Reusable client utilities for the Rigby2D MCP server.

The MCP server speaks JSON-RPC 2.0 over stdio: one message per line on stdin/stdout,
logs on stderr. This module wraps that transport in a small client, locates Node.js and
the server, and provides the per-run logging and path helpers used by the demo scripts.

Standard library only (Python 3.8+), no third-party packages.

Environment overrides:
    NODE_EXE           path to the node executable
    RIGBY_MCP_SERVER   path to the MCP server entry point (server.js)
    RIGBY_WORKSPACE    folder the server may access (default: this file's folder)

Typical use:

    import rigby2d_mcp_utils as mcp

    with mcp.RunLog(mcp.LOG_DIR) as log:
        with mcp.connect(log) as client:
            project = client.call_tool("inspect_project", {"path": "input/test.rigby"})
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

# --------------------------------- settings ---------------------------------

# The repository root: the folder that holds this module (or RIGBY_WORKSPACE).
WORKSPACE = Path(
    os.environ.get("RIGBY_WORKSPACE") or Path(__file__).resolve().parent
).resolve()
LOG_DIR = WORKSPACE / "logs"

# Default project locations for the demo scripts.
DEFAULT_INPUT = "input/test.rigby"
DEFAULT_OUTPUT_DIR = "output"

# Server entry points, in order of preference.
SERVER_CANDIDATES = (
    "_tools/mcp-server/src/mcp/server.js",   # this repository's layout
    "server/src/mcp/server.js",              # unpacked .mcpb extension
    "src/mcp/server.js",                     # loose source tree
    "rigby2d-develop/src/mcp/server.js",     # upstream source checkout
)

PROTOCOL_VERSION = "2025-06-18"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# -------------------------------- errors ------------------------------------


class SetupError(RuntimeError):
    """A configuration problem that stops a run before any MCP call."""


class McpError(RuntimeError):
    """An error returned by the MCP server or by the transport."""


# -------------------------------- logging -----------------------------------


class RunLog:
    """One log file per run: writes to ``logs/run-<timestamp>.log`` and to stdout."""

    def __init__(self, folder: Path = LOG_DIR, prefix: str = "run") -> None:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = folder / f"{prefix}-{stamp}.log"
        counter = 1
        while path.exists():  # two runs within the same second
            path = folder / f"{prefix}-{stamp}-{counter}.log"
            counter += 1
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")

    def _write(self, level: str, message: str) -> None:
        for line in str(message).splitlines() or [""]:
            self._file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {level:<5} {line}\n")
        self._file.flush()

    def info(self, message: str) -> None:
        self._write("INFO", message)
        print(message)
        sys.stdout.flush()

    def detail(self, message: str) -> None:
        """Verbose material (tool arguments, results) that goes only to the log file."""
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)
        print(message, file=sys.stderr)
        sys.stderr.flush()

    def error(self, message: str) -> None:
        self._write("ERROR", message)
        print(message, file=sys.stderr)
        sys.stderr.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ------------------------------ discovery -----------------------------------


def find_node() -> str:
    """Return the path to node, or raise SetupError."""
    override = os.environ.get("NODE_EXE")
    if override and Path(override).exists():
        return override

    candidates = [shutil.which("node"), shutil.which("node.exe")]
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates += [
        str(Path(program_files) / "nodejs" / "node.exe"),
        str(Path(program_files) / "nodejs" / "node"),
        str(Path(local_app_data) / "Programs" / "nodejs" / "node.exe"),
        str(Path(local_app_data) / "Volta" / "bin" / "node.exe"),
        str(Path.home() / ".nvm" / "current" / "bin" / "node"),
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise SetupError(
        "Node.js was not found. Install it from https://nodejs.org or point the "
        "NODE_EXE environment variable at the node executable."
    )


def find_server(workspace: Optional[Path] = None) -> Path:
    """Locate the MCP server entry point (server.js)."""
    root = workspace or WORKSPACE

    override = os.environ.get("RIGBY_MCP_SERVER")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
        raise SetupError(f"RIGBY_MCP_SERVER points at a missing file: {path}")

    for relative in SERVER_CANDIDATES:
        path = root / relative
        if path.exists():
            return path.resolve()

    raise SetupError(
        f"MCP server (src/mcp/server.js) was not found under {root}. "
        "Keep _tools/mcp-server/ in the repository, or set RIGBY_MCP_SERVER."
    )


# ------------------------------ path helpers --------------------------------


def resolve_in_workspace(workspace: Path, value: str, what: str = "path") -> Path:
    """Resolve a user-supplied path against the workspace and check it stays inside."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()

    try:
        path.relative_to(workspace.resolve())
    except ValueError:
        raise SetupError(
            f"The {what} must be inside the workspace {workspace}: {path}"
        ) from None
    return path


def server_path(workspace: Path, path: Path) -> str:
    """The path as the MCP server wants it: relative to the workspace, forward slashes."""
    try:
        return path.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def add_project_arguments(
    parser: argparse.ArgumentParser,
    default_input: str = DEFAULT_INPUT,
    default_output_dir: str = DEFAULT_OUTPUT_DIR,
) -> argparse.ArgumentParser:
    """Add the ``input`` / ``output`` project arguments used by the demo scripts."""
    parser.add_argument(
        "input",
        nargs="?",
        default=default_input,
        help=f"project to read, relative to the repository root (default: {default_input})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "project to write, relative to the repository root "
            f"(default: {default_output_dir}/<input file name>)"
        ),
    )
    return parser


def project_paths(
    workspace: Path, input_value: str, output_value: Optional[str],
    default_output_dir: str = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Turn the CLI arguments into absolute input and output paths."""
    input_path = resolve_in_workspace(workspace, input_value, "input file")
    if not input_path.exists():
        raise SetupError(f"Input project not found: {input_path}")
    if input_path.suffix.lower() not in (".rigby", ".json", ".skel"):
        raise SetupError(
            f"Input must be a .rigby work file or a .json / .skel skeleton export: {input_path}"
        )

    if output_value:
        output_path = resolve_in_workspace(workspace, output_value, "output file")
    else:
        output_path = workspace / default_output_dir / input_path.name
    if output_path.suffix.lower() != input_path.suffix.lower():
        raise SetupError(
            f"The output must have the same extension as the input "
            f"({input_path.suffix}); use convert_project to change format."
        )
    if output_path == input_path:
        raise SetupError("The input and output must be different files; the input is read-only.")

    return input_path, output_path


# --------------------------------- MCP client -------------------------------


class RigbyMcpClient:
    """Small MCP client: initialize, tools/list, tools/call, close."""

    def __init__(
        self,
        node: str,
        server: Path,
        root: Path,
        log: Optional[RunLog] = None,
    ) -> None:
        self._next_id = 0
        self._lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._log = log
        self.server_info: dict = {}
        self.protocol_version: str = ""
        self.proc = subprocess.Popen(
            [node, str(server), "--root", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Drain stderr, otherwise the server can block once log output fills the pipe.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    # -- transport ---------------------------------------------------------

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr_lines.append(line.rstrip("\n"))
            del self._stderr_lines[:-200]  # keep only the tail of the log

    def _send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read_reply(self, request_id: int) -> dict:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = "\n".join(self._stderr_lines[-20:])
                raise McpError(f"The MCP server closed the stream. stderr:\n{stderr}")
            reply = json.loads(line)
            if reply.get("id") == request_id:
                return reply
            # Skip notifications and unrelated messages.

    def request(self, method: str, params: Optional[dict] = None) -> dict:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            message: dict[str, object] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._send(message)
            reply = self._read_reply(request_id)

        if "error" in reply:
            raise McpError(f"{method}: {reply['error']}")
        return reply["result"]

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        message: dict[str, object] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            self._send(message)

    # -- MCP ---------------------------------------------------------------

    def initialize(self, client_name: str = "rigby-mcp-client", client_version: str = "1.0") -> dict:
        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        )
        self.notify("notifications/initialized")
        self.server_info = result.get("serverInfo") or {}
        self.protocol_version = result.get("protocolVersion") or ""
        return result

    def list_tools(self) -> list[str]:
        """Names of the tools the server exposes."""
        result = self.request("tools/list", {})
        return [tool.get("name", "") for tool in result.get("tools", [])]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool and return its structured content (or the raw result as a fallback)."""
        if self._log:
            self._log.detail(f"tools/call {name} {json.dumps(arguments, ensure_ascii=False)}")
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            text = "\n".join(
                item.get("text", "")
                for item in result.get("content", [])
                if item.get("type") == "text"
            )
            raise McpError(f"{name}: {text or result}")
        data = result.get("structuredContent") or result
        if self._log:
            self._log.detail(f"  -> {json.dumps(data, ensure_ascii=False)[:2000]}")
        return data

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def __enter__(self) -> "RigbyMcpClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@contextmanager
def connect(
    log: Optional[RunLog] = None,
    workspace: Optional[Path] = None,
    client_name: str = "rigby-mcp-client",
    client_version: str = "1.0",
) -> Iterator[RigbyMcpClient]:
    """Find Node.js and the server, start it, complete the handshake, and clean up afterwards."""
    root = workspace or WORKSPACE
    node = find_node()
    server = find_server(root)

    if log:
        log.info(f"Node.js:    {node}")
        log.info(f"MCP server: {server}")
        log.info(f"Workspace:  {root}")

    client = RigbyMcpClient(node, server, root, log=log)
    try:
        client.initialize(client_name, client_version)
        if log:
            name = client.server_info.get("name", "unknown")
            version = client.server_info.get("version", "")
            log.info(f"Connected to {name} {version} (protocol {client.protocol_version})".rstrip())
        yield client
    finally:
        client.close()


# ---------------------------- result helpers --------------------------------


def bones_by_name(pose: dict) -> dict[str, dict]:
    """``sample_pose`` results: ``bones`` as a name -> bone dict mapping."""
    bones = pose.get("bones") or []
    if isinstance(bones, dict):
        return bones
    return {bone["name"]: bone for bone in bones if "name" in bone}


def pose_bone(pose: dict, name: str) -> Optional[dict]:
    """One bone's world transform from a ``sample_pose`` result, or None."""
    return bones_by_name(pose).get(name)


def tool_text(result: dict) -> str:
    """The human-readable text content of a raw ``tools/call`` result."""
    return "\n".join(
        item.get("text", "")
        for item in result.get("content", [])
        if item.get("type") == "text"
    )
