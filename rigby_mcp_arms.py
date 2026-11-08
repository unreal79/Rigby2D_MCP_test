"""Minimal MCP client and demo for the Rigby2D MCP server.

Pipeline: read a project from ``input/``, let the MCP server edit it, and write the
result to ``output/``. The input file is never modified. Every run writes its own
log file to ``logs/``.

The edit: a new animation in which the character raises its arms and lowers them again.

Usage:
    python rigby_mcp_arms.py

Requirements: Python 3.8+ standard library only, plus Node.js 20 or newer
(the script looks for node on PATH and in the usual install folders).

Environment overrides:
    NODE_EXE           path to the node executable
    RIGBY_MCP_SERVER   path to the MCP server entry point (server.js)
    RIGBY_WORKSPACE    folder the server may access (default: repository root)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

# ------------------------------- settings -----------------------------------

# The script lives in the repository root, which is also the MCP workspace root.
WORKSPACE = Path(os.environ.get("RIGBY_WORKSPACE") or Path(__file__).resolve().parent)
INPUT_PROJECT = "input/test.rigby"       # relative to WORKSPACE; never modified
OUTPUT_PROJECT = "output/test.rigby"     # relative to WORKSPACE; written by the script
LOG_DIR = WORKSPACE / "logs"

ANIMATION = "arms_up"           # name of the animation to create
DURATION = 1.6                  # animation length, seconds
UP_TIME = 0.8                   # time at which the arms are up
DOWN_TIME = 1.6                 # time at which the arms are down again

# Key values are relative to the setup pose and expressed in degrees.
# +130 degrees turns the left arm from pointing down to pointing up;
# -130 degrees does the same for the mirrored right arm. The arms raise into a
# "V" shape instead of crossing the face.
LEFT_UP = 130.0
RIGHT_UP = -130.0
CURVE = "easeInOut"

# ------------------------------ logging -------------------------------------


class RunLog:
    """One log file per run: writes to ``logs/run-<timestamp>.log`` and to stdout."""

    def __init__(self, folder: Path, prefix: str = "run") -> None:
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


# ------------------------------ Node.js lookup -------------------------------


class SetupError(RuntimeError):
    """A configuration problem that stops the run before any MCP call."""


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


def find_server() -> Path:
    """Locate the MCP server entry point inside the repository."""
    override = os.environ.get("RIGBY_MCP_SERVER")
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise SetupError(f"RIGBY_MCP_SERVER points at a missing file: {path}")

    for relative in (
        "_tools/mcp-server/src/mcp/server.js",   # this repository's layout
        "server/src/mcp/server.js",              # unpacked .mcpb extension
        "src/mcp/server.js",                     # loose source tree
        "rigby2d-develop/src/mcp/server.js",     # upstream source checkout
    ):
        path = WORKSPACE / relative
        if path.exists():
            return path

    raise SetupError(
        f"MCP server (src/mcp/server.js) was not found under {WORKSPACE}. "
        "Keep _tools/mcp-server/ in the repository, or set RIGBY_MCP_SERVER."
    )


# -------------------------------- MCP client --------------------------------


class McpError(RuntimeError):
    """An error returned by the MCP server or by the transport."""


class RigbyMcpClient:
    """Small MCP client: initialize, tools/call, close."""

    def __init__(self, node: str, server: Path, root: Path, log: RunLog) -> None:
        self._next_id = 0
        self._lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._log = log
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

    def request(self, method: str, params: dict | None = None) -> dict:
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

    def notify(self, method: str, params: dict | None = None) -> None:
        message: dict[str, object] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            self._send(message)

    # -- MCP ---------------------------------------------------------------

    def initialize(self) -> dict:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "rigby-arms-demo", "version": "1.1"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def call_tool(self, name: str, arguments: dict) -> dict:
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


# -------------------------------- helpers -----------------------------------


def bones_by_name(pose: dict) -> dict[str, dict]:
    bones = pose.get("bones") or []
    if isinstance(bones, dict):
        return bones
    return {bone["name"]: bone for bone in bones if "name" in bone}


def hands_text(pose: dict) -> str:
    bones = bones_by_name(pose)
    parts = []
    for name in ("hand_l", "hand_r"):
        bone = bones.get(name)
        if bone:
            parts.append(f"{name}=({bone['x']:.1f}, {bone['y']:.1f})")
    return ", ".join(parts)


# --------------------------------- scenario ---------------------------------


def run(log: RunLog) -> int:
    node = find_node()
    server = find_server()
    input_path = (WORKSPACE / INPUT_PROJECT).resolve()
    output_path = (WORKSPACE / OUTPUT_PROJECT).resolve()

    log.info("=== Rigby2D MCP demo ===")
    log.info(f"Node.js:    {node}")
    log.info(f"MCP server: {server}")
    log.info(f"Workspace:  {WORKSPACE}")
    log.info(f"Input:      {input_path}")
    log.info(f"Output:     {output_path}")
    log.info(f"Log:        {log.path}")

    if not input_path.exists():
        log.error(f"Input project not found: {input_path}")
        return 1

    # The MCP server writes only into folders that already exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with RigbyMcpClient(node, server, WORKSPACE, log) as client:
        info = client.initialize()
        server_info = info.get("serverInfo", {})
        log.info(f"Connected to {server_info.get('name')} {server_info.get('version', '')} "
                 f"(protocol {info.get('protocolVersion')})")

        # 1. See what the input already contains.
        project = client.call_tool("inspect_project", {"path": INPUT_PROJECT})
        existing = [anim.get("name") for anim in project.get("animations", [])]
        log.info(f"Animations in the input: {existing}")

        # 2. Remove a previous run's animation so the script is idempotent.
        source = INPUT_PROJECT
        if ANIMATION in existing:
            client.call_tool("delete_animation", {
                "path": INPUT_PROJECT,
                "output": OUTPUT_PROJECT,
                "overwrite": True,
                "animation": ANIMATION,
            })
            source = OUTPUT_PROJECT
            log.info(f"Deleted the previous '{ANIMATION}' animation")

        # 3. Create the animation and all of its keys in one atomic batch.
        operations = [{"op": "add_animation", "name": ANIMATION, "duration": DURATION}]
        for bone, up_value in (("upperarm_l", LEFT_UP), ("upperarm_r", RIGHT_UP)):
            operations += [
                {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": "rotate", "time": 0.0, "value": 0.0, "curve": CURVE,
                },
                {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": "rotate", "time": UP_TIME, "value": up_value, "curve": CURVE,
                },
                {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": "rotate", "time": DOWN_TIME, "value": 0.0, "curve": "linear",
                },
            ]
        client.call_tool("batch_edit", {
            "path": source,
            "output": OUTPUT_PROJECT,
            "overwrite": True,
            "operations": operations,
        })
        log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s, {len(operations) - 1} keys)")

        # 4. Verify the result: keys, world-space hand positions, project health.
        anim = client.call_tool(
            "describe_animation", {"path": OUTPUT_PROJECT, "animation": ANIMATION}
        )
        log.info(f"Duration after the edit: {anim.get('duration'):g} s")
        for timeline in anim.get("bones", []):
            rotate = timeline.get("keys", {}).get("rotate", [])
            log.info(f"  {timeline['bone']}: rotate keys={rotate}")

        log.info("Hand positions in world space (Y up):")
        for time, title in ((0.0, "arms down"), (UP_TIME, "arms up"),
                            (DOWN_TIME, "arms down again")):
            pose = client.call_tool(
                "sample_pose",
                {"path": OUTPUT_PROJECT, "animation": ANIMATION, "time": time},
            )
            log.info(f"  t={time:g} s, {title}: {hands_text(pose)}")

        report = client.call_tool("validate_project", {"path": OUTPUT_PROJECT})
        errors = report.get("errors") or []
        warnings = report.get("warnings") or []
        log.info(f"validate_project: {'OK' if not errors else 'FAILED'}")
        for message in errors:
            log.error(f"  {message}")
        for message in warnings:
            log.detail(f"  warning: {message}")
        if errors:
            return 1

    log.info(f"Done. Output written to {output_path}")
    return 0


def main() -> int:
    try:
        log = RunLog(LOG_DIR)
    except OSError as exc:
        print(f"Cannot create the log file in {LOG_DIR}: {exc}", file=sys.stderr)
        return 1

    try:
        return run(log)
    except SetupError as exc:
        log.error(f"Setup error: {exc}")
        return 1
    except McpError as exc:
        log.error(f"MCP error: {exc}")
        return 1
    except OSError as exc:
        log.error(f"I/O error: {exc}")
        log.detail(traceback.format_exc())
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    except Exception:
        log.error("Unexpected error:\n" + traceback.format_exc())
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
