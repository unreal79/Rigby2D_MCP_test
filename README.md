# Rigby2D MCP sandbox

A small, self-contained workspace for driving [Rigby2D](https://github.com/jaisurya-dev-art/rigby2d) from an AI assistant or from
your own code through the Rigby2D's [Model Context Protocol (MCP)](https://rigby2d.com/docs/mcp.html).

It contains a sample Rigby2D character project, a ready-to-run copy of the Rigby2D MCP
server (Node.js), and a Python demo client that edits the character's animation.

```
add_arms_up.py      ->  spawns MCP server (stdio)  ->  input/test.rigby -> output/test.rigby
rigby2d_mcp_utils.py       (shared client, logging, paths)   + logs/run-<timestamp>.log
```

---

## Contents

| Path | What it is |
| --- | --- |
| `add_arms_up.py` | Demo script. Reads the input project, adds an `arms_up` animation (arms rise, then lower) and writes the output project. |
| `rigby2d_mcp_utils.py` | Reusable MCP client: process launch, JSON-RPC transport, Node.js and server discovery, path validation, `RunLog`. Import it from any new script. |
| `run_add_animation.bat` | Convenience wrapper that sets the environment the script needs and runs it. |
| `input/test.rigby` | Local, git-ignored. Sample character work file ("NILA"): 39 bones, 42 slots, 1 skin, 10 constraints, animations `idle` and `arms_up`. The script never modifies it. |
| `output/` | Local, git-ignored. Generated: the edited project. Recreated on every run. |
| `logs/` | Local, git-ignored. Generated: one `run-<timestamp>.log` per run, with the steps, arguments and errors. |
| `_tools/mcp-server/` | The Rigby2D MCP server, unpacked from `Rigby2D-AI.mcpb`. Self-contained: own `src/` and `node_modules/`. |
| `_tools/start_mcpServer.bat` | Windows launcher for the server with `--root` set to this repository. |
| `_INFO/` | Local, git-ignored. Reference material, not used at runtime: upstream sources and docs (`rigby2d-develop/`), the packaged `.mcpb`, the extension manifest, third-party notices, the original project as received (`backups/`), and a legacy unpacked server copy (`reference-src/`). |

The repository tracks the code and the server: the project data (`input/`, `output/`, `logs/`)
and the reference material (`_INFO/`) stay on your machine.

---

## Requirements

- **Node.js 20 or newer** — runs the MCP server. (`node --version`)
- **Python 3.8 or newer** — the demo client uses the standard libraries only. No `pip install`.
- **Rigby2D** (optional) — only to watch the result in the editor.

The server reads and writes only inside the folders given with `--root`.

---

## Quick start

```powershell
# 1. Add the animation: reads input/test.rigby, writes output/test.rigby
python add_arms_up.py

# 2. Start Rigby2D (Rigby2D_0.5.0_x64_portable.exe, in the folder above),
#    open output\test.rigby and pick "arms_up" in the animation list
```

The script takes the input and output project as optional arguments; paths are relative
to the repository root and the output folder is created when missing:

```powershell
python add_arms_up.py                                      # default paths
python add_arms_up.py input/hero.rigby                     # -> output/hero.rigby
python add_arms_up.py input/hero.rigby build/hero.rigby    # explicit output
python add_arms_up.py --help
```

Expected output (world-space hand positions, Y up):

```
t=0 s, arms down:       hand_l=(541.0, 1409.2)
t=0.8 s, arms up:       hand_l=(297.2, 2400.5)
t=1.6 s, arms down again: hand_l=(541.0, 1409.2)
```

The script is idempotent: a previously created `arms_up` animation is deleted and rebuilt,
so running it twice gives the same output. The input file is only read.

`input/` must contain the project. `output/` and `logs/` are created when missing
(the MCP server itself only writes into folders that already exist).

### Logs

Every run creates a new file, `logs/run-YYYYmmdd-HHMMSS.log` (a `-1`, `-2`, ... suffix when
two runs start within the same second). It records the resolved paths, every `tools/call`
with its arguments and result, the verification output, and any error with its traceback.
The same lines go to the console, except the verbose tool traffic, which is log-only.

The script exits with `0` on success and `1` on failure.

### Calling the server directly

```powershell
node _tools\mcp-server\src\mcp\server.js --help
node _tools\mcp-server\src\mcp\server.js --root .        # talks MCP on stdin/stdout
```

`start_mcpServer.bat` does the same with the repository root as the allowed folder.
Never start the server through `npm run`: npm prints to stdout, which the protocol uses.

---

## Connecting an AI assistant

Point the client at the server entry point and pass this repository as the allowed root.
On Windows use full paths with doubled backslashes.

```json
{
  "mcpServers": {
    "rigby2d": {
      "command": "node",
      "args": [
        "<repo>\\_tools\\mcp-server\\src\\mcp\\server.js",
        "--root",
        "<repo>"
      ]
    }
  }
}
```

The server exposes 41 tools: reading (`inspect_project`, `describe_rig`,
`describe_animation`, `sample_pose`, `validate_project`), editing (`add_bone`,
`set_bone_key`, `batch_edit`, ...), conversion and sprite-sheet export, film/pose/audio
features, AutoRig and retargeting, and `run_script` (sandboxed CPython on Pyodide).

Full references: [Model Context Protocol (MCP)](https://rigby2d.com/docs/mcp.html).

---

## How the demo client works

The code is split in two files:

- **`rigby2d_mcp_utils.py`** — everything that is not specific to this animation:
  `RunLog`, `find_node`, `find_server`, path validation, `RigbyMcpClient`, and the
  `connect()` context manager. Reuse it from any new script.
- **`add_arms_up.py`** — the arm animation: `arms_up_operations()` and `add_arms_up()`.

The run:

1. **Transport** — `connect()` starts `node server.js --root <repo>` and speaks JSON-RPC 2.0
   over stdin/stdout, one message per line. `stderr` is drained on a background thread.
2. **Handshake** — `initialize` (protocol `2025-06-18`), then `notifications/initialized`.
3. **Read** — `inspect_project` to list the animations of the input.
4. **Write** — one atomic `batch_edit` with `add_animation` plus six `set_bone_key`
   operations, saved to the output with `overwrite: true`, so the input file is left
   untouched and the project is loaded and saved once.
5. **Verify** — `describe_animation` for the key times, `sample_pose` for the resulting
   hand positions, and `validate_project` for project health.

### Reusing the client

```python
import rigby2d_mcp_utils as mcp

with mcp.RunLog(mcp.LOG_DIR) as log:
    with mcp.connect(log) as client:
        info = client.call_tool("inspect_project", {"path": "input/test.rigby"})
        log.info(f"{info['name']}: {info['counts']['bones']} bones")
```

`RigbyMcpClient` exposes `initialize`, `list_tools`, `call_tool`, `request`, `notify`
and `close`, and works as a context manager.

Key values are **relative to the setup pose**, in degrees, with **Y up**. The arms are
keyed on `upperarm_l` (+130°) and `upperarm_r` (−130°). The hand IK constraints exist but
their `mix` is `0`, so the IK targets do not drive the arms; see
`.github/copilot-instructions.md` for details.

### Environment overrides

| Variable | Meaning |
| --- | --- |
| `NODE_EXE` | Path to the `node` executable (default: PATH and usual install folders). |
| `RIGBY_MCP_SERVER` | Path to the server entry point. |
| `RIGBY_WORKSPACE` | Root folder the server may access (default: repository root). |

---

## License and credits

Rigby2D and its MCP server are by the Rigby2D project; see
`_INFO/THIRD_PARTY_NOTICES.md` and `_INFO/rigby2d-develop/README.md`.
