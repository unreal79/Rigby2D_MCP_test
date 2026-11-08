"""Raise and lower a Rigby2D character's arms over the Model Context Protocol.

Reads a project, asks the MCP server to add an ``arms_up`` animation, and writes the
result to a second file: the input is never modified. Every run writes its own log to
``logs/``.

The MCP plumbing lives in ``rigby2d_mcp_utils``; this file only holds the arm animation.

Usage:
    python add_arms_up.py [input] [output]

    python add_arms_up.py                                # input/test.rigby -> output/test.rigby
    python add_arms_up.py input/hero.rigby               # -> output/hero.rigby
    python add_arms_up.py input/hero.rigby out/hero.rigby

Paths are relative to the repository root (the MCP workspace). The output folder is
created when missing. Requirements: Python 3.8+ and Node.js 20 or newer.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import rigby2d_mcp_utils as mcp

# ------------------------------- settings -----------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR = mcp.LOG_DIR

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

# Bones and the times at which the hands are reported, for the verification step.
ARMS = (("upperarm_l", LEFT_UP), ("upperarm_r", RIGHT_UP))
HANDS = ("hand_l", "hand_r")


# ------------------------------ animation -----------------------------------


def arms_up_operations() -> list[dict]:
    """The ``batch_edit`` operations that build the animation, one atomic batch."""
    operations: list[dict] = [
        {"op": "add_animation", "name": ANIMATION, "duration": DURATION}
    ]
    for bone, up_value in ARMS:
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
    return operations


def hands_text(pose: dict) -> str:
    """World-space hand positions from a ``sample_pose`` result, formatted."""
    parts = []
    for name in HANDS:
        bone = mcp.pose_bone(pose, name)
        if bone:
            parts.append(f"{name}=({bone['x']:.1f}, {bone['y']:.1f})")
    return ", ".join(parts)


def add_arms_up(client: mcp.RigbyMcpClient, source: str, output: str, log: mcp.RunLog) -> bool:
    """Create the animation in ``source`` and save the project to ``output``."""
    existing = [anim.get("name") for anim in
                client.call_tool("inspect_project", {"path": source}).get("animations", [])]
    log.info(f"Animations in the project: {existing}")

    # A previous run may already have created the animation: drop it first so that
    # running the script twice gives the same result.
    if ANIMATION in existing:
        client.call_tool("delete_animation", {
            "path": source, "output": output, "overwrite": True, "animation": ANIMATION,
        })
        source = output
        log.info(f"Deleted the previous '{ANIMATION}' animation")

    operations = arms_up_operations()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": operations,
    })
    log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s, {len(operations) - 1} keys)")

    # Verify: key times, world-space hand positions, and project health.
    anim = client.call_tool("describe_animation", {"path": output, "animation": ANIMATION})
    log.info(f"Duration after the edit: {anim.get('duration'):g} s")
    for timeline in anim.get("bones", []):
        rotate = timeline.get("keys", {}).get("rotate", [])
        log.info(f"  {timeline['bone']}: rotate keys={rotate}")

    log.info("Hand positions in world space (Y up):")
    for time, title in ((0.0, "arms down"), (UP_TIME, "arms up"),
                        (DOWN_TIME, "arms down again")):
        pose = client.call_tool(
            "sample_pose", {"path": output, "animation": ANIMATION, "time": time}
        )
        log.info(f"  t={time:g} s, {title}: {hands_text(pose)}")

    report = client.call_tool("validate_project", {"path": output})
    errors = report.get("errors") or []
    log.info(f"validate_project: {'OK' if not errors else 'FAILED'}")
    for message in errors:
        log.error(f"  {message}")
    for message in report.get("warnings") or []:
        log.detail(f"  warning: {message}")
    return not errors


# ----------------------------------- run ------------------------------------


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: arms up ===")
    log.info(f"Requested input:  {args.input}")
    log.info(f"Requested output: {args.output or '(default)'}")
    log.info(f"Log:              {log.path}")

    input_path, output_path = mcp.project_paths(WORKSPACE, args.input, args.output)
    source = mcp.server_path(WORKSPACE, input_path)
    target = mcp.server_path(WORKSPACE, output_path)

    log.info(f"Input:  {input_path}")
    log.info(f"Output: {output_path}")
    if output_path.exists():
        log.info("The output file exists and will be replaced")

    # The MCP server writes only into folders that already exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mcp.connect(log, WORKSPACE, client_name="rigby-arms-demo", client_version="2.0") as client:
        ok = add_arms_up(client, source, target, log)

    if not ok:
        log.error(f"The output project has problems; see {log.path}")
        return 1
    log.info(f"Done. Output written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add an animation that raises a Rigby2D character's arms and lowers them.",
    )
    mcp.add_project_arguments(parser)
    args = parser.parse_args(argv)

    try:
        log = mcp.RunLog(LOG_DIR)
    except OSError as exc:
        print(f"Cannot create the log file in {LOG_DIR}: {exc}", file=sys.stderr)
        return 1

    try:
        return run(log, args)
    except mcp.SetupError as exc:
        log.error(f"Setup error: {exc}")
        return 1
    except mcp.McpError as exc:
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
