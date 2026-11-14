"""Give the cook a looping standing idle: he stays put and sways a little.

The motion is made **above the pelvis only**. The whole lower body -- ``root``,
``hip``, both thighs and shins and both foot IK targets -- receives no keys at all,
so the feet stay exactly where the artist drew them for the entire loop, with zero
drift. The life comes from the spine:

* the chest leans forward and back and shifts slightly, following a slow weight
  rock; a breath rides on top of it;
* the head counter-rotates so the gaze stays roughly level, and nods a little late;
* the hands hang from the animated shoulders as straight, slightly lagging
  pendulums, because both arms are fully extended in the setup pose.

Every channel is a sum of sine waves whose periods divide ``DURATION`` and the last
key (``t = DURATION``) equals the first key (``t = 0``), so the cycle loops without a
seam. Keys are sampled densely with linear curves; at 30 keys/s the curves are smooth
to the eye and the slope at the loop seam matches.

The arm IK constraints are kept at exactly full reach, so the solver never bends the
elbows away from the pose the artist drew; the arms only swing as a whole.

Usage:
    python add_cook_idle.py [input] [output]

    python add_cook_idle.py                              # input/cook.rigby -> output/cook.rigby
    python add_cook_idle.py input/cook.rigby out/idle.rigby
    python add_cook_idle.py --preview                    # also render a contact sheet to logs/preview/

Paths are relative to the repository root (the MCP workspace). The output folder is
created when missing. Requirements: Python 3.8+ and Node.js 20 or newer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rigby2d_mcp_utils as mcp

# ------------------------------- settings -----------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR = mcp.LOG_DIR

ANIMATION = "idle"              # name of the animation to create
DURATION = 2.0                  # seconds per loop: one weight rock and one breath
SAMPLE_HZ = 30.0                # keys per second: smooth, still small
VERIFY_HZ = 10.0                # how often the checks sample the finished animation
TAU = 2.0 * math.pi

# Bones that may be keyed. The lower body is deliberately absent: no keys means the
# legs cannot move, however the animation is sampled.
CHEST = "chest"
HEAD = "head"
IK_HAND_L = "ik_hand_l"
IK_HAND_R = "ik_hand_r"
STILL_BONES = ("root", "hip", "thigh_l", "shin_l", "ik_foot_l",
               "thigh_r", "shin_r", "ik_foot_r")

# ------------------------------- the sway -----------------------------------
# One forward/back weight rock per loop plus a second harmonic, and one slow breath
# underneath. Amplitudes are degrees and project units. The character is about 1500
# units tall, so 2 degrees of chest lean moves the head roughly 13 units -- a little
# less than one percent of the height, which is what a relaxed standing idle does.

LEAN_A = 1.8                    # deg, chest lean, one swing per cycle
LEAN_B = 0.45                   # deg, second harmonic (breaks the metronome feel)
LEAN_PHASE = -0.30              # rad, lag of the torso behind the weight rock
SHIFT_X = 2.6                   # units, forward/back drift of the chest
SHIFT_PHASE = 0.55              # rad, the shift arrives after the lean starts

BREATH_ROT = 0.30               # deg, the chest opens as it fills
BREATH_RISE = 1.6               # units, the shoulders rise as the chest fills
BREATH_PHASE = -1.05            # rad, breathing leads the lean

HEAD_NOD = 0.70                 # deg, the head nods with the breath
HEAD_PHASE = -1.35              # rad, the head arrives last
HEAD_COUNTER = 0.85             # how much of the torso lean the head cancels out

ARM_SWING_L = 1.5               # deg, the hanging arms swing as the body moves
ARM_SWING_R = 1.2
ARM_PHASE_L = -1.00             # rad, the hands arrive after the shoulders
ARM_PHASE_R = -1.20
ARM_EXTENSION = 1.0             # of full reach: keeps the drawn elbow untouched

# ------------------------------- geometry -----------------------------------


@dataclass(frozen=True)
class Rig:
    """Everything the pose function needs, measured from the setup pose."""

    hip_rot: float
    chest_rot: float
    chest_origin: tuple[float, float]
    shoulder_local_l: tuple[float, float]
    shoulder_local_r: tuple[float, float]
    hand_setup_l: tuple[float, float]
    hand_setup_r: tuple[float, float]
    arm_dir_l: float
    arm_dir_r: float
    arm_reach_l: float
    arm_reach_r: float


def frame_apply(frame: tuple[float, float, float],
                local_x: float, local_y: float) -> tuple[float, float]:
    """A point given in the frame's local space, in the parent's space."""
    angle = math.radians(frame[2])
    cos, sin = math.cos(angle), math.sin(angle)
    return (frame[0] + cos * local_x - sin * local_y,
            frame[1] + sin * local_x + cos * local_y)


def rotate(vector: tuple[float, float], deg: float) -> tuple[float, float]:
    """A vector turned by ``deg``."""
    angle = math.radians(deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return (cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1])


def measure_rig(setup: dict) -> Rig:
    """Read the setup pose once: frames, shoulders and the two arm directions."""
    bones = mcp.bones_by_name(setup)
    hip = bones["hip"]
    chest = bones[CHEST]
    chest_origin = (chest["x"], chest["y"])
    chest_rot = chest["rotation"]

    def shoulder_local(name: str) -> tuple[float, float]:
        bone = bones[name]
        return rotate((bone["x"] - chest_origin[0], bone["y"] - chest_origin[1]),
                      -chest_rot)

    def arm(prefix: str) -> tuple[tuple[float, float], float, float]:
        shoulder = bones[f"upperarm_{prefix}"]
        hand = bones[f"ik_hand_{prefix}"]
        dx, dy = hand["x"] - shoulder["x"], hand["y"] - shoulder["y"]
        return (hand["x"], hand["y"]), math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy)

    hand_l, dir_l, reach_l = arm("l")
    hand_r, dir_r, reach_r = arm("r")
    return Rig(
        hip_rot=hip["rotation"], chest_rot=chest_rot, chest_origin=chest_origin,
        shoulder_local_l=shoulder_local("upperarm_l"),
        shoulder_local_r=shoulder_local("upperarm_r"),
        hand_setup_l=hand_l, hand_setup_r=hand_r,
        arm_dir_l=dir_l, arm_dir_r=dir_r,
        arm_reach_l=reach_l, arm_reach_r=reach_r,
    )


def polar(origin: tuple[float, float], length: float,
          deg: float) -> tuple[float, float]:
    """The point ``length`` away from ``origin`` in direction ``deg``."""
    angle = math.radians(deg)
    return (origin[0] + length * math.cos(angle),
            origin[1] + length * math.sin(angle))


# -------------------------------- pose --------------------------------------


def pose_at(rig: Rig, time_value: float) -> dict:
    """The idle pose at a time, in root-local space (Y up, degrees)."""
    phase = (time_value / DURATION) % 1.0
    phi = TAU * phase

    # The torso: one lean per cycle plus a second harmonic, with the breath on top.
    lean = (LEAN_A * math.sin(phi + LEAN_PHASE)
            + LEAN_B * math.sin(2.0 * phi + 2.0 * LEAN_PHASE + 0.6))
    breath = math.sin(phi + BREATH_PHASE)
    chest_rot = lean + BREATH_ROT * breath

    # The chest offset is keyed in the hip's frame, so convert the world vector.
    chest_world = (SHIFT_X * math.sin(phi + SHIFT_PHASE), BREATH_RISE * breath)
    chest_local = rotate(chest_world, -rig.hip_rot)

    # Forward kinematics down the animated spine: the hands must hang from the
    # shoulders that actually exist at this moment.
    chest_frame = (rig.chest_origin[0] + chest_world[0],
                   rig.chest_origin[1] + chest_world[1],
                   rig.chest_rot + chest_rot)
    shoulder_l = frame_apply(chest_frame, *rig.shoulder_local_l)
    shoulder_r = frame_apply(chest_frame, *rig.shoulder_local_r)

    # The head keeps the gaze level (mostly) and nods with the breath, a little late.
    head_world = HEAD_NOD * math.sin(phi + HEAD_PHASE)
    head_rot = head_world - HEAD_COUNTER * chest_rot

    # The arms hang as rigid pendulums: the target sits at full reach, so the
    # solver does not bend the elbows away from the drawn pose.
    swing_l = ARM_SWING_L * math.sin(phi + ARM_PHASE_L)
    swing_r = ARM_SWING_R * math.sin(phi + ARM_PHASE_R)
    hand_l = polar(shoulder_l, ARM_EXTENSION * rig.arm_reach_l, rig.arm_dir_l + swing_l)
    hand_r = polar(shoulder_r, ARM_EXTENSION * rig.arm_reach_r, rig.arm_dir_r + swing_r)

    return {
        "phase": phase,
        "lean": lean,
        "chest_rot": chest_rot,
        "chest_world": chest_world,
        "chest_local": chest_local,
        "head_rot": head_rot,
        "shoulders": {"l": shoulder_l, "r": shoulder_r},
        "hands": {"l": hand_l, "r": hand_r},
        "swings": {"l": swing_l, "r": swing_r},
    }


def local_keys(rig: Rig, pose: dict) -> dict[str, dict[str, dict[str, float]]]:
    """The key values for one frame: offsets from the setup pose, degrees."""
    chest = pose["chest_local"]
    hands = pose["hands"]
    return {
        CHEST: {
            "rotate": {"value": pose["chest_rot"]},
            "translate": {"x": chest[0], "y": chest[1]},
        },
        HEAD: {"rotate": {"value": pose["head_rot"]}},
        IK_HAND_L: {"translate": {"x": hands["l"][0] - rig.hand_setup_l[0],
                                  "y": hands["l"][1] - rig.hand_setup_l[1]}},
        IK_HAND_R: {"translate": {"x": hands["r"][0] - rig.hand_setup_r[0],
                                  "y": hands["r"][1] - rig.hand_setup_r[1]}},
    }


# -------------------------------- batch -------------------------------------


def idle_operations(rig: Rig, delete_first: bool) -> list[dict]:
    """The ``batch_edit`` operations that build the whole animation."""
    operations: list[dict] = []
    if delete_first:
        operations.append({"op": "delete_animation", "animation": ANIMATION})
    operations.append({"op": "add_animation", "name": ANIMATION, "duration": DURATION})

    steps = int(round(DURATION * SAMPLE_HZ))
    for step in range(steps + 1):
        time_value = round(step / SAMPLE_HZ, 6)
        pose = pose_at(rig, min(time_value, DURATION))
        for bone, channels in local_keys(rig, pose).items():
            for channel, values in channels.items():
                operation = {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": channel, "time": time_value, "curve": "linear",
                }
                operation.update({name: round(value, 3) for name, value in values.items()})
                operations.append(operation)
    return operations


# ----------------------------- verification ---------------------------------


def bone_delta(first: dict, second: dict, name: str) -> float:
    """Largest positional/rotational difference of one bone between two poses."""
    a = mcp.pose_bone(first, name)
    b = mcp.pose_bone(second, name)
    if a is None or b is None:
        return 0.0
    return max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]),
               abs(a["rotation"] - b["rotation"]))


def hand_tip(pose: dict, side: str, segment: float) -> Optional[tuple[float, float]]:
    """World position of the tip of the forearm of one arm."""
    forearm = mcp.pose_bone(pose, f"forearm_{side}")
    if forearm is None:
        return None
    return polar((forearm["x"], forearm["y"]), segment, forearm["rotation"])


def travel_report(rig: Rig, log: mcp.RunLog) -> None:
    """Log how far each end of the body actually moves, and the timing."""
    samples = 600
    frames = [pose_at(rig, DURATION * i / samples) for i in range(samples)]

    leans = [frame["lean"] for frame in frames]
    log.info(f"Chest lean: {min(leans):+.2f} .. {max(leans):+.2f} deg "
             f"({max(leans) - min(leans):.2f} of travel)")

    shifts = [frame["chest_world"][0] for frame in frames]
    log.info(f"Chest drift: {max(shifts) - min(shifts):.1f} units forward/back")

    heads = [frame_apply((rig.chest_origin[0] + frame["chest_world"][0],
                          rig.chest_origin[1] + frame["chest_world"][1],
                          rig.chest_rot + frame["chest_rot"]),
                         *rig.shoulder_local_l) for frame in frames]
    head_x = [point[0] for point in heads]
    head_y = [point[1] for point in heads]
    log.info(f"Shoulder travel: {max(head_x) - min(head_x):.1f} x "
             f"{max(head_y) - min(head_y):.1f} units")

    hands = [frame["hands"]["l"] for frame in frames]
    hand_x = [point[0] for point in hands]
    hand_y = [point[1] for point in hands]
    log.info(f"Left hand travel: {max(hand_x) - min(hand_x):.1f} x "
             f"{max(hand_y) - min(hand_y):.1f} units")

    log.info(f"Timing: {DURATION:g} s per loop, {60.0 / DURATION:.0f} breaths/min, "
             f"{60.0 / DURATION:.0f} weight rock(s)/min")
    log.info("Feet: never keyed, so they stay exactly at the setup pose")


def smoothness_report(path: Path, log: mcp.RunLog) -> float:
    """Largest second difference of any curve and the velocity jump at the seam."""
    project = json.loads(zipfile.ZipFile(path).read("project.json"))
    animation = next((a for a in project["animations"] if a["name"] == ANIMATION), None)
    if animation is None:
        log.error(f"Animation '{ANIMATION}' is missing from {path}")
        return float("inf")

    worst = 0.0
    worst_name = ""
    changes: dict[str, float] = {}
    seam = 0.0
    seam_name = ""
    for timeline in animation.get("bones", []):
        name = project["bones"][timeline["bone"]]["name"]
        for channel, axis in (("rotate", None), ("translate", 0), ("translate", 1)):
            keys = timeline.get(channel) or []
            if len(keys) < 3:
                continue
            values = [key["value"] if axis is None else key["value"][axis] for key in keys]
            times = [key["time"] for key in keys]
            label = channel if axis is None else f"{channel}[{'xy'[axis]}]"
            for i in range(1, len(values) - 1):
                left = (values[i] - values[i - 1]) / (times[i] - times[i - 1])
                right = (values[i + 1] - values[i]) / (times[i + 1] - times[i])
                span = 0.5 * (times[i + 1] - times[i - 1])
                change = abs(right - left) / span if span else 0.0
                changes[f"{name}.{label}"] = max(changes.get(f"{name}.{label}", 0.0), change)
                if change > worst:
                    worst = change
                    worst_name = f"{name}.{label} at {times[i]:.2f} s"
            first_slope = (values[1] - values[0]) / (times[1] - times[0])
            last_slope = (values[-1] - values[-2]) / (times[-1] - times[-2])
            jump = abs(last_slope - first_slope)
            if jump > seam:
                seam = jump
                seam_name = f"{name}.{label}"

    log.info("Acceleration check (second difference of the saved curves):")
    for label in sorted(changes):
        log.info(f"  {label:22s} max {changes[label]:9.2f}")
    log.detail(f"  worst: {worst_name} {worst:.2f}")
    log.info(f"Loop seam: largest slope mismatch {seam:.4f} ({seam_name})")
    return max(worst, seam * 10.0)


def verify(client: mcp.RigbyMcpClient, rig: Rig, output: str, path: Path,
           log: mcp.RunLog) -> bool:
    """Check the loop, the planted feet, the arms, the curves and the project."""
    anim = client.call_tool("describe_animation", {"path": output, "animation": ANIMATION})
    log.info(f"Duration after the edit: {anim.get('duration'):g} s")
    for timeline in anim.get("bones", []):
        counts = {channel: len(timeline.get("keys", {}).get(channel) or [])
                  for channel in ("rotate", "translate")}
        log.info(f"  {timeline['bone']}: {counts['rotate']} rotate, "
                 f"{counts['translate']} translate key(s)")

    problems: list[str] = []

    # The whole point: nothing below the chest may move, and no foot may leave its spot.
    setup = client.call_tool("sample_pose", {"path": output})
    steps = int(round(DURATION * VERIFY_HZ))
    still_error = 0.0
    still_worst = ""
    for step in range(steps):
        time_value = round(step / VERIFY_HZ, 6)
        pose = client.call_tool(
            "sample_pose", {"path": output, "animation": ANIMATION, "time": time_value})
        for name in STILL_BONES:
            delta = bone_delta(setup, pose, name)
            if delta > still_error:
                still_error, still_worst = delta, f"{name} at {time_value:g} s"
    log.info(f"Lower body check: largest movement of any still bone {still_error:.4f} "
             f"({still_worst or 'none'})")
    if still_error > 0.01:
        problems.append(f"a leg or the pelvis moved by {still_error:.3f}")

    # Both arms have to reach their targets exactly; a shortfall means the arm was
    # asked to stretch and the drawn mesh will detach from the control.
    worst_arm = 0.0
    elbow_range = [180.0, 0.0]
    for step in range(steps):
        pose = client.call_tool("sample_pose", {
            "path": output, "animation": ANIMATION,
            "time": round(step / VERIFY_HZ, 6)})
        for side, segment in (("l", rig.arm_reach_l / 2.0),
                              ("r", rig.arm_reach_r / 2.0)):
            tip = hand_tip(pose, side, segment)
            goal = mcp.pose_bone(pose, f"ik_hand_{side}")
            if tip is not None and goal is not None:
                worst_arm = max(worst_arm,
                                math.hypot(tip[0] - goal["x"], tip[1] - goal["y"]))
            upper = mcp.pose_bone(pose, f"upperarm_{side}")
            fore = mcp.pose_bone(pose, f"forearm_{side}")
            if upper and fore:
                bend = abs(((fore["rotation"] - upper["rotation"] + 180.0) % 360.0) - 180.0)
                elbow_range = [min(elbow_range[0], bend), max(elbow_range[1], bend)]
    log.info(f"IK check: hands miss their target by at most {worst_arm:.4f}; "
             f"elbow bend {elbow_range[0]:.2f}..{elbow_range[1]:.2f} deg")
    if worst_arm > 0.05:
        problems.append(f"an arm is stretched {worst_arm:.2f} units past its reach")

    # Loop: the last frame has to be the first frame.
    first = client.call_tool(
        "sample_pose", {"path": output, "animation": ANIMATION, "time": 0.0})
    last = client.call_tool(
        "sample_pose", {"path": output, "animation": ANIMATION, "time": DURATION})
    loop_error = max((bone_delta(first, last, name)
                      for name in mcp.bones_by_name(first)), default=0.0)
    log.info(f"Loop check: start and end differ by {loop_error:.4f}")
    if loop_error > 0.01:
        problems.append(f"the first and last frame differ by {loop_error:.3f}")

    # The pose at the beats, so the log shows what the character actually does.
    log.info("Poses in world space (Y up, degrees):")
    for time_value in (0.0, 1.0, 2.0, 3.0, DURATION):
        pose = client.call_tool("sample_pose", {
            "path": output, "animation": ANIMATION, "time": time_value})
        line = []
        for name in (CHEST, HEAD, IK_HAND_L, IK_HAND_R):
            bone = mcp.pose_bone(pose, name)
            if bone is not None:
                line.append(f"{name}=({bone['x']:.1f},{bone['y']:.1f}) "
                            f"{bone['rotation']:.2f}")
        log.info(f"  t={time_value:.1f}s: " + " | ".join(line))

    travel_report(rig, log)
    smoothness_report(path, log)

    report = client.call_tool("validate_project", {"path": output})
    errors = report.get("errors") or []
    log.info(f"validate_project: {'OK' if not errors else 'FAILED'}")
    for message in errors:
        log.error(f"  {message}")
        problems.append(message)
    for message in report.get("warnings") or []:
        log.detail(f"  warning: {message}")
    return not problems


# ----------------------------------- run ------------------------------------


def add_idle(client: mcp.RigbyMcpClient, source: str, output: str,
             log: mcp.RunLog) -> bool:
    """Build the animation in ``source``, saving the project to ``output``."""
    info = client.call_tool("inspect_project", {"path": source})
    existing = [anim.get("name") for anim in info.get("animations", [])]
    log.info(f"Animations in the project: {existing}")
    delete_first = ANIMATION in existing
    if delete_first:
        log.info(f"'{ANIMATION}' already exists and will be rebuilt")

    setup = client.call_tool("sample_pose", {"path": source})
    rig = measure_rig(setup)
    log.info(f"Arm hang directions in the setup pose: left {rig.arm_dir_l:.2f} deg "
             f"(reach {rig.arm_reach_l:.2f}), right {rig.arm_dir_r:.2f} deg "
             f"(reach {rig.arm_reach_r:.2f})")
    log.info(f"Chest pivot at ({rig.chest_origin[0]:.2f}, {rig.chest_origin[1]:.2f}), "
             f"world rotation {rig.chest_rot:.2f} deg, hip rotation {rig.hip_rot:.2f} deg")

    operations = idle_operations(rig, delete_first)
    key_count = sum(1 for op in operations if op["op"] == "set_bone_key")
    log.info(f"Building {key_count} keys at {SAMPLE_HZ:g} keys/s...")
    started = time.perf_counter()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": operations,
    })
    log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s, "
             f"{len(operations)} operations) in {time.perf_counter() - started:.1f} s")

    return verify(client, rig, output, WORKSPACE / output, log)


def export_preview(client: mcp.RigbyMcpClient, output: str, log: mcp.RunLog) -> None:
    """Render the loop as a contact sheet, the quickest way to judge the motion."""
    folder = LOG_DIR / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    client.call_tool("export_sprite_sheet", {
        "input": output, "output_folder": mcp.server_path(WORKSPACE, folder),
        "name": ANIMATION, "format": "json", "layout": "grid",
        "animations": [ANIMATION], "fps": 4, "scale": 0.3,
        "trim": True, "dedupe": False, "loop": False, "overwrite": True,
    })
    log.info(f"Preview written to {folder}")


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: looping cook idle ===")
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

    with mcp.connect(log, WORKSPACE, client_name="rigby-cook-idle",
                     client_version="1.0") as client:
        ok = add_idle(client, source, target, log)
        if ok and args.preview:
            export_preview(client, target, log)

    if not ok:
        log.error(f"The output project has problems; see {log.path}")
        return 1
    log.info(f"Done. Output written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add a looping standing idle with a gentle sway to the cook.",
    )
    mcp.add_project_arguments(parser, default_input="input/cook.rigby")
    parser.add_argument("--preview", action="store_true",
                        help="also render a contact sheet of the loop to logs/preview/")
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
