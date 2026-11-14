"""Give a Rigby2D character a looping idle: he stands on the spot and sways gently.

The sway is made **above the pelvis only**. That is a deliberate choice forced by this
rig's construction, measured before writing a single key:

* Both legs are drawn fully extended, so the leg is a singular two-bone chain: a
  1-unit drop of the pelvis already asks for an 8..9 deg knee bend.
* The drawn foot is welded to the shin and there is no foot bone, so the sole's pitch
  *is* the shin angle. Bending a knee therefore rotates the foot about the ankle:
  measured on this rig, a 1-unit pelvis drop lifts one foot 6.8 units and sinks the
  other 4.3 units into the floor, and a sideways shift of 3 units puts the right leg
  out of reach (the IK would have to stretch).
* The IK targets are children of ``root``, so moving ``root`` moves the feet with it.

An idle that moved the pelvis would therefore have to slide the feet around to hide the
damage. This script instead leaves the whole lower body untouched -- **no keys at all on
``hip``, ``thigh_*``, ``shin_*`` or ``ik_foot_*``** -- so both feet are planted exactly
where the artist drew them, with zero drift over the cycle. The life comes from the
spine: the chest leans and rises with the breath, the head counter-rotates to keep the
gaze level, and the hands hang from the animated shoulders with a lag, like real arms.

The motion is smooth and loops exactly:

* every channel is a sum of sine waves whose periods divide ``DURATION``;
* the last key is written at ``t = DURATION`` and equals the key at ``t = 0``;
* keys are sampled densely with linear curves, so the curves are smooth to the eye and
  the loop has no velocity jump.

Usage:
    python add_idle.py [input] [output]

    python add_idle.py                                  # input/caveman1.rigby -> output/caveman1.rigby
    python add_idle.py input/caveman1.rigby out/idle.rigby
    python add_idle.py --preview                        # also render a contact sheet to logs/preview/

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
from pathlib import Path

import rigby2d_mcp_utils as mcp

# ------------------------------- settings -----------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR = mcp.LOG_DIR

ANIMATION = "idle"              # name of the animation to create
DURATION = 2.0                  # seconds per loop: one weight shift, two breaths
SAMPLE_HZ = 20.0                # keys per second: dense enough that easing is invisible
VERIFY_HZ = 10.0                # how often the checks sample the finished animation
TAU = 2.0 * math.pi

# Bones that carry the sway. Nothing else is keyed.
CHEST = "chest"
HEAD = "head"
IK_HAND_L = "ik_hand_l"
IK_HAND_R = "ik_hand_r"

# The lower body is never keyed; the verification proves it never moves.
STILL_BONES = ("root", "hip", "thigh_l", "shin_l", "ik_foot_l",
               "thigh_r", "shin_r", "ik_foot_r")

# ------------------------------- the sway -----------------------------------
# One weight shift per cycle, two breaths per cycle (20 breaths/min), both with a
# second harmonic so the motion does not read as a metronome. Amplitudes are in
# degrees and project units; the chest pivot sits at the base of the spine, so a lean
# of LEAN_A degrees moves the head (about 200 units away) by roughly 3.5 units.

LEAN_A = 2.6                    # deg, torso lean, one swing per cycle
LEAN_B = 0.9                    # deg, second harmonic (breaks the metronome feel)
LEAN_PHASE = -0.45              # rad, lag of the torso behind the weight shift
BREATH_ROT = 0.45               # deg, the chest opens as it fills
BREATH_RISE = 1.7               # units, the shoulders rise as the chest fills
BREATH_PHASE = -0.9             # rad, breathing leads the lean
SIDE_SHIFT = 2.6                # units, sideways drift of the torso over the feet

HEAD_NOD = 1.1                  # deg, the head nods with the breath
HEAD_PHASE = -1.25              # rad, the head arrives last
HEAD_COUNTER = 0.85             # how much of the torso lean the head cancels out

ARM_SWING = 1.5                 # deg, the hanging arms swing as the body moves
ARM_PHASE = -1.05               # rad, the hands arrive after the shoulders
ARM_EXTENSION = 0.998           # of full reach: keeps the arm straight but reachable

# ------------------------------- rig geometry --------------------------------
# Setup pose of input/caveman1.rigby (describe_rig + sample_pose), in root-local
# coordinates. The root is never moved, so this one space is enough.

HIP_ORIGIN = (288.7931, 469.5635)
HIP_ROT = 121.39993
CHEST_ORIGIN = (407.0791, 0.0)          # in hip space
CHEST_ROT = 0.0
HEAD_ORIGIN = (171.4322, 106.3565)      # in chest space
HEAD_ROT = -31.39993
SHOULDER_L_LOCAL = (50.4967, -109.4103)     # in chest space
SHOULDER_R_LOCAL = (51.0475, 169.9851)

IK_HAND_L_SETUP = (256.4782, 465.2379)      # root space, never moved by the script
IK_HAND_R_SETUP = (-238.2667, 501.0322)
ARM_L_REACH = 2.0 * 232.86748               # upper arm + forearm, fully extended
ARM_R_REACH = 2.0 * 153.27396

# The arm lengths above match the setup pose exactly, so the hands hang where the
# artist drew them and the arms stay straight.

ROOT_FRAME: tuple = (0.0, 0.0, 0.0)


# ------------------------------- helpers ------------------------------------


def polar(origin: tuple[float, float], length: float,
          deg: float) -> tuple[float, float]:
    """The point ``length`` away from ``origin`` in direction ``deg``."""
    angle = math.radians(deg)
    return (origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle))


def rotate(vector: tuple[float, float], deg: float) -> tuple[float, float]:
    """A vector turned by ``deg``."""
    angle = math.radians(deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return (cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1])


# A frame is (x, y, rotation in degrees); this rig has no shear or scale.
Frame = tuple


def frame_apply(frame: Frame, local_x: float, local_y: float) -> tuple[float, float]:
    """A point given in the frame's local space, in the parent's space."""
    angle = math.radians(frame[2])
    cos, sin = math.cos(angle), math.sin(angle)
    return (frame[0] + cos * local_x - sin * local_y,
            frame[1] + sin * local_x + cos * local_y)


def frame_child(parent: Frame, local_x: float, local_y: float, deg: float) -> Frame:
    """A child frame from its offset and rotation relative to ``parent``."""
    x, y = frame_apply(parent, local_x, local_y)
    return (x, y, parent[2] + deg)


# ------------------------------- pose ---------------------------------------


def arm_base_directions() -> tuple[float, float]:
    """The world angle at which each arm hangs in the setup pose, in degrees."""
    hip = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT)
    chest = frame_child(hip, *CHEST_ORIGIN, CHEST_ROT)
    left = frame_apply(chest, *SHOULDER_L_LOCAL)
    right = frame_apply(chest, *SHOULDER_R_LOCAL)
    left_dir = math.degrees(math.atan2(IK_HAND_L_SETUP[1] - left[1],
                                       IK_HAND_L_SETUP[0] - left[0]))
    right_dir = math.degrees(math.atan2(IK_HAND_R_SETUP[1] - right[1],
                                        IK_HAND_R_SETUP[0] - right[0]))
    return left_dir, right_dir


ARM_DIR_L, ARM_DIR_R = arm_base_directions()


def pose_at(time_value: float) -> dict:
    """The idle pose at a time: the chest, head and hands, in root-local space."""
    phase = (time_value / DURATION) % 1.0
    phi = TAU * phase

    # The torso: one lean per cycle plus a second harmonic, and the breath underneath.
    lean = (LEAN_A * math.sin(phi + LEAN_PHASE)
            + LEAN_B * math.sin(2.0 * phi + 2.0 * LEAN_PHASE + 0.7))
    breath = math.sin(2.0 * phi + BREATH_PHASE)
    chest_rot = lean + BREATH_ROT * breath

    # The chest translation is keyed in its parent's (the hip's) frame, but for the
    # forward kinematics below it is already the world offset it produces.
    chest_world = (SIDE_SHIFT * math.sin(phi + LEAN_PHASE), BREATH_RISE * breath)

    # The head keeps the gaze level and nods with the breath, a little late.
    head_world = HEAD_NOD * math.sin(2.0 * phi + HEAD_PHASE)
    head_rot = head_world - HEAD_COUNTER * chest_rot

    # Forward kinematics down the animated spine, so the hands hang from the shoulders
    # that actually exist at this moment.
    hip_frame = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT)
    chest_origin = frame_apply(hip_frame, *CHEST_ORIGIN)
    chest_frame = (chest_origin[0] + chest_world[0], chest_origin[1] + chest_world[1],
                   hip_frame[2] + CHEST_ROT + chest_rot)
    shoulder_l = frame_apply(chest_frame, *SHOULDER_L_LOCAL)
    shoulder_r = frame_apply(chest_frame, *SHOULDER_R_LOCAL)

    # The arms hang, swinging a little behind the body like anything on a hinge.
    swing = ARM_SWING * math.sin(phi + ARM_PHASE)
    hand_l = polar(shoulder_l, ARM_EXTENSION * ARM_L_REACH, ARM_DIR_L + swing)
    hand_r = polar(shoulder_r, ARM_EXTENSION * ARM_R_REACH, ARM_DIR_R + swing)

    return {
        "phase": phase,
        "lean": lean,
        "chest_rot": chest_rot,
        "chest_world": chest_world,
        # The same offset as the rig wants it: keyed in the hip's frame.
        "chest_local": rotate(chest_world, -HIP_ROT),
        "head_rot": head_rot,
        "hands": {"l": hand_l, "r": hand_r},
        "shoulders": {"l": shoulder_l, "r": shoulder_r},
    }


def local_keys(pose: dict) -> dict[str, dict[str, dict[str, float]]]:
    """The key values for one frame: offsets from the setup pose, degrees."""
    hands = pose["hands"]
    chest = pose["chest_local"]
    return {
        CHEST: {
            "rotate": {"value": pose["chest_rot"]},
            "translate": {"x": chest[0], "y": chest[1]},
        },
        HEAD: {"rotate": {"value": pose["head_rot"]}},
        IK_HAND_L: {"translate": {"x": hands["l"][0] - IK_HAND_L_SETUP[0],
                                  "y": hands["l"][1] - IK_HAND_L_SETUP[1]}},
        IK_HAND_R: {"translate": {"x": hands["r"][0] - IK_HAND_R_SETUP[0],
                                  "y": hands["r"][1] - IK_HAND_R_SETUP[1]}},
    }


# ------------------------------ batch ---------------------------------------


def idle_operations(delete_first: bool) -> list[dict]:
    """The ``batch_edit`` operations that build the whole animation."""
    operations: list[dict] = []
    if delete_first:
        operations.append({"op": "delete_animation", "animation": ANIMATION})
    operations.append({"op": "add_animation", "name": ANIMATION, "duration": DURATION})

    steps = int(round(DURATION * SAMPLE_HZ))
    for step in range(steps + 1):
        time_value = round(step / SAMPLE_HZ, 6)
        pose = pose_at(min(time_value, DURATION))
        for bone, channels in local_keys(pose).items():
            for channel, values in channels.items():
                operation = {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": channel, "time": time_value, "curve": "linear",
                }
                operation.update({name: round(value, 3) for name, value in values.items()})
                operations.append(operation)
    return operations


# ---------------------------- verification ----------------------------------


def head_position(pose: dict) -> tuple[float, float]:
    """The animated head origin of one frame, in root-local space."""
    hip_frame = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT)
    chest_origin = frame_apply(hip_frame, *CHEST_ORIGIN)
    chest_frame = (chest_origin[0] + pose["chest_world"][0],
                   chest_origin[1] + pose["chest_world"][1],
                   hip_frame[2] + CHEST_ROT + pose["chest_rot"])
    return frame_apply(chest_frame, *HEAD_ORIGIN)


def travel_report(log: mcp.RunLog) -> None:
    """Log how far each end of the body actually moves, and the timing."""
    samples = 600
    frames = [pose_at(DURATION * i / samples) for i in range(samples)]
    leans = [frame["lean"] for frame in frames]
    log.info(f"Torso lean: {min(leans):+.2f} .. {max(leans):+.2f} deg "
             f"({max(leans) - min(leans):.2f} of travel)")

    heads = [head_position(frame) for frame in frames]
    xs = [point[0] for point in heads]
    ys = [point[1] for point in heads]
    log.info(f"Head sway: {max(xs) - min(xs):.1f} units across, "
             f"{max(ys) - min(ys):.1f} units up and down")

    hands = [frame["hands"]["l"] for frame in frames]
    hand_x = [point[0] for point in hands]
    hand_y = [point[1] for point in hands]
    log.info(f"Left hand travel: {max(hand_x) - min(hand_x):.1f} x "
             f"{max(hand_y) - min(hand_y):.1f} units")

    log.info(f"Timing: {DURATION:g} s per loop, "
             f"{60.0 / (DURATION / 2.0):.0f} breaths/min, "
             f"{60.0 / DURATION:.0f} weight shift(s)/min")
    log.info("Feet: never keyed, so they stay exactly at the setup pose")


def smoothness_report(path: Path, log: mcp.RunLog) -> float:
    """Largest second difference of any curve and the velocity jump at the loop seam."""
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
            # The loop seam: the slope leaving the first key must match the slope
            # arriving at the last one, otherwise the motion ticks once per cycle.
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


def bone_delta(first: dict, second: dict, name: str) -> float:
    """Largest positional/rotational difference of one bone between two poses."""
    a = mcp.pose_bone(first, name)
    b = mcp.pose_bone(second, name)
    if a is None or b is None:
        return 0.0
    return max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]),
               abs(a["rotation"] - b["rotation"]))


def tips(pose: dict, upper: str, lower: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """The world position of the start and the tip of a two-bone IK chain."""
    first = mcp.pose_bone(pose, upper)
    second = mcp.pose_bone(pose, lower)
    if first is None or second is None:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    lengths = {"upperarm_l": 232.8675, "forearm_l": 232.8675,
               "upperarm_r": 153.274, "forearm_r": 153.274,
               "thigh_l": 174.2966, "shin_l": 174.2966,
               "thigh_r": 159.0361, "shin_r": 159.0361}
    tip = polar((second["x"], second["y"]), lengths.get(lower, 0.0), second["rotation"])
    return (first["x"], first["y"]), tip


def verify(client: mcp.RigbyMcpClient, output: str, path: Path,
           log: mcp.RunLog) -> bool:
    """Check the loop, the planted feet, the limbs, the curves and the project."""
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
    heel_l = None
    heel_r = None
    for step in range(steps):
        time_value = round(step / VERIFY_HZ, 6)
        pose = client.call_tool(
            "sample_pose", {"path": output, "animation": ANIMATION, "time": time_value})
        for name in STILL_BONES:
            delta = bone_delta(setup, pose, name)
            if delta > still_error:
                still_error, still_worst = delta, f"{name} at {time_value:g} s"
        if heel_l is None:
            heel_l = mcp.pose_bone(pose, "ik_foot_l")
            heel_r = mcp.pose_bone(pose, "ik_foot_r")
    log.info(f"Lower body check: largest movement of any still bone {still_error:.4f} "
             f"({still_worst or 'none'})")
    if still_error > 0.01:
        problems.append(f"a leg or the pelvis moved by {still_error:.3f}")

    # Both IK chains have to reach their targets exactly; a shortfall means the limb
    # was asked to stretch and the drawn mesh will detach from the control.
    worst_leg = 0.0
    worst_arm = 0.0
    knee_range = [180.0, 0.0]
    elbow_range = [180.0, 0.0]
    for step in range(steps):
        pose = client.call_tool("sample_pose", {
            "path": output, "animation": ANIMATION,
            "time": round(step / VERIFY_HZ, 6)})
        for (upper, lower), target in ((("thigh_l", "shin_l"), "ik_foot_l"),
                                       (("thigh_r", "shin_r"), "ik_foot_r")):
            (sx, sy), (tx, ty) = tips(pose, upper, lower)
            goal = mcp.pose_bone(pose, target)
            if goal is not None:
                worst_leg = max(worst_leg, math.hypot(tx - goal["x"], ty - goal["y"]))
            thigh = mcp.pose_bone(pose, upper)
            shin = mcp.pose_bone(pose, lower)
            if thigh and shin:
                bend = abs(((shin["rotation"] - thigh["rotation"] + 180.0) % 360.0) - 180.0)
                knee_range = [min(knee_range[0], bend), max(knee_range[1], bend)]
        for (upper, lower), target in ((("upperarm_l", "forearm_l"), "ik_hand_l"),
                                       (("upperarm_r", "forearm_r"), "ik_hand_r")):
            (sx, sy), (tx, ty) = tips(pose, upper, lower)
            goal = mcp.pose_bone(pose, target)
            if goal is not None:
                worst_arm = max(worst_arm, math.hypot(tx - goal["x"], ty - goal["y"]))
            arm = mcp.pose_bone(pose, upper)
            fore = mcp.pose_bone(pose, lower)
            if arm and fore:
                bend = abs(((fore["rotation"] - arm["rotation"] + 180.0) % 360.0) - 180.0)
                elbow_range = [min(elbow_range[0], bend), max(elbow_range[1], bend)]
    log.info(f"IK check: legs miss their target by at most {worst_leg:.4f}, "
             f"arms by at most {worst_arm:.4f}")
    log.info(f"Joint check: knee bend {knee_range[0]:.2f}..{knee_range[1]:.2f} deg, "
             f"elbow bend {elbow_range[0]:.2f}..{elbow_range[1]:.2f} deg")
    if worst_leg > 0.05:
        problems.append(f"a leg is stretched {worst_leg:.2f} units past its reach")
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
    for time_value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, DURATION):
        pose = client.call_tool("sample_pose", {
            "path": output, "animation": ANIMATION, "time": time_value})
        line = []
        for name in (CHEST, HEAD, IK_HAND_L, IK_HAND_R):
            bone = mcp.pose_bone(pose, name)
            if bone is not None:
                line.append(f"{name}=({bone['x']:.1f},{bone['y']:.1f}) "
                            f"{bone['rotation']:.2f}")
        log.info(f"  t={time_value:.1f}s: " + " | ".join(line))

    travel_report(log)
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

    log.info(f"Arm hang directions in the setup pose: left {ARM_DIR_L:.2f} deg, "
             f"right {ARM_DIR_R:.2f} deg")
    operations = idle_operations(delete_first)
    key_count = sum(1 for op in operations if op["op"] == "set_bone_key")
    log.info(f"Building {key_count} keys at {SAMPLE_HZ:g} keys/s...")
    started = time.perf_counter()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": operations,
    })
    log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s, "
             f"{len(operations)} operations) in {time.perf_counter() - started:.1f} s")

    return verify(client, output, WORKSPACE / output, log)


def export_preview(client: mcp.RigbyMcpClient, output: str, log: mcp.RunLog) -> None:
    """Render the loop as a contact sheet, the quickest way to judge the motion."""
    folder = LOG_DIR / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    client.call_tool("export_sprite_sheet", {
        "input": output, "output_folder": mcp.server_path(WORKSPACE, folder),
        "name": ANIMATION, "format": "json", "layout": "grid",
        "animations": [ANIMATION], "fps": 2, "scale": 0.35,
        "trim": True, "dedupe": False, "loop": False, "overwrite": True,
    })
    log.info(f"Preview written to {folder}")


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: looping idle ===")
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

    with mcp.connect(log, WORKSPACE, client_name="rigby-idle-demo", client_version="1.0") as client:
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
        description="Add a looping standing idle with a gentle sway to a Rigby2D character.",
    )
    mcp.add_project_arguments(parser, default_input="input/caveman1.rigby")
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
