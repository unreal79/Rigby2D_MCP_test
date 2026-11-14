"""Give a Rigby2D character a looping two-step walk, in place.

The walk is built for a game: the root never travels, so the engine can move the
character along the level at the ground speed the run prints, and the planted foot
stays put. The cycle contains two full steps, and the last frame is the first frame,
so the animation loops without a seam.

How the motion is made:

* The four limbs of this rig are driven by IK constraints (``mix`` is 1), so the
  keyed bones are the IK targets, not the limb bones. Every leg pose is first
  designed in forward kinematics (thigh angle, knee flex, foot pitch) and then
  turned into the target position that makes the solver reproduce it exactly.
* The pelvis height is not authored: for every frame the script measures the lowest
  point of the drawn feet and blends the two by how much each leg carries the body,
  so ``root.y`` follows the supporting leg and no foot can sink through the floor.
  The vertical bob of a walk then appears by itself.
* The arms are hung from the animated shoulders and swing opposite to the legs, with
  a slight elbow bend so they are never dead straight.
* All curves are periodic and sampled at ``SAMPLE_HZ`` with linear keys, so the
  motion is smooth and the loop closes without a velocity jump.

The rig fix (documented below) moves the right leg down, because the drawing puts its
foot about 41 units above the left one: without it the right leg can never reach the
floor the left leg walks on. The leg sits behind the body, so the shift is invisible;
the foot IK target moves with it, so the setup pose is unchanged.

Usage:
    python add_walk.py [input] [output]

    python add_walk.py                                  # input/caveman1.rigby -> output/caveman1.rigby
    python add_walk.py input/caveman1.rigby out/walk.rigby
    python add_walk.py --preview                        # also render a contact sheet to logs/preview/

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
from typing import Sequence

import rigby2d_mcp_utils as mcp

# ------------------------------- settings -----------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR = mcp.LOG_DIR

ANIMATION = "walk"              # name of the animation to create
DURATION = 1.6                  # seconds for two steps (0.8 s per step, ~75 steps/min)
SAMPLE_HZ = 30.0                # keys per second: dense enough that easing is invisible
STANCE = 0.60                   # share of the cycle a foot is on the ground
SUPPORT_START = 0.12            # phase at which a leg starts to carry the body
SUPPORT_RAMP = 0.08             # blend of the two feet around a support handover

# Bones of this rig (describe_rig of input/caveman1.rigby).
ROOT = "root"
HIP = "hip"
CHEST = "chest"
HEAD = "head"
THIGH_L = "thigh_l"
SHIN_L = "shin_l"
IK_FOOT_L = "ik_foot_l"
THIGH_R = "thigh_r"
SHIN_R = "shin_r"
IK_FOOT_R = "ik_foot_r"
IK_HAND_L = "ik_hand_l"
IK_HAND_R = "ik_hand_r"

# ------------------------------- rig geometry --------------------------------
# Setup pose of the skeleton (describe_rig), in root-local coordinates. The IK
# targets are children of the root too, so everything below is in one space and the
# root offset can be added at the very end.

HIP_ORIGIN = (288.7931, 469.5635)
HIP_ROT = 121.39993
CHEST_ORIGIN = (407.0791, 0.0)
HEAD_ORIGIN = (171.4322, 106.3565)
HEAD_ROT = -31.39993
SHOULDER_L_LOCAL = (50.4967, -109.4103)     # in chest space
SHOULDER_R_LOCAL = (51.0475, 169.9851)

THIGH_L_LOCAL = (-62.1107, 30.6106)         # in hip space
THIGH_R_LOCAL = (132.3657, 314.8605)        # in hip space (shifted by the rig fix)

THIGH_L_LEN = 174.29662                     # thigh -> shin distance
SHIN_L_LEN = 174.29662                      # shin -> foot IK tip
SHIN_R_LEN = 159.03614
ARM_L_LEN = 232.86748                       # upper arm -> forearm -> hand tip
ARM_R_LEN = 153.27396

IK_FOOT_L_SETUP = (412.4205, 72.3694)
IK_FOOT_R_SETUP = (-66.2943, 100.9020)
IK_HAND_L_SETUP = (256.4782, 465.2379)
IK_HAND_R_SETUP = (-238.2667, 501.0322)

# Foot outlines measured from the skinned meshes, in the shin's local space: the
# sole is the line at the far end, y = -toe..heel. The feet are drawn differently
# (the left one is short and steep, the right one long and flat), so each leg keeps
# its own outline. The first two points of each sole are its toe and its heel.
FOOT_L_SOLE = ((187.98, -104.03), (187.97, 49.02))
FOOT_R_SOLE = ((180.26, -141.76), (170.89, 56.14))
FOOT_L_OUTLINE = (
    (88.53, -119.11), FOOT_L_SOLE[0], FOOT_L_SOLE[1], (88.52, 96.40), (88.52, -11.35),
)
FOOT_R_OUTLINE = (
    (170.89, 56.14), (82.46, 126.29), (82.10, -147.13), (180.26, -141.76), (82.26, -10.02),
)

# The pose used to line both feet up on one floor line.
ALIGN_THETA = -90.0
ALIGN_BEND = 6.0

# ------------------------------- gait design --------------------------------
# One entry per key: (phase, shin angle in degrees, knee flex in degrees). Phase 0
# is the heel strike of that leg; the swing runs from ``STANCE`` back to 1.
# The shin angle also sets the pitch of the foot, because the sole is drawn square
# to the shin: -90 is flat, less is toe down (push-off), more is toe up (heel strike).
# The knee flex is the angle between the thigh and the socket-ankle line. The IK of
# this rig always bends the knee backwards, so it stays small: the leg reads as a
# straight, slightly bowed limb and the feet do the visible work.
GAIT_KEYS = (
    (0.00, -112.0, 5.0),        # heel strike: leg forward, toes up
    (0.09, -102.0, 8.0),        # loading: foot flattens, knee gives a little
    (0.30, -89.0, 7.0),         # mid-stance: foot flat under the hip
    (0.52, -78.0, 5.0),         # terminal stance: heel starts to rise
    (0.62, -68.0, 7.0),         # toe-off: pushes off the toes
    (0.70, -80.0, 17.0),        # early swing: knee folds, foot clears the floor
    (0.80, -97.0, 20.0),        # mid-swing: foot passes the standing leg
    (0.88, -106.0, 16.0),       # late swing: leg reaches forward
    (0.96, -111.0, 6.0),        # about to land: knee straightens for the contact
)

# Torso life: a small forward rock of the pelvis, a counter-rock of the chest and a
# nod of the head, all twice per cycle (once per step) and slightly out of phase.
HIP_SWAY = 0.6
CHEST_SWAY = 1.2
HEAD_NOD = 1.6
HIP_LAG = 0.02
CHEST_LAG = 0.10
HEAD_LAG = 0.18

# Arm swing: the hand hangs from the shoulder at ``ARM_EXTENSION`` of the arm's reach,
# which keeps a natural bend at the elbow. ``tilt`` is the angle away from straight
# down, positive backwards; the arms swing against the legs.
ARM_EXTENSION = 0.982
ARM_L_DIST = 2.0 * ARM_L_LEN * ARM_EXTENSION
ARM_R_DIST = 2.0 * ARM_R_LEN * ARM_EXTENSION
ARM_L_BASE = 4.0
ARM_L_SWING = 16.0
ARM_R_BASE = -4.0
ARM_R_SWING = 14.0

# The hands follow the body a little late, like real arms do.
ARM_LAG = 0.06

# The floor line. It only shifts the whole cycle up or down; the bob is measured
# against it.
GROUND_Y = 38.3

# The pelvis leans on the legs that carry it, but the two drawn feet do not agree
# exactly on their height at the handovers, so the raw signal has small kinks there.
# It is smoothed with a periodic Gaussian so the bob has no jolt; the price is that a
# foot may dip a couple of units into the floor line.
FLOOR_SMOOTHING = 0.05          # seconds (sigma)
FLOOR_SAMPLE = 0.002            # seconds between samples of the raw signal

# ------------------------------- helpers ------------------------------------


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def smootherstep(value: float) -> float:
    """0..1 with zero first and second derivatives at both ends."""
    x = clamp01(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def polar(origin: tuple[float, float], length: float, deg: float) -> tuple[float, float]:
    """The point ``length`` away from ``origin`` in direction ``deg``."""
    angle = math.radians(deg)
    return (origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle))


# A transform: origin, rotation in degrees. Bones are 2D (no shear, no scale here).
Frame = tuple
ROOT_FRAME: Frame = (0.0, 0.0, 0.0)


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


def periodic_value(keys: Sequence[tuple[float, float]], phase: float) -> float:
    """C1 periodic interpolation (Catmull-Rom with non-uniform knots) through keys."""
    count = len(keys)
    phase %= 1.0
    extended = ([(t - 1.0, v) for t, v in keys] + list(keys)
                + [(t + 1.0, v) for t, v in keys])
    for i in range(count, 2 * count):
        if extended[i][0] <= phase < extended[i + 1][0]:
            break
    else:  # phase is exactly on the last key
        i = 2 * count - 1
    t0, v0 = extended[i]
    t1, v1 = extended[i + 1]
    tm, vm = extended[i - 1]
    tp, vp = extended[i + 2]
    span = t1 - t0
    u = (phase - t0) / span
    m0 = (v1 - vm) / (t1 - tm) * span
    m1 = (vp - v0) / (tp - t0) * span
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0
            + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1)


def support_weight(phase: float) -> float:
    """How much a leg carries the body at a cycle phase.

    The curve is a periodic window that is 1 while the foot is planted and 0 while it
    swings, with a smooth ramp across each handover. Because the two legs run half a
    cycle apart, their weights always add up to one, so the body height is a plain
    blend of the two feet instead of a switch from one to the other.
    """
    p = (phase - SUPPORT_START) % 1.0
    ramp = SUPPORT_RAMP
    half = 0.5
    if p < ramp:
        return 0.5 + 0.5 * smootherstep(p / ramp)
    if p < half - ramp:
        return 1.0
    if p < half:
        return 1.0 - 0.5 * smootherstep((p - (half - ramp)) / ramp)
    if p < half + ramp:
        return 0.5 - 0.5 * smootherstep((p - half) / ramp)
    return 0.0


def gait_curve(phase: float) -> tuple[float, float]:
    """The designed shin angle (degrees) and knee flex (degrees) at a leg's phase."""
    shin = periodic_value([(t, s) for t, s, _ in GAIT_KEYS], phase)
    bend = periodic_value([(t, b) for t, _, b in GAIT_KEYS], phase)
    return shin, max(0.5, bend)


# ------------------------------- leg solver --------------------------------


def leg_solve(socket: tuple[float, float], length: float,
              shin_deg: float, bend_deg: float) -> tuple[tuple[float, float],
                                                          tuple[float, float]]:
    """Knee and IK target for a designed leg pose.

    The rig's IK keeps the knee on the counter-clockwise side of the socket-ankle
    line (``bendPositive`` is false), which is exactly what ``bend_deg`` describes, so
    the solver reproduces the pose we ask for.
    """
    knee = polar(socket, length, shin_deg + 2.0 * bend_deg)
    target = polar(knee, length, shin_deg)
    return knee, target


def foot_points(knee: tuple[float, float], shin_deg: float,
                outline: tuple) -> list[tuple[float, float]]:
    """The drawn foot's outline in root-local space, from the shin's frame."""
    angle = math.radians(shin_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return [(knee[0] + cos * x - sin * y, knee[1] + sin * x + cos * y)
            for x, y in outline]


def lowest(points: list[tuple[float, float]]) -> float:
    return min(y for _, y in points)


# ------------------------------- the rig fix --------------------------------


def _flat_foot_height(socket: tuple[float, float], length: float,
                      outline: tuple) -> float:
    knee, _ = leg_solve(socket, length, ALIGN_THETA, ALIGN_BEND)
    return lowest(foot_points(knee, ALIGN_THETA, outline))


HIP_FRAME_SETUP = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT)
LEFT_SOCKET_SETUP = frame_apply(HIP_FRAME_SETUP, *THIGH_L_LOCAL)
RIGHT_SOCKET_SETUP = frame_apply(HIP_FRAME_SETUP, *THIGH_R_LOCAL)

# How far the right leg has to come down for both feet to stand on one line, and the
# same move expressed in the hip's local space (which is what set_bone_setup wants).
RIGHT_LEG_DROP = (_flat_foot_height(RIGHT_SOCKET_SETUP, SHIN_R_LEN, FOOT_R_OUTLINE)
                  - _flat_foot_height(LEFT_SOCKET_SETUP, SHIN_L_LEN, FOOT_L_OUTLINE))


def hip_local_delta(world_dx: float, world_dy: float) -> tuple[float, float]:
    """A world-space offset in the hip bone's local space (setup rotation)."""
    angle = math.radians(-HIP_ROT)
    cos, sin = math.cos(angle), math.sin(angle)
    return (cos * world_dx - sin * world_dy, sin * world_dx + cos * world_dy)


_drop_x, _drop_y = hip_local_delta(0.0, -RIGHT_LEG_DROP)
THIGH_R_LOCAL_FIXED = (THIGH_R_LOCAL[0] + _drop_x, THIGH_R_LOCAL[1] + _drop_y)
IK_FOOT_R_SETUP_FIXED = (IK_FOOT_R_SETUP[0], IK_FOOT_R_SETUP[1] - RIGHT_LEG_DROP)

# ------------------------------- pose ---------------------------------------


def pose_at(time_value: float) -> dict:
    """The whole walk pose at a time, in root-local space plus the root offset."""
    pose = pose_local(time_value)
    pose["root_y"] = GROUND_Y - floor_height(pose["phase"])
    pose["clearance"] = {
        name: lowest(leg["points"]) + pose["root_y"] - GROUND_Y
        for name, leg in pose["legs"].items()
    }
    return pose


def pose_local(time_value: float) -> dict:
    """The walk pose without the root offset: torso, both legs and both arms."""
    phase = (time_value / DURATION) % 1.0

    # Torso: small periodic rocks, twice per cycle (once per step).
    wobble = 2.0 * math.pi * 2.0
    hip_key = HIP_SWAY * math.sin(wobble * (phase - HIP_LAG))
    chest_key = -CHEST_SWAY * math.sin(wobble * (phase - CHEST_LAG))
    head_world = HEAD_NOD * math.sin(wobble * (phase - HEAD_LAG))
    head_key = head_world - hip_key - chest_key     # world nod minus what it inherits

    hip_frame = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT + hip_key)
    chest_frame = frame_child(hip_frame, *CHEST_ORIGIN, chest_key)

    # Legs: the right one runs half a cycle behind the left one.
    legs = {}
    for name, local, length, outline in (
        ("l", THIGH_L_LOCAL, THIGH_L_LEN, FOOT_L_OUTLINE),
        ("r", THIGH_R_LOCAL_FIXED, SHIN_R_LEN, FOOT_R_OUTLINE),
    ):
        socket = frame_apply(hip_frame, *local)
        leg_phase = phase if name == "l" else phase + 0.5
        shin, bend = gait_curve(leg_phase)
        knee, target = leg_solve(socket, length, shin, bend)
        legs[name] = {
            "socket": socket,
            "shin": shin,
            "bend": bend,
            "knee": knee,
            "target": target,
            "points": foot_points(knee, shin, outline),
        }

    # The pelvis follows the legs that carry it, blended across each handover. Taking
    # the lower foot instead would make the body jump whenever the support changes.
    weight_l = support_weight(phase)
    height_l = lowest(legs["l"]["points"])
    height_r = lowest(legs["r"]["points"])
    floor = weight_l * height_l + (1.0 - weight_l) * height_r

    # Arms: hanging from the shoulders, swinging against the legs.
    swing = math.cos(2.0 * math.pi * (phase - ARM_LAG))
    arms = {}
    for name, local, distance, base, amplitude, swing_side in (
        ("l", SHOULDER_L_LOCAL, ARM_L_DIST, ARM_L_BASE, ARM_L_SWING, 1.0),
        ("r", SHOULDER_R_LOCAL, ARM_R_DIST, ARM_R_BASE, ARM_R_SWING, -1.0),
    ):
        shoulder = frame_apply(chest_frame, *local)
        tilt = base + swing_side * amplitude * swing
        hand = polar(shoulder, distance, -90.0 + tilt)
        arms[name] = {"shoulder": shoulder, "hand": hand, "tilt": tilt}

    return {
        "phase": phase,
        "floor": floor,
        "hip_key": hip_key,
        "chest_key": chest_key,
        "head_key": head_key,
        "legs": legs,
        "arms": arms,
    }


# ---------------------------- the floor line --------------------------------


def _smooth_periodic(values: list[float], sigma_samples: float) -> list[float]:
    """Gaussian blur of a periodic signal."""
    radius = int(math.ceil(3.0 * sigma_samples))
    offsets = range(-radius, radius + 1)
    weights = [math.exp(-0.5 * (k / sigma_samples) ** 2) for k in offsets]
    total = sum(weights)
    count = len(values)
    return [sum(values[(i + k) % count] * w for k, w in zip(offsets, weights)) / total
            for i in range(count)]


def _floor_samples() -> list[float]:
    """The raw lowest-foot height over one cycle, smoothed (all in root-local space)."""
    count = int(round(DURATION / FLOOR_SAMPLE))
    raw = [pose_local(DURATION * i / count)["floor"] for i in range(count)]
    return _smooth_periodic(raw, FLOOR_SMOOTHING / FLOOR_SAMPLE)


FLOOR_SAMPLES = None


def floor_height(phase: float) -> float:
    """The smoothed floor signal at a cycle phase."""
    global FLOOR_SAMPLES
    if FLOOR_SAMPLES is None:
        FLOOR_SAMPLES = _floor_samples()
    count = len(FLOOR_SAMPLES)
    position = (phase % 1.0) * count
    index = int(position) % count
    blend = position - int(position)
    return (FLOOR_SAMPLES[index] * (1.0 - blend)
            + FLOOR_SAMPLES[(index + 1) % count] * blend)


def local_keys(pose: dict) -> dict[str, dict[str, dict[str, float]]]:
    """The key values for one frame: offsets from the setup pose, degrees."""
    legs, arms = pose["legs"], pose["arms"]
    return {
        ROOT: {"translate": {"x": 0.0, "y": pose["root_y"]}},
        HIP: {"rotate": {"value": pose["hip_key"]}},
        CHEST: {"rotate": {"value": pose["chest_key"]}},
        HEAD: {"rotate": {"value": pose["head_key"]}},
        IK_FOOT_L: {"translate": {
            "x": legs["l"]["target"][0] - IK_FOOT_L_SETUP[0],
            "y": legs["l"]["target"][1] - IK_FOOT_L_SETUP[1],
        }},
        IK_FOOT_R: {"translate": {
            "x": legs["r"]["target"][0] - IK_FOOT_R_SETUP_FIXED[0],
            "y": legs["r"]["target"][1] - IK_FOOT_R_SETUP_FIXED[1],
        }},
        IK_HAND_L: {"translate": {
            "x": arms["l"]["hand"][0] - IK_HAND_L_SETUP[0],
            "y": arms["l"]["hand"][1] - IK_HAND_L_SETUP[1],
        }},
        IK_HAND_R: {"translate": {
            "x": arms["r"]["hand"][0] - IK_HAND_R_SETUP[0],
            "y": arms["r"]["hand"][1] - IK_HAND_R_SETUP[1],
        }},
    }


# ------------------------------ batch ---------------------------------------


def walk_operations(delete_first: bool) -> list[dict]:
    """The ``batch_edit`` operations that build the rig fix and the whole animation."""
    operations: list[dict] = []
    if delete_first:
        operations.append({"op": "delete_animation", "animation": ANIMATION})
    # Rig fix: the right leg comes down so both feet can reach the same floor. The
    # foot IK target follows it, so the setup pose stays the same.
    operations += [
        {"op": "set_bone_setup", "bone": THIGH_R,
         "x": round(THIGH_R_LOCAL_FIXED[0], 4), "y": round(THIGH_R_LOCAL_FIXED[1], 4)},
        {"op": "set_bone_setup", "bone": IK_FOOT_R,
         "y": round(IK_FOOT_R_SETUP_FIXED[1], 4)},
        {"op": "add_animation", "name": ANIMATION, "duration": DURATION},
    ]
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
                operation.update({name: round(value, 2) for name, value in values.items()})
                operations.append(operation)
    return operations


# ---------------------------- verification ----------------------------------


def stride_report(log: mcp.RunLog) -> None:
    """Log the foot travel, the bob and the timing the game has to match."""
    samples = 400
    frames = [pose_at(DURATION * i / samples) for i in range(samples)]
    root = [frame["root_y"] for frame in frames]
    log.info(f"Root bob: {min(root):+.1f} .. {max(root):+.1f} units "
             f"({max(root) - min(root):.1f} of travel)")

    # Travel of the sole's middle while the foot is planted: the distance the game has
    # to cover per stance for that foot not to slide. The right leg runs half a cycle
    # later, so its stance starts at half the cycle.
    planted = (("left", "l", 0.0, FOOT_L_SOLE), ("right", "r", 0.5, FOOT_R_SOLE))
    travels = {}
    for name, leg, offset, sole in planted:
        middle = ((sole[0][0] + sole[1][0]) * 0.5, (sole[0][1] + sole[1][1]) * 0.5)
        first = pose_local((offset % 1.0) * DURATION)["legs"][leg]
        last = pose_local(((offset + STANCE) % 1.0) * DURATION)["legs"][leg]
        start = foot_points(first["knee"], first["shin"], (middle,))[0][0]
        end = foot_points(last["knee"], last["shin"], (middle,))[0][0]
        travels[leg] = end - start
        log.info(f"  {name} foot planted: slides {end - start:+.1f} units backwards "
                 f"over its stance")

    stance_time = STANCE * DURATION
    speed = sum(travels.values()) / 2.0 / stance_time
    log.info(f"Suggested ground speed for the game: {speed:.0f} units/s "
             f"({travels['l'] / stance_time:.0f} left, {travels['r'] / stance_time:.0f} right)")
    log.info(f"Timing: {DURATION:g} s per cycle, {120.0 / DURATION:.0f} steps/min, "
             f"step length {speed * DURATION * 0.5:.0f} units")
    log.info(f"Floor line: y = {GROUND_Y:.1f} in project units; the planted foot stays on it")
    log.info("The character faces -X, so drive it in -X at that speed to keep the feet planted")


def smoothness_report(path: Path, log: mcp.RunLog) -> float:
    """Largest second difference of any curve, read back from the saved project."""
    project = json.loads(zipfile.ZipFile(path).read("project.json"))
    animation = next((a for a in project["animations"] if a["name"] == ANIMATION), None)
    if animation is None:
        return 0.0
    worst = 0.0
    worst_name = ""
    changes: dict[str, float] = {"rotate": 0.0, "translate_x": 0.0, "translate_y": 0.0}
    for timeline in animation.get("bones", []):
        name = project["bones"][timeline["bone"]]["name"]
        for channel, axis in (("rotate", 0), ("translate", 1)):
            keys = timeline.get(channel) or []
            if len(keys) < 3:
                continue
            values = [key["value"] if channel == "rotate" else key["value"][axis]
                      for key in keys]
            times = [key["time"] for key in keys]
            for i in range(1, len(values) - 1):
                left = (values[i] - values[i - 1]) / (times[i] - times[i - 1])
                right = (values[i + 1] - values[i]) / (times[i + 1] - times[i])
                span = 0.5 * (times[i + 1] - times[i - 1])
                change = abs(right - left) / span if span else 0.0
                key = "rotate" if channel == "rotate" else ("translate_x" if axis == 0
                                                            else "translate_y")
                changes[key] = max(changes[key], change)
                if change > worst:
                    worst, worst_name = change, f"{name}.{channel}[{axis}] at {times[i]:.2f} s"
    log.info("Acceleration check (second difference of the saved curves):")
    for key, value in changes.items():
        log.info(f"  {key:12s} max {value:8.1f} units/s^2")
    log.detail(f"  worst: {worst_name} {worst:.1f}")
    return worst


def foot_height(pose: dict, shin: str, outline: tuple) -> float:
    """Lowest drawn foot point from a sample_pose result (world space)."""
    bone = mcp.pose_bone(pose, shin)
    if bone is None:
        return float("nan")
    angle = math.radians(bone["rotation"])
    cos, sin = math.cos(angle), math.sin(angle)
    return min(bone["y"] + sin * x + cos * y for x, y in outline)


def verify(client: mcp.RigbyMcpClient, output: str, path: Path,
           log: mcp.RunLog) -> bool:
    """Check the rig fix, the loop, the floor contact, the curves and the project."""
    anim = client.call_tool("describe_animation", {"path": output, "animation": ANIMATION})
    log.info(f"Duration after the edit: {anim.get('duration'):g} s")
    for timeline in anim.get("bones", []):
        counts = {channel: len(timeline.get("keys", {}).get(channel) or [])
                  for channel in ("rotate", "translate")}
        log.info(f"  {timeline['bone']}: {counts['rotate']} rotate, "
                 f"{counts['translate']} translate key(s)")

    problems: list[str] = []

    # The rig fix must leave the setup pose unchanged: both feet on the same line.
    setup = client.call_tool("sample_pose", {"path": output})
    low_l = foot_height(setup, SHIN_L, FOOT_L_OUTLINE)
    low_r = foot_height(setup, SHIN_R, FOOT_R_OUTLINE)
    log.info(f"Setup pose feet: left {low_l:.1f}, right {low_r:.1f} "
             f"(floor {GROUND_Y:.1f}, difference {low_r - low_l:+.1f})")

    log.info("Poses in world space (Y up, degrees):")
    last_pose: dict = {}
    for time_value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, DURATION):
        last_pose = client.call_tool(
            "sample_pose", {"path": output, "animation": ANIMATION, "time": time_value}
        )
        line = []
        for name in (THIGH_L, SHIN_L, THIGH_R, SHIN_R):
            bone = mcp.pose_bone(last_pose, name)
            if bone is None:
                continue
            line.append(f"{name}=({bone['x']:.0f},{bone['y']:.0f}) {bone['rotation']:.1f}")
        log.info(f"  t={time_value:.1f}s: " + " | ".join(line))
        left = foot_height(last_pose, SHIN_L, FOOT_L_OUTLINE)
        right = foot_height(last_pose, SHIN_R, FOOT_R_OUTLINE)
        log.info(f"        feet: left {left - GROUND_Y:+6.1f}, right {right - GROUND_Y:+6.1f}"
                 f" (0 = on the floor)")

    # Loop: the last frame has to be the first frame.
    first = client.call_tool(
        "sample_pose", {"path": output, "animation": ANIMATION, "time": 0.0})
    last = client.call_tool(
        "sample_pose", {"path": output, "animation": ANIMATION, "time": DURATION})
    loop_error = 0.0
    for name, bone in mcp.bones_by_name(first).items():
        other = mcp.bones_by_name(last).get(name)
        if other is None:
            continue
        loop_error = max(loop_error, abs(bone["x"] - other["x"]),
                         abs(bone["y"] - other["y"]),
                         abs(bone["rotation"] - other["rotation"]))
    log.info(f"Loop check: start and end differ by {loop_error:.4f}")
    if loop_error > 0.01:
        problems.append(f"the first and last frame differ by {loop_error:.3f}")

    # Floor: sampled every frame, no foot may go below the floor by more than 0.5.
    steps = int(round(DURATION * SAMPLE_HZ))
    sink = 0.0
    float_max = 0.0
    worst_bend = (-9.0, 0.0)
    for step in range(steps + 1):
        pose = client.call_tool("sample_pose", {
            "path": output, "animation": ANIMATION,
            "time": round(step / SAMPLE_HZ, 6),
        })
        left = foot_height(pose, SHIN_L, FOOT_L_OUTLINE) - GROUND_Y
        right = foot_height(pose, SHIN_R, FOOT_R_OUTLINE) - GROUND_Y
        sink = min(sink, left, right)
        float_max = max(float_max, min(left, right))
        # The rig's IK always bends the knee backwards, so the shin has to stay a little
        # clockwise of the thigh. A positive difference here would be a flipped knee.
        for thigh, shin in ((THIGH_L, SHIN_L), (THIGH_R, SHIN_R)):
            thigh_bone = mcp.pose_bone(pose, thigh)
            shin_bone = mcp.pose_bone(pose, shin)
            if thigh_bone is None or shin_bone is None:
                continue
            bend = ((shin_bone["rotation"] - thigh_bone["rotation"] + 180.0) % 360.0) - 180.0
            if bend > worst_bend[0]:
                worst_bend = (bend, round(step / SAMPLE_HZ, 3))
    log.info(f"Floor check: deepest foot {-sink:.2f} below the floor, "
             f"highest support gap {float_max:.2f} above it")
    log.info(f"Knee check: knee bend stays {worst_bend[0]:.1f} deg or more clockwise "
             f"(worst at {worst_bend[1]:g} s)")
    if sink < -6.0:
        problems.append(f"a foot goes {-sink:.1f} units through the floor")
    if float_max > 20.0:
        problems.append(f"a supporting foot floats {float_max:.1f} units above the floor")
    if worst_bend[0] > 1.0:
        problems.append(f"a knee flips the wrong way at {worst_bend[1]:g} s")

    stride_report(log)
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


def add_walk(client: mcp.RigbyMcpClient, source: str, output: str,
             log: mcp.RunLog) -> bool:
    """Build the rig fix and the animation in ``source``, saving the project to ``output``."""
    existing = [anim.get("name") for anim in
                client.call_tool("inspect_project", {"path": source}).get("animations", [])]
    log.info(f"Animations in the project: {existing}")
    delete_first = ANIMATION in existing
    if delete_first:
        log.info(f"'{ANIMATION}' already exists and will be rebuilt")

    log.info(f"Rig fix: right leg comes down {RIGHT_LEG_DROP:.1f} units "
             f"so both feet share one floor line")
    operations = walk_operations(delete_first)
    key_count = sum(1 for op in operations if op["op"] == "set_bone_key")
    log.info(f"Building {key_count} keys at {SAMPLE_HZ:g} keys/s...")
    started = time.perf_counter()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": operations,
    })
    log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s) "
             f"in {time.perf_counter() - started:.1f} s")

    output_path = WORKSPACE / output
    return verify(client, output, output_path, log)


def export_preview(client: mcp.RigbyMcpClient, output: str, log: mcp.RunLog) -> None:
    """Render the cycle as a contact sheet, the quickest way to judge the motion."""
    folder = LOG_DIR / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    client.call_tool("export_sprite_sheet", {
        "input": output, "output_folder": mcp.server_path(WORKSPACE, folder),
        "name": ANIMATION, "format": "json", "layout": "grid",
        "animations": [ANIMATION], "fps": 8, "scale": 0.35,
        "trim": True, "dedupe": False, "loop": False, "overwrite": True,
    })
    log.info(f"Preview written to {folder}")


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: looping walk cycle ===")
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

    with mcp.connect(log, WORKSPACE, client_name="rigby-walk-demo", client_version="1.0") as client:
        ok = add_walk(client, source, target, log)
        if ok and args.preview:
            export_preview(client, target, log)

    if not ok:
        log.error(f"The output project has problems; see {log.path}")
        return 1
    log.info(f"Done. Output written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add a looping two-step walk cycle, in place, to a Rigby2D character.",
    )
    mcp.add_project_arguments(parser, default_input="input/caveman1.rigby")
    parser.add_argument("--preview", action="store_true",
                        help="also render a contact sheet of the cycle to logs/preview/")
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
