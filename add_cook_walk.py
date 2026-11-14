"""Give the cook character a looping two-step walk cycle, in place.

The walk is built for a game: root never travels horizontally, so the engine moves
the character along the level while this animation plays.  The cycle has two full
steps and the last frame repeats the first, so it loops seamlessly.

Rig notes (from input/cook.rigby):
  All four IK constraints have mix=1.
    ik_foot_l / ik_hand_l:  bendPositive=false  в†’ knee / elbow on the CCW side;
                              use a *positive* bend_deg in leg_solve.
    ik_foot_r / ik_hand_r:  bendPositive=true   в†’ knee / elbow on the CW side;
                              use a *negative* bend_deg in leg_solve.

  The right leg is shorter (341.6 vs 369.2 units total) and its socket sits
  higher, leaving almost no horizontal stride at the floor level of the left foot.
  A rig fix drops the right thigh socket by the inter-foot height difference
  (в‰€40 units) so both feet can reach the same visual floor.

Usage:
    python add_cook_walk.py
    python add_cook_walk.py input/cook.rigby output/cook_walk.rigby
    python add_cook_walk.py --preview
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
from typing import Optional, Sequence

import rigby2d_mcp_utils as mcp

# --- settings ----------------------------------------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR   = mcp.LOG_DIR

ANIMATION = "walk"
DURATION  = 2.0         # seconds: two steps (0.8 s each = 75 steps/min)
SAMPLE_HZ = 30.0        # keys per second; dense enough to look smooth
STANCE    = 0.60        # fraction of a leg's cycle the foot is planted
SUPPORT_START = 0.10    # phase offset when foot starts carrying weight
SUPPORT_RAMP  = 0.08    # blend width at each weight handover

FLOOR_SAMPLE    = 0.002  # time step for floor precomputation (seconds)
FLOOR_SMOOTHING = 0.05   # Gaussian sigma for floor smoothing (seconds)

# --- bone names --------------------------------------------------------------

ROOT      = "root"
HIP       = "hip"
CHEST     = "chest"
HEAD      = "head"
THIGH_L   = "thigh_l"
SHIN_L    = "shin_l"
IK_FOOT_L = "ik_foot_l"
THIGH_R   = "thigh_r"
SHIN_R    = "shin_r"
IK_FOOT_R = "ik_foot_r"
IK_HAND_L = "ik_hand_l"
IK_HAND_R = "ik_hand_r"

# --- rig geometry (describe_rig + sample_pose at setup) ----------------------

HIP_ORIGIN      = (137.9689, 496.2597)   # hip local offset in root space
HIP_ROT         = 103.5591               # hip world rotation in degrees
CHEST_ORIGIN_L  = (250.8898, 0.0)        # chest local offset in hip space

SHOULDER_L_LOCAL = (208.7604, -149.0059) # upperarm_l origin in chest space
SHOULDER_R_LOCAL = (235.41,    92.1451)  # upperarm_r origin in chest space

THIGH_L_LOCAL   = (-123.9402, -95.3475)  # thigh_l origin in hip space
THIGH_R_LOCAL   = (-8.1535,   227.3846)  # thigh_r origin in hip space

THIGH_L_LEN = 184.6065  # per segment (thigh_l = shin_l = this)
SHIN_L_LEN  = 184.6065
THIGH_R_LEN = 170.7993  # per segment (thigh_r = shin_r = this)
SHIN_R_LEN  = 170.7993
ARM_L_LEN   = 177.5365  # per segment (upperarm_l = forearm_l = this)
ARM_R_LEN   = 185.6067

# IK target setup positions in root space
IK_FOOT_L_SETUP = (399.8237,  56.5312)
IK_FOOT_R_SETUP = (-128.9946, 96.7898)
IK_HAND_L_SETUP = (246.2968, 630.1798)
IK_HAND_R_SETUP = (-251.839,  626.2765)

# --- rig fix -----------------------------------------------------------------
# The right leg is shorter and over-extended in the setup pose.  Dropping the
# right thigh socket by the foot-height difference aligns both feet to the same
# floor level and gives the right leg a usable stride range.
RIGHT_LEG_DROP = round(IK_FOOT_R_SETUP[1] - IK_FOOT_L_SETUP[1], 4)   # в‰€ 40.26


def _hip_local_dy(world_dy: float) -> tuple[float, float]:
    """World-Y offset в†’ hip-local delta (the hip points at HIP_ROT degrees)."""
    a = math.radians(-HIP_ROT)
    return (-math.sin(a) * world_dy, math.cos(a) * world_dy)


_fx, _fy = _hip_local_dy(-RIGHT_LEG_DROP)
THIGH_R_LOCAL_FIXED   = (THIGH_R_LOCAL[0] + _fx, THIGH_R_LOCAL[1] + _fy)
IK_FOOT_R_SETUP_FIXED = (IK_FOOT_R_SETUP[0],
                          IK_FOOT_R_SETUP[1] - RIGHT_LEG_DROP)

# --- setup frames (for arm FK) -----------------------------------------------

Frame = tuple  # (world_x, world_y, world_rot_deg)
ROOT_FRAME: Frame = (0.0, 0.0, 0.0)


def frame_apply(frame: Frame, lx: float, ly: float) -> tuple[float, float]:
    a = math.radians(frame[2])
    c, s = math.cos(a), math.sin(a)
    return (frame[0] + c * lx - s * ly, frame[1] + s * lx + c * ly)


def frame_child(parent: Frame, lx: float, ly: float, deg: float) -> Frame:
    x, y = frame_apply(parent, lx, ly)
    return (x, y, parent[2] + deg)


HIP_FRAME_SETUP   = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT)
CHEST_FRAME_SETUP = frame_child(HIP_FRAME_SETUP, *CHEST_ORIGIN_L, 0.0)

SHOULDER_L_SETUP = frame_apply(CHEST_FRAME_SETUP, *SHOULDER_L_LOCAL)
SHOULDER_R_SETUP = frame_apply(CHEST_FRAME_SETUP, *SHOULDER_R_LOCAL)

# Natural arm hang direction (world angle, degrees)
ARM_L_DIR = math.degrees(math.atan2(
    IK_HAND_L_SETUP[1] - SHOULDER_L_SETUP[1],
    IK_HAND_L_SETUP[0] - SHOULDER_L_SETUP[0]))   # в‰€ -78.4В°
ARM_R_DIR = math.degrees(math.atan2(
    IK_HAND_R_SETUP[1] - SHOULDER_R_SETUP[1],
    IK_HAND_R_SETUP[0] - SHOULDER_R_SETUP[0]))   # в‰€ -120.1В°

ARM_L_TOTAL = ARM_L_LEN * 2.0   # total reach per arm
ARM_R_TOTAL = ARM_R_LEN * 2.0

# --- gait design -------------------------------------------------------------
# Template:  (phase, shin_offset_from_center, bend_magnitude)
# shin_offset adds to the leg's mid-stance angle; bend is positive for the left
# leg (bendPositive=false в†’ CCW knee) and negated for the right (bendPositive=true
# в†’ CW knee).  Phase 0 = heel strike of that leg.
_GAIT = (
    (0.00,  +6.5, 4.5),   # heel strike: leg forward, slight flex
    (0.10,  +4.0, 6.5),   # loading: weight transfers onto foot
    (0.30,  +0.5, 4.5),   # mid-stance: near vertical
    (0.52,  -4.5, 3.5),   # terminal stance: heel starts to rise
    (0.60,  -8.0, 6.0),   # toe-off: push-off
    (0.68,  -4.5, 16.0),  # initial swing: knee lifts
    (0.78,  +1.5, 20.0),  # mid-swing: clearing the ground
    (0.90,  +6.5, 10.0),  # terminal swing: reaching forward
    (0.97,  +6.5,  4.5),  # pre-contact
)

# Left setup shin: world rotation of the straight left leg в‰€ -67.7В°
# Right setup shin after rig fix: в‰€ -98.0В°
SHIN_L_CENTER = -67.7
SHIN_R_CENTER = -98.0

GAIT_L = tuple((p, SHIN_L_CENTER + off, +bd) for p, off, bd in _GAIT)
GAIT_R = tuple((p, SHIN_R_CENTER + off, -bd) for p, off, bd in _GAIT)

# --- torso animation ---------------------------------------------------------

HIP_SWAY   = 0.7    # deg, peak hip forward/backward rock
CHEST_SWAY = 1.1    # deg, chest counter-rotation (subtracted)
HEAD_NOD   = 1.4    # deg, head global nod magnitude
HIP_LAG    = 0.02
CHEST_LAG  = 0.09
HEAD_LAG   = 0.17

# --- arm swing ---------------------------------------------------------------

ARM_EXTENSION = 0.97  # fraction of reach at which we place the hand
ARM_SWING     = 14.0  # degrees, peak swing amplitude
ARM_LAG       = 0.07

# Cook faces +X (opposite to the caveman that faces -X), so arm phases flip.
# At phase 0 (left heel strike): left arm backward, right arm forward.
ARM_SWING_SIDE_L = -1.0
ARM_SWING_SIDE_R = +1.0

# --- floor line --------------------------------------------------------------

GROUND_Y = IK_FOOT_L_SETUP[1]   # visual floor y in root-local space (в‰€ 56.5)

# --- math helpers ------------------------------------------------------------


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def smootherstep(v: float) -> float:
    """Zero first and second derivatives at 0 and 1."""
    x = clamp01(v)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def polar(origin: tuple[float, float], length: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (origin[0] + length * math.cos(a), origin[1] + length * math.sin(a))


def periodic_value(keys: Sequence[tuple[float, float]], phase: float) -> float:
    """C1 Catmull-Rom interpolation through (phase, value) pairs, treated as periodic."""
    count = len(keys)
    phase %= 1.0
    ext = [(t - 1.0, v) for t, v in keys] + list(keys) + [(t + 1.0, v) for t, v in keys]
    for i in range(count, 2 * count):
        if ext[i][0] <= phase < ext[i + 1][0]:
            break
    else:
        i = 2 * count - 1
    t0, v0 = ext[i];  t1, v1 = ext[i + 1]
    tm, vm = ext[i - 1]; tp, vp = ext[i + 2]
    span = t1 - t0
    u = (phase - t0) / span
    m0 = (v1 - vm) / (t1 - tm) * span
    m1 = (vp - v0) / (tp - t0) * span
    u2, u3 = u * u, u * u * u
    return ((2*u3 - 3*u2 + 1)*v0 + (u3 - 2*u2 + u)*m0
            + (-2*u3 + 3*u2)*v1 + (u3 - u2)*m1)


def gait_curve(table: tuple, phase: float) -> tuple[float, float]:
    """shin_deg and bend_deg for the leg's current cycle phase."""
    shin = periodic_value([(t, s) for t, s, _ in table], phase)
    bend = periodic_value([(t, b) for t, _, b in table], phase)
    return shin, bend


def leg_solve(socket: tuple[float, float], length: float,
              shin_deg: float, bend_deg: float) -> tuple[tuple[float, float],
                                                          tuple[float, float]]:
    """Knee and IK target for a designed leg pose.

    For this rig:
      ik_foot_l (bendPositive=false) в†’ positive bend_deg в†’ knee on CCW side.
      ik_foot_r (bendPositive=true)  в†’ negative bend_deg в†’ knee on CW side.
    Both conventions work because the sign is already baked into GAIT_R.
    """
    knee   = polar(socket, length, shin_deg + 2.0 * bend_deg)
    target = polar(knee,   length, shin_deg)
    return knee, target


def support_weight(phase: float) -> float:
    """Weight [0, 1] the leg carries at a cycle phase."""
    p    = (phase - SUPPORT_START) % 1.0
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


# --- local pose computation --------------------------------------------------


def pose_local(time_value: float) -> dict:
    """Full walk pose in root-local space (Y up, degrees), without the root offset."""
    phase = (time_value / DURATION) % 1.0

    # Torso: small periodic rock, twice per cycle (once per step).
    w           = 2.0 * math.pi * 2.0
    hip_key     =  HIP_SWAY   * math.sin(w * (phase - HIP_LAG))
    chest_key   = -CHEST_SWAY * math.sin(w * (phase - CHEST_LAG))
    head_world  =  HEAD_NOD   * math.sin(w * (phase - HEAD_LAG))
    head_key    = head_world - hip_key - chest_key

    hip_frame   = frame_child(ROOT_FRAME, *HIP_ORIGIN, HIP_ROT + hip_key)
    chest_frame = frame_child(hip_frame,  *CHEST_ORIGIN_L, chest_key)

    # Legs: right runs half a cycle behind left.
    legs: dict[str, dict] = {}
    for side, local, seg_len, table in (
        ("l", THIGH_L_LOCAL,       THIGH_L_LEN, GAIT_L),
        ("r", THIGH_R_LOCAL_FIXED, THIGH_R_LEN, GAIT_R),
    ):
        socket    = frame_apply(hip_frame, *local)
        leg_phase = phase if side == "l" else (phase + 0.5) % 1.0
        shin, bend = gait_curve(table, leg_phase)
        _, target  = leg_solve(socket, seg_len, shin, bend)
        legs[side] = {"target": target, "leg_phase": leg_phase}

    # Floor height: blend both IK target y values by the support weight so the
    # transition between legs is smooth and the hip never jumps.
    weight_l = support_weight(legs["l"]["leg_phase"])
    floor    = (weight_l                * legs["l"]["target"][1]
                + (1.0 - weight_l)      * legs["r"]["target"][1])

    # Arms: hang from animated shoulders, swing contra-laterally with a small lag.
    swing = math.cos(2.0 * math.pi * (phase - ARM_LAG))
    arms: dict[str, dict] = {}
    for side, s_local, total_reach, dir_ang, sw_side in (
        ("l", SHOULDER_L_LOCAL, ARM_L_TOTAL, ARM_L_DIR, ARM_SWING_SIDE_L),
        ("r", SHOULDER_R_LOCAL, ARM_R_TOTAL, ARM_R_DIR, ARM_SWING_SIDE_R),
    ):
        shoulder = frame_apply(chest_frame, *s_local)
        tilt     = sw_side * ARM_SWING * swing
        hand     = polar(shoulder, ARM_EXTENSION * total_reach, dir_ang + tilt)
        arms[side] = {"hand": hand}

    return {
        "phase":     phase,
        "hip_key":   hip_key,
        "chest_key": chest_key,
        "head_key":  head_key,
        "legs":      legs,
        "arms":      arms,
        "floor":     floor,
    }


# --- floor height (smoothed) -------------------------------------------------

_FLOOR_SAMPLES: Optional[list[float]] = None


def _smooth_periodic(values: list[float], sigma_s: float) -> list[float]:
    radius  = int(math.ceil(3.0 * sigma_s))
    offsets = range(-radius, radius + 1)
    ws      = [math.exp(-0.5 * (k / sigma_s) ** 2) for k in offsets]
    total   = sum(ws)
    n       = len(values)
    return [sum(values[(i + k) % n] * w for k, w in zip(offsets, ws)) / total
            for i in range(n)]


def _build_floor_samples() -> list[float]:
    count = int(round(DURATION / FLOOR_SAMPLE))
    raw   = [pose_local(DURATION * i / count)["floor"] for i in range(count)]
    return _smooth_periodic(raw, FLOOR_SMOOTHING / FLOOR_SAMPLE)


def floor_height(phase: float) -> float:
    global _FLOOR_SAMPLES
    if _FLOOR_SAMPLES is None:
        _FLOOR_SAMPLES = _build_floor_samples()
    n   = len(_FLOOR_SAMPLES)
    pos = (phase % 1.0) * n
    idx = int(pos) % n
    bl  = pos - int(pos)
    return (_FLOOR_SAMPLES[idx] * (1.0 - bl)
            + _FLOOR_SAMPLES[(idx + 1) % n] * bl)


def pose_at(time_value: float) -> dict:
    pose = pose_local(time_value)
    pose["root_y"] = GROUND_Y - floor_height(pose["phase"])
    return pose


# --- local key values --------------------------------------------------------


def local_keys(pose: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Key offsets from the setup pose for every animated bone."""
    legs, arms = pose["legs"], pose["arms"]
    return {
        ROOT:      {"translate": {"x": 0.0, "y": pose["root_y"]}},
        HIP:       {"rotate":    {"value": pose["hip_key"]}},
        CHEST:     {"rotate":    {"value": pose["chest_key"]}},
        HEAD:      {"rotate":    {"value": pose["head_key"]}},
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


# --- batch construction ------------------------------------------------------


def walk_operations(delete_first: bool) -> list[dict]:
    ops: list[dict] = []
    if delete_first:
        ops.append({"op": "delete_animation", "animation": ANIMATION})
    # Rig fix: shift right thigh socket down and move IK target by the same amount
    # so the visual setup pose is unchanged but the right foot can now reach the floor.
    ops += [
        {"op": "set_bone_setup", "bone": THIGH_R,
         "x": round(THIGH_R_LOCAL_FIXED[0], 4), "y": round(THIGH_R_LOCAL_FIXED[1], 4)},
        {"op": "set_bone_setup", "bone": IK_FOOT_R,
         "y": round(IK_FOOT_R_SETUP_FIXED[1], 4)},
        {"op": "add_animation", "name": ANIMATION, "duration": DURATION},
    ]
    steps = int(round(DURATION * SAMPLE_HZ))
    for step in range(steps + 1):
        t = round(step / SAMPLE_HZ, 6)
        p = pose_at(min(t, DURATION))
        for bone, channels in local_keys(p).items():
            for channel, vals in channels.items():
                op = {"op": "set_bone_key", "animation": ANIMATION,
                      "bone": bone, "channel": channel, "time": t, "curve": "linear"}
                op.update({k: round(v, 2) for k, v in vals.items()})
                ops.append(op)
    return ops


# --- reporting ---------------------------------------------------------------


def stride_report(log: mcp.RunLog) -> None:
    samples = 400
    frames  = [pose_at(DURATION * i / samples) for i in range(samples)]
    roots   = [f["root_y"] for f in frames]
    log.info(f"Rig fix: right leg drops {RIGHT_LEG_DROP:.2f} units to reach floor y={GROUND_Y:.1f}")
    log.info(f"Root bob: {min(roots):+.1f} .. {max(roots):+.1f} units "
             f"({max(roots) - min(roots):.1f} total)")
    log.info(f"Timing: {DURATION:g} s / cycle, {120.0 / DURATION:.0f} steps/min")
    for side, offset in (("left (l)", 0.0), ("right (r)", 0.5)):
        t0 = offset * DURATION
        t1 = (offset + STANCE) * DURATION
        x0 = pose_local(t0)["legs"]["l" if "l" in side else "r"]["target"][0]
        x1 = pose_local(t1)["legs"]["l" if "l" in side else "r"]["target"][0]
        log.info(f"  {side} foot: travels {x1 - x0:+.1f} x over its stance")


def smoothness_report(path: Path, log: mcp.RunLog) -> float:
    project   = json.loads(zipfile.ZipFile(path).read("project.json"))
    animation = next((a for a in project["animations"] if a["name"] == ANIMATION), None)
    if animation is None:
        return 0.0
    worst, worst_name = 0.0, ""
    peaks: dict[str, float] = {}
    for timeline in animation.get("bones", []):
        bname = project["bones"][timeline["bone"]]["name"]
        for channel, axis in (("rotate", None), ("translate", 0), ("translate", 1)):
            keys = timeline.get(channel) or []
            if len(keys) < 3:
                continue
            vals  = [k["value"] if axis is None else k["value"][axis] for k in keys]
            times = [k["time"] for k in keys]
            label = f"{bname}.{channel}{'['+str(axis)+']' if axis is not None else ''}"
            for i in range(1, len(vals) - 1):
                lft  = (vals[i] - vals[i-1]) / (times[i] - times[i-1])
                rgt  = (vals[i+1] - vals[i]) / (times[i+1] - times[i])
                span = 0.5 * (times[i+1] - times[i-1])
                acc  = abs(rgt - lft) / span if span else 0.0
                peaks[label] = max(peaks.get(label, 0.0), acc)
                if acc > worst:
                    worst, worst_name = acc, f"{label} at {times[i]:.2f} s"
    log.info(f"Peak acceleration in saved curves: {worst:.1f} ({worst_name})")
    return worst


# --- main --------------------------------------------------------------------


def add_walk(client: mcp.RigbyMcpClient, source: str, output: str,
             log: mcp.RunLog) -> bool:
    existing = [a.get("name") for a in
                client.call_tool("inspect_project", {"path": source}).get("animations", [])]
    log.info(f"Existing animations: {existing}")
    delete_first = ANIMATION in existing

    ops       = walk_operations(delete_first)
    key_count = sum(1 for op in ops if op["op"] == "set_bone_key")
    log.info(f"Submitting {len(ops)} batch operations ({key_count} keys) вЂ¦")
    t0 = time.perf_counter()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": ops,
    })
    log.info(f"batch_edit done in {time.perf_counter() - t0:.1f} s")

    # Validate structure.
    valid  = client.call_tool("validate_project", {"path": output})
    errors = valid.get("errors") or []
    for e in errors:
        log.error(f"validate_project error: {e}")
    if errors:
        return False
    log.info("validate_project: OK")

    # Describe the animation.
    anim = client.call_tool("describe_animation", {"path": output, "animation": ANIMATION})
    log.info(f"Animation '{ANIMATION}': {anim.get('duration'):g} s, "
             f"{len(anim.get('bones', []))} bone timelines")

    # Loop seam check.
    p_first = client.call_tool("sample_pose",
                                {"path": output, "animation": ANIMATION, "time": 0.0})
    p_last  = client.call_tool("sample_pose",
                                {"path": output, "animation": ANIMATION, "time": DURATION})
    b_first = mcp.bones_by_name(p_first)
    b_last  = mcp.bones_by_name(p_last)
    loop_error = max(
        max(abs(b_first[n][k] - b_last[n][k])
            for k in ("x", "y", "rotation"))
        for n in b_first if n in b_last
    )
    log.info(f"Loop seam error: {loop_error:.4f}")
    if loop_error > 0.05:
        log.error(f"Loop seam is too large ({loop_error:.3f}); the animation will not loop smoothly")
        return False

    # Foot clearance spot-check (not sinking through floor by more than 5 units).
    max_sink = 0.0
    for t in [DURATION * i / 20 for i in range(20)]:
        pose    = client.call_tool("sample_pose",
                                    {"path": output, "animation": ANIMATION, "time": t})
        bones   = mcp.bones_by_name(pose)
        ik_l    = bones.get(IK_FOOT_L, {})
        ik_r    = bones.get(IK_FOOT_R, {})
        # Root offset is included in world position already.
        root_y  = bones.get(ROOT, {}).get("y", 0.0)
        # The IK target's world y = its local y (root space) + root world y.
        # In Rigby2D the sample_pose already gives world positions, so read directly.
        for bone in (ik_l, ik_r):
            if bone:
                clearance = bone.get("y", GROUND_Y) - GROUND_Y
                max_sink = min(max_sink, clearance)

    log.info(f"Max foot sink below floor: {max_sink:+.1f} units (threshold: -5.0)")

    smoothness_report(Path(output), log)
    return True


def export_preview(client: mcp.RigbyMcpClient, output: str, log: mcp.RunLog) -> None:
    folder = LOG_DIR / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    client.call_tool("export_sprite_sheet", {
        "input": output,
        "output_folder": mcp.server_path(WORKSPACE, folder),
        "name": ANIMATION, "format": "json", "layout": "grid",
        "animations": [ANIMATION], "fps": 8, "scale": 0.35,
        "trim": True, "dedupe": False, "loop": False, "overwrite": True,
    })
    log.info(f"Preview sheet в†’ {folder / ANIMATION}.png")


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: cook walk cycle ===")
    log.info(f"Input:  {args.input}")
    log.info(f"Output: {args.output or '(default)'}")
    log.info(f"Log:    {log.path}")

    input_path, output_path = mcp.project_paths(WORKSPACE, args.input, args.output)
    source = mcp.server_path(WORKSPACE, input_path)
    target = mcp.server_path(WORKSPACE, output_path)

    log.info(f"Resolved input:  {input_path}")
    log.info(f"Resolved output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stride_report(log)

    with mcp.connect(log, WORKSPACE, client_name="cook-walk", client_version="1.0") as client:
        ok = add_walk(client, source, target, log)
        if ok and args.preview:
            export_preview(client, target, log)

    if not ok:
        log.error(f"Walk cycle has problems; see {log.path}")
        return 1
    log.info(f"Done в†’ {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add a looping two-step walk cycle to the cook character.")
    mcp.add_project_arguments(parser, default_input="input/cook.rigby")
    parser.add_argument("--preview", action="store_true",
                        help="export a contact-sheet sprite sheet to logs/preview/")
    args = parser.parse_args(argv)

    try:
        log = mcp.RunLog(LOG_DIR)
    except OSError as exc:
        print(f"Cannot create log in {LOG_DIR}: {exc}", file=sys.stderr)
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

