"""Animate a Rigby2D stick figure: a 20-second dance that ends in a split.

Reads a project, builds the ``dance`` animation with one atomic ``batch_edit`` and
writes the result to a second file: the input is never modified. Every run writes its
own log to ``logs/``.

The motion comes from a handful of smooth *section* poses (groove, arms overhead,
side steps with leg lifts, wind-down, split). The sections are cross-faded with smooth
weights and sampled densely, so the body never stops dead between moves. Every value
is a world-space intent relative to the setup pose; the animated hip rotation is
subtracted when a limb is keyed, so the hips can lean without dragging the arms,
head or legs along.

The feet stay planted during the sway: the legs are counter-rotated by the angle the
pelvis travels, which turns the side-to-side movement into a natural weight shift
instead of a slide.

Usage:
    python add_dance.py [input] [output]

    python add_dance.py                          # input/human.rigby -> output/human.rigby
    python add_dance.py input/human.rigby out/human.rigby

Paths are relative to the repository root (the MCP workspace). The output folder is
created when missing. Requirements: Python 3.8+ and Node.js 20 or newer.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import rigby2d_mcp_utils as mcp

# ------------------------------- settings -----------------------------------

WORKSPACE = mcp.WORKSPACE
LOG_DIR = mcp.LOG_DIR

ANIMATION = "dance"             # name of the animation to create
DURATION = 20.0                 # animation length, seconds
SAMPLE_HZ = 20.0                # keys per second: dense enough to read as smooth

TAU = 2.0 * math.pi

# The music is 120 BPM: one beat every 0.5 s. The body sways from side to side
# once per four beats and bounces a little on every beat.
BEAT_PERIOD = 0.5
SWAY_PERIOD = 2.0

# Bone names of this rig (describe_rig).
ROOT = "root"
HIP = "hip"
HEAD = "head"
ARM_L = "arm left"
ARM_R = "arm right"
LEG_L = "leg left"
LEG_R = "leg right"

# World rotations (degrees) of the legs in the setup pose.
LEG_L_SETUP = -103.627
LEG_R_SETUP = -79.492

# Leg geometry (describe_rig): lengths and the hip-local y offsets where the leg
# bones start. Used to keep the feet on the floor while the pelvis moves.
LEG_L_LENGTH = 234.4482
LEG_R_LENGTH = 230.46
LEG_L_OFFSET_Y = -9.0186
LEG_R_OFFSET_Y = -12.4709

# Average leg length, used to keep the feet in place while the pelvis sways.
LEG_LENGTH = 232.0

# A full side split: both legs horizontal, one pointing left, one right.
LEG_L_SPLIT = -180.0 - LEG_L_SETUP     # -76.373 degrees from the setup pose
LEG_R_SPLIT = 0.0 - LEG_R_SETUP        # +79.492 degrees from the setup pose

# The pelvis in the setup pose sits at this height and the feet stand at about
# y = -25. Dropping the root by 235 puts the pelvis on the floor in the split.
HIP_SETUP_Y = 241.03
SPLIT_DROP = 235.0

# The leg tips (bone ends) in the held split, kept at this height on the way down.
FINAL_HIP_Y = HIP_SETUP_Y - SPLIT_DROP
TIP_L_Y = FINAL_HIP_Y + LEG_L_OFFSET_Y + LEG_L_LENGTH * math.sin(
    math.radians(LEG_L_SETUP + LEG_L_SPLIT)
)
TIP_R_Y = FINAL_HIP_Y + LEG_R_OFFSET_Y + LEG_R_LENGTH * math.sin(
    math.radians(LEG_R_SETUP + LEG_R_SPLIT)
)


# ------------------------------ pose helpers --------------------------------


def clamp01(value: float) -> float:
    """Clamp to 0..1."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def smootherstep(value: float) -> float:
    """0..1 with zero first and second derivatives at both ends (no visible jolt)."""
    x = clamp01(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def pulse(time_value: float, center: float, half_width: float) -> float:
    """A smooth 0 -> 1 -> 0 bump, 1 at ``center`` and 0 outside ``half_width``."""
    offset = (time_value - center) / half_width
    if abs(offset) >= 1.0:
        return 0.0
    return 0.5 + 0.5 * math.cos(math.pi * offset)


def empty_pose() -> dict[str, float]:
    """A pose with every channel at zero (the setup pose)."""
    return {
        "root_x": 0.0, "root_y": 0.0, "hip": 0.0, "head": 0.0,
        "arm_l": 0.0, "arm_r": 0.0, "leg_l": 0.0, "leg_r": 0.0,
    }


def leg_compensation(root_x: float) -> float:
    """World rotation that keeps the feet under a pelvis moved by ``root_x``."""
    return -math.degrees(math.atan2(root_x, LEG_LENGTH))


def beat_bounce(time_value: float, amplitude: float) -> float:
    """A dip on every beat (negative is down, Y is up)."""
    return -amplitude * (0.5 + 0.5 * math.cos(TAU * time_value / BEAT_PERIOD))


# ------------------------------ section poses -------------------------------


def still_pose(time_value: float) -> dict[str, float]:
    """A barely moving standing pose, so the animation starts from rest."""
    breath = math.sin(TAU * time_value / 4.0)
    pose = empty_pose()
    pose["hip"] = 0.6 * breath
    pose["head"] = -0.6 * breath
    pose["arm_l"] = -1.5 - 1.0 * breath
    pose["arm_r"] = 1.5 + 1.0 * breath
    return pose


def groove_pose(
    time_value: float,
    *,
    sway: float = 40.0,
    arm_base: float = 8.0,
    arm_swing: float = 16.0,
    bounce: float = 6.0,
    hip_sway: float = 4.5,
    head_sway: float = 2.5,
    leg_sway: float = 2.0,
) -> dict[str, float]:
    """Hips from side to side, arms swinging in counter-time, feet planted."""
    swing = math.sin(TAU * time_value / SWAY_PERIOD)
    # The arms follow a little behind the hips, as a real body's do.
    arm_swing_now = math.sin(TAU * time_value / SWAY_PERIOD - 0.22)
    pose = empty_pose()
    pose["root_x"] = sway * swing
    pose["root_y"] = beat_bounce(time_value, bounce)
    pose["hip"] = -hip_sway * swing + 1.0 * math.sin(2.0 * TAU * time_value / SWAY_PERIOD)
    pose["head"] = (
        head_sway * math.sin(TAU * time_value / SWAY_PERIOD - 0.35)
        + 1.2 * math.sin(TAU * time_value / BEAT_PERIOD + 0.5)
    )
    pose["arm_l"] = -arm_base - arm_swing * arm_swing_now
    pose["arm_r"] = arm_base - arm_swing * arm_swing_now
    compensation = leg_compensation(pose["root_x"])
    pose["leg_l"] = compensation + leg_sway * swing
    pose["leg_r"] = compensation + leg_sway * swing
    return pose


def intro_groove(time_value: float) -> dict[str, float]:
    """The first, relaxed groove."""
    return groove_pose(time_value)


def energetic_groove(time_value: float) -> dict[str, float]:
    """The same groove with more amplitude and a pumping arm swing."""
    return groove_pose(
        time_value, sway=46.0, arm_base=14.0, arm_swing=26.0,
        bounce=8.0, hip_sway=5.5, head_sway=3.0, leg_sway=2.5,
    )


def overhead_pose(time_value: float) -> dict[str, float]:
    """Both arms up in a V, waving them alternately above the head."""
    swing = math.sin(TAU * time_value / SWAY_PERIOD)
    wave = math.sin(TAU * time_value / 1.0)
    pose = empty_pose()
    pose["root_x"] = 26.0 * swing
    pose["root_y"] = beat_bounce(time_value, 6.0)
    pose["hip"] = -3.0 * swing + 1.2 * math.sin(TAU * time_value / 1.0 + 0.6)
    pose["head"] = 4.0 * swing + 1.0 * math.sin(TAU * time_value / BEAT_PERIOD + 0.3)
    # Negative raises the left arm, positive raises the right one. The two wave
    # in opposition: one stretches higher while the other comes down a little.
    pose["arm_l"] = -104.0 - 13.0 * wave
    pose["arm_r"] = 104.0 - 13.0 * wave
    compensation = leg_compensation(pose["root_x"])
    pose["leg_l"] = compensation - 1.5 * swing
    pose["leg_r"] = compensation - 1.5 * swing
    return pose


# Side steps: a raised leg swings outward on the beat when the weight is on the
# other leg. The bumps are smooth, so the foot lifts and lands without a snap.
LIFT_ANGLE = 20.0
LIFT_HALF_WIDTH = 0.42
LIFT_LEFT = (10.5, 12.5)
LIFT_RIGHT = (11.5, 13.5)


def side_step_pose(time_value: float) -> dict[str, float]:
    """Arms wide, weight shifting, alternate legs lifted to the side."""
    swing = math.sin(TAU * time_value / SWAY_PERIOD)
    pose = empty_pose()
    pose["root_x"] = 38.0 * swing
    pose["root_y"] = beat_bounce(time_value, 7.0)
    pose["hip"] = -4.5 * swing
    pose["head"] = 3.0 * swing + 1.0 * math.sin(TAU * time_value / BEAT_PERIOD)
    pose["arm_l"] = -52.0 - 7.0 * math.sin(TAU * time_value / 1.0)
    pose["arm_r"] = 52.0 - 7.0 * math.sin(TAU * time_value / 1.0)
    compensation = leg_compensation(pose["root_x"])
    lift_l = LIFT_ANGLE * sum(pulse(time_value, center, LIFT_HALF_WIDTH) for center in LIFT_LEFT)
    lift_r = LIFT_ANGLE * sum(pulse(time_value, center, LIFT_HALF_WIDTH) for center in LIFT_RIGHT)
    pose["leg_l"] = compensation + 1.5 * swing - lift_l
    pose["leg_r"] = compensation + 1.5 * swing + lift_r
    return pose


def wind_down_pose(time_value: float) -> dict[str, float]:
    """Center the body, settle a little and open the legs for the split."""
    swing = math.sin(TAU * time_value / SWAY_PERIOD)
    pose = empty_pose()
    pose["root_x"] = 12.0 * swing
    pose["root_y"] = -8.0 + beat_bounce(time_value, 4.0)
    pose["hip"] = -1.5 * swing
    pose["head"] = 1.0 * swing
    pose["arm_l"] = -56.0 - 5.0 * math.sin(TAU * time_value / 1.0)
    pose["arm_r"] = 56.0 - 5.0 * math.sin(TAU * time_value / 1.0)
    compensation = leg_compensation(pose["root_x"])
    pose["leg_l"] = compensation - 14.0
    pose["leg_r"] = compensation + 14.0
    return pose


# Split: the legs slide out to horizontal while the pelvis goes down, the arms
# rise into a V for balance, and a small damped settle ends the move.
SPLIT_START = 17.5
SPLIT_END = 19.3
SPLIT_LEAN = 2.5
FINAL_HEAD_TILT = 3.0


def split_hip_height(descent: float) -> float:
    """The pelvis height that keeps both feet on the floor while the legs spread.

    The legs are straight bones, so lowering the pelvis without changing the leg
    angles would push the feet through the floor. This walks the hip down along the
    arc the leg tips trace, from the stance to the held split.
    """
    world_l = LEG_L_SETUP + (-14.0 + (LEG_L_SPLIT + 14.0) * descent)
    world_r = LEG_R_SETUP + (14.0 + (LEG_R_SPLIT - 14.0) * descent)
    hip_l = TIP_L_Y - LEG_L_OFFSET_Y - LEG_L_LENGTH * math.sin(math.radians(world_l))
    hip_r = TIP_R_Y - LEG_R_OFFSET_Y - LEG_R_LENGTH * math.sin(math.radians(world_r))
    return 0.5 * (hip_l + hip_r)


def split_pose(time_value: float) -> dict[str, float]:
    """Sit down into a full side split and hold it."""
    descent = smootherstep((time_value - SPLIT_START) / (SPLIT_END - SPLIT_START))
    settle_time = max(0.0, time_value - SPLIT_END)
    # The settle starts with zero velocity, so the landing does not pop.
    settle_ramp = smootherstep(settle_time / 0.15)
    settle = settle_ramp * math.exp(-5.0 * settle_time) * math.sin(12.0 * settle_time)
    pose = empty_pose()
    pose["root_x"] = 0.0
    pose["root_y"] = split_hip_height(descent) - HIP_SETUP_Y - 4.0 * settle
    pose["hip"] = -SPLIT_LEAN * math.sin(math.pi * descent) + 0.8 * settle
    pose["head"] = (
        2.0 * math.sin(math.pi * descent)
        + FINAL_HEAD_TILT * smootherstep((time_value - 19.5) / 0.4)
    )
    pose["arm_l"] = -58.0 - 12.0 * descent + 2.0 * settle
    pose["arm_r"] = 58.0 + 12.0 * descent - 2.0 * settle
    pose["leg_l"] = -14.0 + (LEG_L_SPLIT + 14.0) * descent - 1.5 * settle
    pose["leg_r"] = 14.0 + (LEG_R_SPLIT - 14.0) * descent + 1.5 * settle
    return pose


# ----------------------------- blending -------------------------------------


@dataclass(frozen=True)
class Section:
    """A pose function active during a window, cross-faded at both ends."""

    pose: Callable[[float], dict[str, float]]
    enter: float
    full: float
    leave: float
    gone: float

    def weight(self, time_value: float) -> float:
        if self.full > self.enter:
            fade_in = smootherstep((time_value - self.enter) / (self.full - self.enter))
        else:
            fade_in = 1.0 if time_value >= self.enter else 0.0
        if self.gone > self.leave:
            fade_out = 1.0 - smootherstep((time_value - self.leave) / (self.gone - self.leave))
        else:
            fade_out = 1.0 if time_value <= self.leave else 0.0
        return fade_in * fade_out


SECTIONS = (
    Section(still_pose,       0.0, 0.0, 0.5, 1.6),
    Section(intro_groove,     0.8, 1.8, 5.4, 6.4),
    Section(overhead_pose,    5.2, 6.4, 9.0, 10.0),
    Section(side_step_pose,   8.8, 10.0, 13.6, 14.6),
    Section(energetic_groove, 13.4, 14.6, 16.2, 17.0),
    Section(wind_down_pose,   16.0, 17.0, 17.8, 18.6),
    Section(split_pose,       17.6, 18.4, DURATION, DURATION),
)


def blended_pose(time_value: float) -> dict[str, float]:
    """All section poses mixed by their smooth weights (world-space intents)."""
    totals = empty_pose()
    weight_sum = 0.0
    for section in SECTIONS:
        weight = section.weight(time_value)
        if weight <= 0.0:
            continue
        pose = section.pose(time_value)
        for channel, value in pose.items():
            totals[channel] += weight * value
        weight_sum += weight
    if weight_sum <= 0.0:
        return empty_pose()
    return {channel: value / weight_sum for channel, value in totals.items()}


def local_keys(time_value: float) -> dict[str, dict[str, dict[str, float]]]:
    """Convert the blended world intents to the local values to key."""
    pose = blended_pose(time_value)
    hip = pose["hip"]
    return {
        ROOT: {"translate": {"x": pose["root_x"], "y": pose["root_y"]}},
        HIP: {"rotate": {"value": hip}},
        HEAD: {"rotate": {"value": pose["head"] - hip}},
        ARM_L: {"rotate": {"value": pose["arm_l"] - hip}},
        ARM_R: {"rotate": {"value": pose["arm_r"] - hip}},
        LEG_L: {"rotate": {"value": pose["leg_l"] - hip}},
        LEG_R: {"rotate": {"value": pose["leg_r"] - hip}},
    }


def dance_operations() -> list[dict]:
    """The ``batch_edit`` operations that build the whole animation."""
    operations: list[dict] = [
        {"op": "add_animation", "name": ANIMATION, "duration": DURATION}
    ]
    steps = int(round(DURATION * SAMPLE_HZ))
    for step in range(steps + 1):
        time_value = round(step / SAMPLE_HZ, 6)
        for bone, channels in local_keys(time_value).items():
            for channel, values in channels.items():
                operation = {
                    "op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                    "channel": channel, "time": time_value, "curve": "linear",
                }
                operation.update({name: round(value, 2) for name, value in values.items()})
                operations.append(operation)
    return operations


# ----------------------------- verification ---------------------------------


def bone_text(pose: dict, names: tuple[str, ...]) -> str:
    """World positions and rotations of the given bones, formatted."""
    parts = []
    for name in names:
        bone = mcp.pose_bone(pose, name)
        if bone:
            parts.append(f"{name}=({bone['x']:.0f}, {bone['y']:.0f}) {bone['rotation']:.1f} deg")
    return "; ".join(parts)


def split_problems(pose: dict) -> list[str]:
    """Check the held split: both legs horizontal and the pelvis on the floor."""
    problems = []
    for name, expected in ((LEG_L, -180.0), (LEG_R, 0.0)):
        bone = mcp.pose_bone(pose, name)
        if bone is None:
            problems.append(f"{name} is missing from the pose")
            continue
        error = abs(((bone["rotation"] - expected + 180.0) % 360.0) - 180.0)
        if error > 2.0:
            problems.append(
                f"{name} is at {bone['rotation']:.1f} deg, expected about {expected:.0f}"
            )
    hip = mcp.pose_bone(pose, HIP)
    expected_y = HIP_SETUP_Y - SPLIT_DROP
    if hip is None:
        problems.append(f"{HIP} is missing from the pose")
    elif abs(hip["y"] - expected_y) > 6.0:
        problems.append(f"{HIP} is at y={hip['y']:.1f}, expected about {expected_y:.1f}")
    return problems


def verify(client: mcp.RigbyMcpClient, output: str, log: mcp.RunLog) -> bool:
    """Log key counts, a few poses, the held split and the project health."""
    anim = client.call_tool("describe_animation", {"path": output, "animation": ANIMATION})
    duration = anim.get("duration") or 0.0
    log.info(f"Duration after the edit: {duration:g} s")
    for timeline in anim.get("bones", []):
        counts = {
            channel: len(timeline.get("keys", {}).get(channel) or [])
            for channel in ("rotate", "translate")
        }
        log.info(
            f"  {timeline['bone']}: {counts['rotate']} rotate, "
            f"{counts['translate']} translate key(s)"
        )

    timestamps = (
        (0.0, "start"), (6.0, "arms overhead"), (10.5, "side step"),
        (14.0, "groove"), (17.0, "wind down"), (19.95, "split"),
    )
    log.info("Poses in world space (Y up, degrees):")
    last_pose: dict = {}
    for time_value, title in timestamps:
        last_pose = client.call_tool(
            "sample_pose", {"path": output, "animation": ANIMATION, "time": time_value}
        )
        bones = bone_text(last_pose, (HIP, HEAD, ARM_L, ARM_R, LEG_L, LEG_R))
        log.info(f"  t={time_value:g} s, {title}: {bones}")

    problems = split_problems(last_pose)
    log.info(f"Split check: {'OK' if not problems else 'FAILED'}")
    for message in problems:
        log.error(f"  {message}")

    report = client.call_tool("validate_project", {"path": output})
    errors = report.get("errors") or []
    log.info(f"validate_project: {'OK' if not errors else 'FAILED'}")
    for message in errors:
        log.error(f"  {message}")
    for message in report.get("warnings") or []:
        log.detail(f"  warning: {message}")
    return not errors and not problems


def add_dance(
    client: mcp.RigbyMcpClient, source: str, output: str, log: mcp.RunLog,
) -> bool:
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

    operations = dance_operations()
    log.info(f"Building {len(operations) - 1} keys at {SAMPLE_HZ:g} keys/s...")
    started = time.perf_counter()
    client.call_tool("batch_edit", {
        "path": source, "output": output, "overwrite": True, "operations": operations,
    })
    elapsed = time.perf_counter() - started
    log.info(f"Created animation '{ANIMATION}' ({DURATION:g} s) in {elapsed:.1f} s")

    return verify(client, output, log)


# ----------------------------------- run ------------------------------------


def run(log: mcp.RunLog, args: argparse.Namespace) -> int:
    log.info("=== Rigby2D MCP: dance, ending in a split ===")
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

    with mcp.connect(log, WORKSPACE, client_name="rigby-dance-demo", client_version="1.0") as client:
        ok = add_dance(client, source, target, log)

    if not ok:
        log.error(f"The output project has problems; see {log.path}")
        return 1
    log.info(f"Done. Output written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add a 20-second dance that ends in a split to a Rigby2D stick figure.",
    )
    mcp.add_project_arguments(parser, default_input="input/human.rigby")
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
