---
name: rigby2d-animation
description: 'Use when adding or editing an animation in a Rigby2D .rigby project over the Rigby2D MCP server: planning poses, keying bones, generating smooth procedural motion (dance, walk cycle, gesture), keeping feet planted, and verifying the result with describe_animation, sample_pose, validate_project and sprite sheets.'
---

# Rigby2D animation over MCP

Procedures and hard-won details for animating a `.rigby` project with the Rigby2D MCP
server. Based on `add_arms_up.py` (minimal) and `add_dance.py` (20 s procedural dance
ending in a split) in this repository.

## When to use

- Add, rebuild or extend an animation in `input/*.rigby` and save it to `output/`.
- Generate motion from code (dance, cycles, gestures) rather than posing by hand.
- Diagnose jerky, sliding or floating limbs in an existing animation.

## Non-negotiable rules

1. **Never hand-edit a `.rigby`.** It is a ZIP holding `project.json` plus `images/`;
   every change goes through MCP tools.
2. **`input/` is read only.** Call editing tools with `output` + `overwrite: true`.
   The output folder must already exist — create it before the first call.
3. **Reuse `rigby2d_mcp_utils.py`.** Import it as `mcp`; do not duplicate the transport.
   Take paths through `mcp.add_project_arguments()` / `mcp.project_paths()`, log through
   `mcp.RunLog` (`log.detail` for tool traffic, `log.error` + non-zero exit for failures).
4. **One `batch_edit`.** It loads and saves once and is atomic; a few thousand keys take
   about 0.1 s. Never issue one `tools/call` per key.
5. **Be idempotent.** If the animation exists, `delete_animation` first, then continue
   from the file you just wrote (`source = output`).

## Workflow

### 1. Discover the rig before planning anything

```
describe_rig   -> bones with PARENT, LOCAL x/y, LOCAL rotation, length; slots and images
sample_pose    -> WORLD x/y, rotation, scale of every bone (omit `animation` for setup)
inspect_project-> counts, animation list, film/output settings
```

Record for every bone: parent, local offset, local rotation, world rotation, length.
The difference between local and world is the whole game (see "Local vs world" below).

### 2. Plan in world space, convert to local

Decide where each limb should point **in the world** (e.g. "left leg horizontal at
180 deg"), then convert to the local key value by subtracting the animated parent:

```python
local_value = world_target - hip_animated_rotation      # for every child of the hip
```

`set_bone_key` values are **offsets from the setup pose**: `rotate` is added to the setup
rotation, `translate`/`shear` x/y are added, `scale` x/y multiply. Angles are degrees,
time is seconds, **Y is up**.

### 3. Generate keys in Python, send one batch

```python
operations = [{"op": "add_animation", "name": ANIMATION, "duration": DURATION}]
for step in range(steps + 1):
    t = round(step / SAMPLE_HZ, 6)                    # rounding avoids duplicate times
    for bone, channels in local_keys(t).items():
        for channel, values in channels.items():
            op = {"op": "set_bone_key", "animation": ANIMATION, "bone": bone,
                  "channel": channel, "time": t, "curve": "linear"}
            op.update({k: round(v, 2) for k, v in values.items()})
            operations.append(op)

client.call_tool("batch_edit", {"path": source, "output": output,
                                "overwrite": True, "operations": operations})
```

`SAMPLE_HZ = 20` with `curve: linear` is the sweet spot for procedural motion: dense
enough that easing is invisible, light enough to keep the file small (401 keys per
channel for 20 s, ~340 KB project). Only raise it for very fast motions.

### 4. Verify numerically

- `describe_animation` → duration and **key times only** (arrays of numbers, no values).
- `sample_pose` at the important beats → world angles/positions; assert with tolerance.
- `validate_project` → `errors` must be empty.
- Read the saved values straight from the ZIP when you need the curves:

```python
import json, zipfile
project = json.loads(zipfile.ZipFile(path).read("project.json"))
anim = next(a for a in project["animations"] if a["name"] == NAME)
# anim["bones"] -> [{"bone": <index>, "rotate": [{"time", "value"}],
#                   "translate": [{"time", "value": [x, y]}]}]
```

- **Smoothness check** (catches the defects the eye sees as a "jerk"): differentiate each
  channel twice and look for spikes. A pose that is continuous but not *velocity*
  continuous shows up as a lone acceleration spike. In the finished dance the worst
  values were ~1000 deg/s² on the arms and ~830 units/s² on `root.y`, all at intended
  accents — not at section joints.

### 5. Verify visually

`export_sprite_sheet` with `format: "json"`, `layout: "grid"`, `loop: false`, low `fps`
and small `scale` renders the whole animation as one contact sheet — the fastest way to
review 20 s of motion in a single image.

```python
client.call_tool("export_sprite_sheet", {
    "input": "output/human.rigby", "output_folder": "logs/preview",
    "name": "dance", "format": "json", "layout": "grid",
    "animations": ["dance"], "fps": 2, "scale": 0.3,
    "trim": True, "dedupe": False, "loop": False, "overwrite": True,
})
```

`format: "frames"` writes `name/name_000.png`, ...; `loop` defaults to **true**, which
drops the last frame because it repeats the first. The output folder must exist.

## Local vs world

Every child of the animated hip inherits its rotation. Subtracting the hip angle from a
limb's key keeps the limb's **orientation** fixed in the world, but its **origin** still
swings with the hip (at 10.5 s the hip leaned 4.5 deg and the head origin moved 58 units).
That is what reads as a weight shift — just keep hip sway modest (about ±3..6 deg) or the
torso will visibly translate.

## Making motion look human

- **Never change sections abruptly.** Give each section a pose function plus a window
  `(enter, full, leave, gone)` and blend with a weight = fade-in × fade-out.
- **Use `smootherstep`** `x³(6x²−15x+10)` for every fade and every one-shot move: zero
  first *and* second derivative at both ends, so nothing pops.
- **Layer periodic motion**: sway once per 4 beats, bounce once per beat, head and arms at
  slightly different phases. Pure sync looks mechanical.
- **Add follow-through**: arms lag the hips by ~0.2 rad (`sin(phase − 0.22)`) and the head
  lags further. Limbs never arrive with the body.
- **Use a damped settle, not a stop**: after a landing, `exp(-k·t)·sin(ω·t)` scaled by a
  short smootherstep ramp, so the oscillation starts with zero velocity.
- **Keep a hold at the end**: the last pose must stay readable for ~0.5 s.
- Push amplitude with overshoot (arms above the shoulders, head tilt ~4 deg), and keep
  the neutral breathing motion in the "still" section so the loop does not look frozen.

## Keeping feet on the floor

The legs are straight bones, so translating the pelvis slides the feet. Two corrections,
both used in `add_dance.py`:

```python
# 1. Weight shift: counter-rotate the legs by the angle the pelvis travelled.
compensation = -degrees(atan2(root_x, leg_length))     # legs get += compensation

# 2. Pelvis descent: keep the bone tip at a fixed height while the leg angle changes.
tip_y     = hip_y + offset_y + length * sin(radians(world_angle))
hip_y     = tip_y - offset_y - length * sin(radians(world_angle_now))
```

`hip_y`, `offset_y` and `length` come from `describe_rig`; `offset_y` matters (the legs
attach ~28..33 units below/above the hip head). For a full side split: legs at world
`180` / `0` deg and `root.y = -235` puts the pelvis on the floor with the tips at y ≈ −3.

## This repository's rigs

### `input/human.rigby` — 7 bones, no constraints, no IK, no animations

| bone | parent | local x, y | local rot | world rot | length |
| --- | --- | --- | --- | --- | --- |
| `root` | — | 0, 0 | 0 | 0 | 0 |
| `hip` | root | 0, 241.03 | 90 | 90 | 251.8 |
| `head` | hip | 254.081, 0 | 0 | 90 | 130.18 |
| `leg right` | hip | −12.471, −28.243 | −169.492 | −79.492 | 230.46 |
| `leg left` | hip | −9.019, 32.747 | 166.373 | −103.627 | 234.448 |
| `arm right` | hip | 221.623, −80.026 | −147.529 | −57.529 | 270.071 |
| `arm left` | hip | 227.377, 83.380 | 147.212 | −122.789 | 223.121 |

Conventions: `right` bones sit at **+X**, `left` bones at **−X**; negative rotation
raises `arm left`, positive raises `arm right`; the pelvis is at y = 241 and the feet at
y ≈ −25 in the setup pose. Bone names contain spaces — always quote them. Signs are
mirrored in `test.rigby`, where the `_l` bones sit at +X and positive rotation raises
the left arm — **check `sample_pose` instead of assuming**.

### `input/test.rigby` — "NILA"

39 bones, hand IK `mix` is 0 (animate the arm bones directly), foot IK `mix` is 1 (move
`ik_foot_l/r`), physics drives the braids and jewelry.

## Pitfalls

- `describe_animation` gives key **times**, not values; use `sample_pose` or the ZIP.
- `validate_project` reports missing images and animations without keys — treat any
  `errors` entry as a failure and exit non-zero.
- Batch payloads are huge: `log.detail` of a 2807-operation batch grows the log to ~400 KB.
  Log a summary (operation counts, duration) and keep full payloads behind a flag.
- `export_sprite_sheet` and every editing tool refuse to write into a missing folder and
  refuse to replace files without `overwrite: true`.
- Paths must stay inside the MCP workspace (`--root`); `input == output` is refused.
- `add_animation` keeps the larger of its `duration` argument and the last key time, so a
  declared duration of 20 s survives even when the last key is at 18 s.

## Reference files

- `add_arms_up.py` — minimal one-batch edit, verification, path handling.
- `add_dance.py` — section blending, smootherstep, leg compensation, geometric split.
- `rigby2d_mcp_utils.py` — transport, `RunLog`, path helpers, `pose_bone()`.
- `_tools/rigby2d-develop/docs/mcp-tools.md` — every tool and parameter.
