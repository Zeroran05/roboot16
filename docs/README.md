# RoBoot16 AMP Export Project

This folder is a self-contained bundle for the `roboot16` motion-to-AMP workflow we built in this repo.

## Structure

```text
roboot16_amp_project/
├── assets/
│   └── Roboot1.6/
│       ├── config/
│       │   └── joint_names_Roboot1.6.yaml
│       └── xml/
│           ├── roboot16_1.xml
│           └── scene_1.xml
├── data/
│   ├── source_pkl/
│   │   └── run1_subject2.pkl
│   ├── segments/
│   │   ├── run1_1.pkl
│   │   ├── run1_2.pkl
│   │   ├── run1_3.pkl
│   │   ├── run1_4.pkl
│   │   └── run1_5.pkl
│   └── amp_expert/
│       ├── run1_1.txt
│       ├── run1_1_debug.pkl
│       ├── ...
│       ├── run1_5.txt
│       └── run1_5_debug.pkl
├── docs/
│   └── README.md
└── scripts/
    ├── export_roboot16_amp_expert.py
    └── split_robot_motion.py
```

## What Each Part Does

- `scripts/export_roboot16_amp_expert.py`
  Replays a `roboot16` GMR motion PKL in MuJoCo, computes root-local foot positions, and exports a TianKung-style AMP expert `txt`.

- `scripts/split_robot_motion.py`
  Splits a long GMR motion PKL into multiple PKL segments by frame range.

- `assets/Roboot1.6/xml/roboot16_1.xml`
  The MuJoCo model used for forward kinematics and foot position extraction.

- `data/source_pkl/run1_subject2.pkl`
  Original full motion.

- `data/segments/run1_*.pkl`
  Segmented motion clips.

- `data/amp_expert/run1_*.txt`
  Exported AMP expert files.

- `data/amp_expert/run1_*_debug.pkl`
  Debug PKLs containing the original motion plus extracted foot positions:
  `foot_link_body_pos` is root-local.
  `foot_link_body_pos_world` is world-frame.

## AMP Expert Frame Layout

Each frame in `run1_*.txt` is 30-dimensional:

```text
right_leg_q(6),
left_leg_q(6),
right_leg_dq(6),
left_leg_dq(6),
left_foot_pos_local(3),
right_foot_pos_local(3)
```

Joint order is:

```text
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
```

This order was verified against the MuJoCo model order:
- `qpos[7:]`
- `qvel[6:]`
- actuator-to-joint order

## Environment

Required Python packages:

- `mujoco`
- `numpy`

If you want visualization on macOS, use `mjpython`.

## Common Commands

Run AMP export headless:

```bash
python scripts/export_roboot16_amp_expert.py \
  --input_pkl data/segments/run1_1.pkl \
  --output_txt data/amp_expert/run1_1.txt \
  --output_debug_pkl data/amp_expert/run1_1_debug.pkl \
  --robot_xml assets/Roboot1.6/xml/roboot16_1.xml
```

Run AMP export with visualization:

```bash
mjpython scripts/export_roboot16_amp_expert.py \
  --input_pkl data/segments/run1_1.pkl \
  --output_txt data/amp_expert/run1_1.txt \
  --output_debug_pkl data/amp_expert/run1_1_debug.pkl \
  --robot_xml assets/Roboot1.6/xml/roboot16_1.xml \
  --visualize
```

Split a full motion:

```bash
python scripts/split_robot_motion.py \
  --input data/source_pkl/run1_subject2.pkl \
  --output_dir data/segments \
  --segment 3362:3415 \
  --segment 3465:3548
```

## Migration Note

For server migration, copy this whole folder as-is. The export script no longer depends on
`general_motion_retargeting.params`; it can work directly from the bundled `roboot16_1.xml`
via `--robot_xml`.
