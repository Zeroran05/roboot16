# Roboot16 Workflow Notes

This folder summarizes how the current `roboot16` repo is organized inside Isaac Lab.

## Current Scope

The repository now contains three connected workflows:

1. AMP locomotion training
2. Mimic motion-tracking training
3. MuJoCo deployment and replay

## Main Project Structure

```text
roboot16/
├── assets/Roboot1.6/
│   ├── config/
│   └── xml/
├── data/
│   ├── amp_expert/
│   └── mimic/
├── deploy/
├── GMR/
├── logs/
├── scripts/
│   ├── mimic/
│   └── reinforcement_learning/
└── src/
    ├── amp_rsl_rl/
    └── roboot16_amp_project/
```

## Task Registration

Task registration lives under:

```text
src/roboot16_amp_project/tasks/
```

Current task groups:

- `roboot16/`: flat velocity task
- `roboot16_amp/`: AMP flat and AMP high-speed tasks
- `roboot16_mimic/`: motion-tracking mimic task

## Data Flow

### AMP

```text
PKL / source motion
-> GMR export helper
-> data/amp_expert/*.txt
-> AMP training
```

### Mimic

```text
CSV motion
-> scripts/mimic/csv_to_npz.py
-> data/mimic/*.npz
-> mimic training
-> exported/policy.onnx
-> MuJoCo mimic deploy
```

## Important Entry Points

AMP train:

```text
scripts/reinforcement_learning/amp_rsl_rl/train.py
```

RSL-RL train/play:

```text
scripts/reinforcement_learning/rsl_rl/train.py
scripts/reinforcement_learning/rsl_rl/play.py
```

Mimic data tools:

```text
scripts/mimic/csv_to_npz.py
scripts/mimic/replay_npz.py
```

MuJoCo deploy:

```text
deploy/deploy_mujoco/deploy_mujoco_roboot16.py
deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py
```

## Git Note

`roboot16/` has its own `.git` directory. If you want to restore or diff files here, run Git commands in
`roboot16/` itself, not only in the outer `IsaacLab/` repository.
