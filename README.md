# Roboot16 Project

`roboot16` is a standalone Git repository nested inside your Isaac Lab workspace. It currently bundles several connected workflows around the same robot:

- AMP locomotion training
- Velocity-tracking RSL-RL training
- Mimic / motion-tracking training and MuJoCo deployment
- CSV motion visualization and editing via `CSVEditor/`

## Repository Layout

```text
roboot16/
├── CSVEditor/                      # motion CSV visualization and editing tool
├── assets/                         # Roboot16 USD / MuJoCo assets
├── data/
│   ├── amp_expert/                 # AMP expert txt / debug data
│   └── mimic/                      # CSV / NPZ motion data for mimic
├── deploy/
│   ├── deploy_mujoco/              # sim2sim scripts
│   ├── pre_train/                  # exported policies
│   └── scripts/                    # export / validation helpers
├── docs/                           # project notes for packaging / workflow
├── GMR/                            # motion retargeting / export helpers
├── logs/                           # local training outputs
├── scripts/
│   ├── mimic/                      # csv<->npz / replay tools
│   └── reinforcement_learning/     # train / play entry points
├── src/
│   ├── amp_rsl_rl/                 # local AMP runner implementation
│   └── roboot16_amp_project/       # assets, tasks, task registration
├── COMMON_COMMANDS.md              # frequently used commands
└── 笔记.md                          # personal notes, intentionally left untouched
```

## Registered Tasks

Current task IDs registered under `src/roboot16_amp_project/tasks/`:

| Type | Train Task | Play Task |
| --- | --- | --- |
| Flat velocity | `Isaac-Roboot16-Flat-Project-v0` | `Isaac-Roboot16-Flat-Project-Play-v0` |
| AMP flat | `Isaac-Roboot16-AMP-Flat-Project-v0` | `Isaac-Roboot16-AMP-Flat-Project-Play-v0` |
| AMP high-speed | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0` | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-Play-v0` |
| Mimic tracking | `Isaac-Roboot16-Mimic-Project-v0` | `Isaac-Roboot16-Mimic-Project-Play-v0` |

## Working Directory

Most Isaac Lab launch commands are intended to be run from the Isaac Lab root:

```bash
cd /path/to/IsaacLab
```

Git operations for this project should be run inside the nested repo:

```bash
cd /path/to/IsaacLab/roboot16
```

## Common Workflows

### 1. Train AMP

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --headless \
  --video \
  --video_interval 2000
```

### 2. Train High-Speed AMP

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0 \
  --headless \
  --video \
  --video_interval 2000
```

### 3. Train Mimic

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Mimic-Project-v0 \
  --headless \
  --video \
  --video_interval 8000
```

### 4. Play and Export a Policy

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Roboot16-Mimic-Project-Play-v0 \
  --checkpoint /path/to/roboot16/logs/rsl_rl/roboot16_mimic_project/<run>/model_<iter>.pt
```

The play script exports artifacts under:

```text
roboot16/logs/rsl_rl/<experiment>/<run>/exported/
```

For mimic policies this includes `policy.onnx` with metadata used by the MuJoCo deploy script.

### 5. Run MuJoCo Mimic sim2sim

```bash
mjpython roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py \
  --onnx roboot16/deploy/pre_train/roboot16_mimic/policy5.onnx \
  --motion-npz roboot16/data/mimic/walk1_subject1_test1_30s.npz \
  --model roboot16/assets/Roboot1.6/xml/scene_1.xml
```

## Logs

Training outputs are written locally under:

```text
roboot16/logs/amp_rsl_rl/
roboot16/logs/rsl_rl/
```

## Related Docs

- `COMMON_COMMANDS.md`: day-to-day command reference
- `deploy/README.md`: deployment and MuJoCo notes
- `docs/README.md`: project structure and data flow summary
