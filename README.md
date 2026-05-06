# Roboot16 AMP Project

This directory now contains a standalone Isaac Lab extension package for your Roboot16 AMP task.

## What was migrated

- Project-local Roboot16 asset config package under `src/roboot16_amp_project/assets/robots/`
- Project-local AMP task registration under `src/roboot16_amp_project/tasks/roboot16_amp/`
- Project-local non-AMP Roboot16 task registration under `src/roboot16_amp_project/tasks/roboot16/`
- Project-local AMP dataset location: `data/amp_npy/`
- Project-local training launcher: `scripts/train_amp.py`
- Project-local RSL-RL training launcher: `scripts/train_roboot16_rsl_rl.py`
- Project-local play launcher: `scripts/play_amp.py`

## Install

No installation is required for basic usage.

The launcher scripts automatically add `src/` to `PYTHONPATH`, so you can run the project directly from the
Isaac Lab environment.

If you still want an editable install, use a virtual environment or your Isaac Lab Python environment instead of
the system `python3`.

## Train AMP

Run from the Isaac Lab repository root:

```bash
cd /root/isaaclab
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --headless
```

If you want to override the task explicitly:

```bash
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --headless
```

## Train Roboot16 With RSL-RL

```bash
cd /root/isaaclab
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Flat-Project-v0 \
  --headless
```

If you want to override the task explicitly:

```bash
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Flat-Project-v0 \
  --headless
```

## Play AMP

```bash
cd /root/isaaclab
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/amp_rsl_rl/play.py \
  --task Isaac-Roboot16-AMP-Flat-Project-Play-v0 \
  --checkpoint /path/to/model.pt
```

## Play Roboot16

```bash
cd /root/isaaclab
./isaaclab.sh -p roboot16_amp_project/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Roboot16-Flat-Project-Play-v0 \
  --checkpoint /path/to/model.pt
```

## AMP dataset

The default AMP training config now reads from:

```text
data/amp_npy/run1_1.npy
data/amp_npy/run1_2.npy
data/amp_npy/run1_3.npy
data/amp_npy/run1_4.npy
data/amp_npy/run1_5.npy
```

If you regenerate debug pickles and want to rebuild the `.npy` dataset:

```bash
cd /root/isaaclab/roboot16_amp_project
python3 scripts/convert_amp_debug_to_npy.py data/amp_expert/run1_1_debug.pkl data/amp_npy/run1_1.npy
```

## Logs

Both AMP and RSL-RL runs now write under this project directory instead of the Isaac Lab root:

```text
roboot16_amp_project/logs/amp_rsl_rl/...
roboot16_amp_project/logs/rsl_rl/...
```
