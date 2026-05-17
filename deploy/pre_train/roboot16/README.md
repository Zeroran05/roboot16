# Pretrained Policies for Classic Deploy

This folder stores deploy artifacts for the classic Roboot16 MuJoCo sim2sim flow.

## Expected File

Default artifact name:

```text
deploy/pre_train/roboot16/policy.pt
```

This is a TorchScript / JIT policy used by:

```text
deploy/deploy_mujoco/configs/roboot16.yaml
```

## How to Export

From the Isaac Lab root:

```bash
python roboot16/deploy/scripts/export_roboot16_policy_jit.py \
  --checkpoint /path/to/IsaacLab/roboot16/logs/rsl_rl/<run>/model_<iter>.pt \
  --output /path/to/IsaacLab/roboot16/deploy/pre_train/roboot16/policy.pt
```

## Notes

- Use this folder for classic velocity / non-mimic deploy artifacts.
- Mimic deploy uses ONNX files instead, and those should go under `deploy/pre_train/roboot16_mimic/`.
