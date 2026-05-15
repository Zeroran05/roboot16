# Grounded Retargeting for Roboot16

This is a side-path implementation inspired by PHUMA's retargeting constraints.
The original GMR files are left untouched. The new path adds ground-aware
post-processing to each IK frame so the retargeted Roboot16 motion is less
likely to penetrate the floor.

## Files

- `general_motion_retargeting/motion_retarget_grounded.py`
  - Wraps the existing `GeneralMotionRetargeting`.
  - Runs the original `mink.solve_ik()` frame tasks first.
  - Projects the root upward if the configured foot geoms go below ground.
  - Optionally locks stance-foot XY position to reduce foot skating.
  - Recomputes qvel after projection so exported velocity matches qpos.
- `scripts/bvh_to_robot_grounded.py`
  - BVH/LAFAN1 entry point using the grounded retargeter.
  - Exports the same PKL fields as `scripts/bvh_to_robot.py`.

## Recommended Roboot16 Command

Run from `roboot16/GMR`:

```bash
mjpython scripts/bvh_to_robot_grounded.py \
  --bvh_file lafan1/walk1_subject1.bvh \
  --format lafan1 \
  --robot roboot16 \
  --motion_fps 30 \
  --headless \
  --save_path retargeting_data/roboot16/walk1_subject1_grounded.pkl
```

Useful tuning flags:

```bash
--ground_height 0.0
--ground_clearance 0.005
--foot_bodies left_ankle_roll_link,right_ankle_roll_link
--foot_geom_groups 3
--stance_height_threshold 0.04
--stance_speed_threshold 0.35
--stance_xy_blend 0.75
--max_root_z_correction 0.25
--max_stance_xy_correction 0.04
```

Disable stance-foot XY locking when you only want anti-penetration:

```bash
--disable_stance_lock
```

## How The Constraint Works

The pipeline keeps the existing GMR IK objective:

1. Human keypoints are scaled and offset by the existing IK config.
2. `mink.FrameTask` targets are solved exactly as before.
3. After the IK step, the grounded retargeter measures the lowest z of the
   configured robot foot collision geoms.
4. If the foot is below `ground_height + ground_clearance`, it shifts the
   floating root upward by the required amount.
5. If stance locking is enabled, a foot that is near the ground and moving
   slowly becomes a stance foot. Its XY anchor is held by shifting the root
   in XY, which reduces visible foot skating.
6. The final qvel is recomputed from the previous and corrected qpos using
   MuJoCo's `mj_differentiatePos`.

This is intentionally not a full replacement for PHUMA's differentiable
optimization. It is a conservative projection layer around GMR's existing IK,
which makes it easier to debug and safer for the current Roboot16 pipeline.

## Foot Model Requirements

For Roboot16, the current XML already has the required collision geoms:

- `left_ankle_roll_link`
- `right_ankle_roll_link`
- foot collision geoms in MuJoCo geom group `3`

For a new robot, adapt these parts:

- The foot body names must be stable and passed with `--foot_bodies`, or be
  auto-detectable from names containing `foot` or `ankle` + `roll`.
- Each foot body should contain collision geoms that approximate the true sole.
  Boxes are best; mesh geoms also work.
- The sole collision geoms should be in a known MuJoCo geom group. For Roboot16
  this is group `3`, so the default `--foot_geom_groups 3` is correct.
- The lowest point of the foot collision should represent the actual contact
  plane. If visual meshes hang below collision geoms, keep using collision-only
  groups.
- The world floor height should match `--ground_height`. Current Roboot16
  retargeting assumes ground z is `0.0`.
- If the robot has heel/toe keypoint bodies, they can be used for diagnostics,
  but this grounded GMR path only requires collision geoms.

## Tuning Notes

- Increase `--ground_clearance` if you still see tiny numerical penetration.
- Decrease `--max_root_z_correction` if bad source data causes large root pops.
- Lower `--stance_speed_threshold` if swing feet are accidentally locked.
- Lower `--stance_xy_blend` if stance correction makes the pelvis path too stiff.
- Use `--foot_geom_groups all` only if the collision group is missing or wrong.
