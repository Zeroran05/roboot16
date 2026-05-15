# Grounded Retargeting for Roboot16

This is a side-path implementation inspired by PHUMA's retargeting constraints.
The original GMR files are left untouched. The new path adds ground-aware
post-processing to each IK frame so the retargeted Roboot16 motion is less
likely to hover, skate, or penetrate the floor during stance.

## Files

- `general_motion_retargeting/motion_retarget_grounded.py`
  - Wraps the existing `GeneralMotionRetargeting`.
  - Runs the original `mink.solve_ik()` frame tasks first.
  - Infers stance feet from human foot height and velocity, similar to PHUMA's
    contact-weighted losses.
  - Optionally adjusts support ankle pitch/roll in a small local search so
    fore/rear/heel foot collision geoms have closer heights.
  - Calibrates the sole plane from each foot geom's local bottom height, so a
    heel box that is modeled higher than a forefoot box does not confuse the
    foot-flat objective.
  - Projects the root in z: upward for penetration, and gently downward for
    stance-foot hover.
  - Optionally locks stance-foot XY position to reduce foot skating.
  - Runs an export-time whole-sequence foot-flattening pass for support frames.
  - Runs an export-time whole-sequence root-z smoother so late contact
    detection does not look like a sudden downward stomp.
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
  --ground_mode support \
  --contact_source hybrid \
  --support_height_mode mean \
  --foot_flatten_iterations 4 \
  --foot_flatten_step 0.02 \
  --save_path retargeting_data/roboot16/walk1_subject1_grounded.pkl
```

Useful tuning flags:

```bash
--ground_height 0.0
--ground_clearance 0.005
--foot_bodies left_ankle_roll_link,right_ankle_roll_link
--foot_geom_groups 3
--ground_mode support
--contact_source hybrid
--support_height_mode mean
--human_contact_height_threshold 0.06
--human_contact_speed_threshold 0.35
--foot_flatten_iterations 4
--foot_flatten_step 0.02
--root_z_pull_down_alpha 0.45
--max_root_z_step_down 0.025
--max_root_z_step_up 0.08
--sequence_foot_flatten_contact_pre_roll 8
--sequence_ground_window 11
--sequence_ground_passes 2
--sequence_ground_max_step_down 0.006
--sequence_ground_contact_pre_roll 6
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

Disable ankle pitch/roll flattening if it changes the original pose too much:

```bash
--disable_foot_flattening
```

Disable the export-time whole-sequence smoother if you only want to debug the
online per-frame projection:

```bash
--disable_sequence_ground_smoothing
```

## How The Constraint Works

The pipeline keeps the existing GMR IK objective:

1. Human keypoints are scaled and offset by the existing IK config.
2. `mink.FrameTask` targets are solved exactly as before.
3. Stance feet are inferred from the source human motion. A foot is treated as
   stance when it is close to its running minimum height and moving slowly.
4. For stance feet, a small coordinate search adjusts ankle pitch/roll so
   Roboot16's `fore`, `rear`, and `heel` collision boxes become more level.
   The loss uses calibrated sole heights, because Roboot16's heel collision box
   is modeled higher than the forefoot collision box in local foot coordinates.
5. The grounded retargeter measures the configured robot foot collision geoms.
   If any foot is below `ground_height + ground_clearance`, it shifts the root
   upward. If a stance foot is hovering, it gently pulls the root downward,
   clipped so the lowest foot point cannot go below the floor.
6. If stance locking is enabled, the support foot XY anchor is held by shifting
   the root in XY, which reduces visible foot skating.
7. Before export, support frames are expanded by a small pre/post-roll and
   foot-flattening is run again over the whole sequence. This catches support
   frames that were missed by the online pass.
8. Before export, the root-z trajectory is recomputed over the whole sequence.
   This spreads contact corrections over nearby frames and limits downward
   velocity, which reduces the "hover, then suddenly step down" artifact.
9. The final qvel is recomputed from the previous and corrected qpos using
   MuJoCo's `mj_differentiatePos`.

This is intentionally not a full replacement for PHUMA's differentiable
optimization. It is a conservative projection layer around GMR's existing IK,
which makes it easier to debug and safer for the current Roboot16 pipeline.

## Foot Model Requirements

For Roboot16, the current XML already has the required collision geoms:

- `left_ankle_roll_link`
- `right_ankle_roll_link`
- foot collision geoms in MuJoCo geom group `3`
- named foot collision geoms:
  `*_foot_fore_collision`, `*_foot_rear_collision`, `*_foot_heel_collision`
- ankle joints named `*_ankle_pitch_joint` and `*_ankle_roll_joint`

For a new robot, adapt these parts:

- The foot body names must be stable and passed with `--foot_bodies`, or be
  auto-detectable from names containing `foot` or `ankle` + `roll`.
- Each foot body should contain collision geoms that approximate the true sole.
  Boxes are best; mesh geoms also work.
- For better foot-flat behavior, split the sole into at least two named parts,
  such as `fore/toe` and `rear/heel`. One single foot collision box can prevent
  penetration, but it cannot diagnose forefoot-only or heel-only contact well.
- The sole collision geoms should be in a known MuJoCo geom group. For Roboot16
  this is group `3`, so the default `--foot_geom_groups 3` is correct.
- The lowest point of the foot collision should represent the actual contact
  plane. If visual meshes hang below collision geoms, keep using collision-only
  groups.
- If you want the script to actively flatten a support foot, the robot should
  expose side-specific ankle pitch/roll joints or an equivalent small set of
  foot-orientation joints.
- The world floor height should match `--ground_height`. Current Roboot16
  retargeting assumes ground z is `0.0`.
- If the robot has heel/toe keypoint bodies, they can be used for diagnostics,
  but this grounded GMR path only requires collision geoms.

## Tuning Notes

- Increase `--ground_clearance` if you still see tiny numerical penetration.
- Decrease `--max_root_z_correction` if bad source data causes large root pops.
- Increase `--root_z_pull_down_alpha` if stance feet still hover, but keep
  `--max_root_z_step_down` small to avoid sudden sinking.
- If you still see a sudden down-step, first lower
  `--sequence_ground_max_step_down` to `0.003` to `0.004`, then increase
  `--sequence_ground_window` to `15`.
- If the foot still hangs in the air for too long before contact, increase
  `--sequence_ground_contact_pre_roll` to `8` or `10`.
- If the heel is still high, first increase `--foot_flatten_iterations` to `5`
  or `6`, then reduce `--foot_flatten_step` to `0.015` so the ankle correction
  is stronger but less jumpy.
- Keep calibrated sole enabled unless you intentionally changed the XML so
  fore/rear/heel collision bottoms are exactly coplanar.
- Lower `--human_contact_speed_threshold` if swing feet are accidentally treated
  as stance feet.
- Increase `--foot_flatten_iterations` to `3` or `4` if the heel is still high.
  If the ankle starts looking unnatural, decrease `--foot_flatten_step` or
  increase `--foot_flatten_tracking_weight`.
- Lower `--stance_speed_threshold` if robot-inferred contacts are accidentally
  locked.
- Lower `--stance_xy_blend` if stance correction makes the pelvis path too stiff.
- Use `--foot_geom_groups all` only if the collision group is missing or wrong.

## Limits

This is still a GMR-compatible projection layer, not PHUMA's full
differentiable sequence optimization. Root z can remove hover/penetration, and
ankle pitch/roll can reduce forefoot-only contact, but perfect full-sole contact
requires the robot model to have usable foot orientation joints and correctly
placed sole collision geoms.
