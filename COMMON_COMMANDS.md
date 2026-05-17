# Roboot16 常用指令

本文按当前仓库结构整理常用命令。

默认 Isaac Lab 根目录：

```bash
cd /path/to/IsaacLab
```

默认 `roboot16` Git 仓库目录：

```bash
cd /path/to/IsaacLab/roboot16
```

## 1. Git

查看状态：

```bash
git status
git branch --show-current
git log --oneline -5
```

只恢复 `roboot16` 里的某个文件夹到历史提交：

```bash
git restore --source <commit> src/roboot16_amp_project/tasks/roboot16_amp
```

注意：`roboot16` 是独立 Git 仓库，恢复目录时路径要相对于 `roboot16/`，不要再写一层 `roboot16/...`。

查看某个目录当前改动：

```bash
git diff -- src/roboot16_amp_project/tasks/roboot16_amp
```

## 2. 任务一览

| 任务类型 | Task ID |
| --- | --- |
| 普通速度训练 | `Isaac-Roboot16-Flat-Project-v0` |
| 普通速度播放 | `Isaac-Roboot16-Flat-Project-Play-v0` |
| AMP 训练 | `Isaac-Roboot16-AMP-Flat-Project-v0` |
| AMP 播放 | `Isaac-Roboot16-AMP-Flat-Project-Play-v0` |
| 高速 AMP 训练 | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0` |
| 高速 AMP 播放 | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-Play-v0` |
| Mimic 训练 | `Isaac-Roboot16-Mimic-Project-v0` |
| Mimic 播放 | `Isaac-Roboot16-Mimic-Project-Play-v0` |

## 3. AMP 训练

普通 AMP：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --video \
  --video_interval 2000
```

无界面：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --video \
  --video_interval 2000 \
  --headless
```

高速 AMP：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0 \
  --video \
  --video_interval 2000 \
  --headless
```

恢复训练：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --headless \
  --resume \
  --load_run <run_dir> \
  --checkpoint model_<iter>.pt
```

## 4. 非 AMP / Mimic 训练

普通速度训练：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Flat-Project-v0 \
  --headless
```

Mimic 训练：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Mimic-Project-v0 \
  --headless \
  --video \
  --video_interval 8000
```

Mimic 恢复训练：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Mimic-Project-v0 \
  --headless \
  --resume \
  --load_run <run_dir> \
  --checkpoint model_<iter>.pt \
  --video \
  --video_interval 6000
```

## 5. Play 与导出模型

播放普通策略：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Roboot16-Flat-Project-Play-v0 \
  --checkpoint /path/to/model.pt
```

播放 Mimic 并导出 ONNX：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Roboot16-Mimic-Project-Play-v0 \
  --checkpoint /path/to/roboot16/logs/rsl_rl/roboot16_mimic_project/<run>/model_<iter>.pt \
  --headless \
  --video \
  --video_length 1500
```

导出的文件默认会写到：

```text
roboot16/logs/rsl_rl/roboot16_mimic_project/<run>/exported/
```

如果要拷到部署目录：

```bash
cp /path/to/roboot16/logs/rsl_rl/roboot16_mimic_project/<run>/exported/policy.onnx \
   /path/to/IsaacLab/roboot16/deploy/pre_train/roboot16_mimic/policy.onnx
```

## 6. TensorBoard

查看整个 AMP 实验目录：

```bash
tensorboard --logdir /path/to/IsaacLab/roboot16/logs/amp_rsl_rl/roboot16_amp_flat_project
```

查看某次 Mimic 实验目录：

```bash
tensorboard --logdir /path/to/IsaacLab/roboot16/logs/rsl_rl/roboot16_mimic_project/<run>
```

## 7. AMP 数据导出与回放

从 PKL 导出 AMP expert txt：

```bash
cd /path/to/IsaacLab/roboot16

python3 GMR/scripts/export_roboot16_amp_expert.py \
  --input_pkl data/source_pkl/run1_subject2.pkl \
  --output_txt data/amp_expert/run1_subject2.txt \
  --output_debug_pkl data/amp_expert/run1_subject2_debug.pkl \
  --robot_xml assets/Roboot1.6/xml/roboot16_1.xml
```

在 Isaac Sim 中回放 txt：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /path/to/IsaacLab/roboot16/data/amp_expert/run1_2.txt
```

同时给 `txt + pkl` 做对照：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /path/to/IsaacLab/roboot16/data/amp_expert/run1_2.txt \
  --pkl /path/to/IsaacLab/roboot16/data/segments/run1_2.pkl
```

## 8. Mimic 数据处理

CSV 转 NPZ：

```bash
./isaaclab.sh -p roboot16/scripts/mimic/csv_to_npz.py \
  --input_file roboot16/data/mimic/walk1_subject1_short.csv \
  --input_fps 30 \
  --output_file roboot16/data/mimic/walk1_subject1_short_50hz.npz \
  --output_fps 50 \
  --headless
```

只转换部分帧：

```bash
./isaaclab.sh -p roboot16/scripts/mimic/csv_to_npz.py \
  --input_file roboot16/data/mimic/walk1_subject1_test.csv \
  --input_fps 30 \
  --output_file roboot16/data/mimic/walk1_subject1_test_clip.npz \
  --output_fps 50 \
  --frame_range 1 900 \
  --headless
```

在 Isaac Sim 中回放 NPZ：

```bash
./isaaclab.sh -p roboot16/scripts/mimic/replay_npz.py \
  --motion_file roboot16/data/mimic/walk1_subject1_short_50hz.npz
```

## 9. MuJoCo sim2sim

普通策略 sim2sim：

```bash
python roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16.py roboot16.yaml
```

Mimic 策略 sim2sim：

```bash
mjpython roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py \
  --onnx roboot16/deploy/pre_train/roboot16_mimic/policy5.onnx \
  --motion-npz roboot16/data/mimic/walk1_subject1_test1_30s.npz \
  --model roboot16/assets/Roboot1.6/xml/scene_1.xml \
  --simulation-dt 0.005 \
  --control-decimation 4 \
  --camera-follow \
  --time-step-start 0 \
  --show-reference-frames \
  --show-reference-robot \
  --reference-robot-offset-y 1.0 \
  --reference-robot-alpha 0.35
```
