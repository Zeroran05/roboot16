# Roboot16 AMP 常用指令

本文整理了 `roboot16` 当前最常用的一组命令，默认工作目录为：

```bash
cd /root/isaaclab
```

## Git 常用命令

进入仓库目录：

```bash
cd /root/isaaclab/roboot16
```

查看当前状态：

```bash
git status
```

查看当前分支和最近一次提交：

```bash
git branch --show-current
git log --oneline -1
```

查看当前 commit 的短编号：

```bash
git rev-parse --short HEAD
```

拉取远程最新代码：

```bash
git pull origin main
```

添加本次修改：

```bash
git add .
```

提交本次修改：

```bash
git commit -m "your message"
```

推送到远程仓库：

```bash
git push origin main
```

查看最近几次提交：

```bash
git log --oneline -5
```

查看某次训练对应的 Git 记录：

```bash
find logs -maxdepth 4 -type f | rg '/git/|\.diff$'
```

从服务器同步日志回本地（示例）：

```bash
rsync -avz <server>:/path/to/roboot16/logs/ ./logs_remote/
```

## 0. 常用任务一览

| 任务类型 | Task ID | 用途说明 |
| --- | --- | --- |
| 普通速度任务 | `Isaac-Roboot16-Flat-Project-v0` | 不带 AMP 的普通速度跟踪训练任务 |
| 普通速度任务 Play | `Isaac-Roboot16-Flat-Project-Play-v0` | 不带 AMP 的普通速度任务可视化/播放版本 |
| 普通 AMP 任务 | `Isaac-Roboot16-AMP-Flat-Project-v0` | 当前默认 AMP 训练任务，使用 `data/amp_expert/*.txt` |
| 普通 AMP 任务 Play | `Isaac-Roboot16-AMP-Flat-Project-Play-v0` | 当前默认 AMP 任务可视化/播放版本 |
| 高速 AMP 任务 | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0` | 在原 AMP 参数基础上额外加入高速速度课程的任务 |
| 高速 AMP 任务 Play | `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-Play-v0` | 高速 AMP 任务的可视化/播放版本 |

如果你现在想做：

- 普通 AMP 训练：用 `Isaac-Roboot16-AMP-Flat-Project-v0`
- 高速 AMP 训练：用 `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0`
- 普通非 AMP 速度训练：用 `Isaac-Roboot16-Flat-Project-v0`

## 1. 启动 AMP 训练

带录像，录像间隔为 `2000` step：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --video \
  --video_interval 2000
```

无界面训练：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --video \
  --video_interval 2000 \
  --headless
```

如果想额外控制每段录像长度，例如 `500` step：

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --video \
  --video_interval 2000 \
  --video_length 500
```

## 2. 用 TensorBoard 查看训练曲线

查看整个 AMP 训练目录：

```bash
tensorboard --logdir /root/isaaclab/roboot16/logs/amp_rsl_rl/roboot16_amp_flat_project
```

只看某一次 run：

```bash
tensorboard --logdir /root/isaaclab/roboot16/logs/amp_rsl_rl/roboot16_amp_flat_project/2026-05-04_18-28-17_run1_subject2
```

如果只想先找到最新一次训练目录：

```bash
ls -lt /root/isaaclab/roboot16/logs/amp_rsl_rl/roboot16_amp_flat_project | head
```

## 3. 导出 MuJoCo pkl 为 AMP expert txt

示例：把 `run1_subject2.pkl` 导出为当前 AMP 使用的 `txt`

```bash
cd /root/isaaclab/roboot16

python3 GMR/scripts/export_roboot16_amp_expert.py \
  --input_pkl data/source_pkl/run1_subject2.pkl \
  --output_txt data/amp_expert/run1_subject2.txt \
  --output_debug_pkl data/amp_expert/run1_subject2_debug.pkl \
  --robot_xml assets/Roboot1.6/xml/roboot16_1.xml
```

如果想边导出边可视化：

```bash
cd /root/isaaclab/roboot16

python3 GMR/scripts/export_roboot16_amp_expert.py \
  --input_pkl data/source_pkl/run1_subject2.pkl \
  --output_txt data/amp_expert/run1_subject2.txt \
  --output_debug_pkl data/amp_expert/run1_subject2_debug.pkl \
  --robot_xml assets/Roboot1.6/xml/roboot16_1.xml \
  --visualize
```

## 4. 在 Isaac Sim 中回放并校验 txt / pkl

只有 `txt`，固定 root：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /root/isaaclab/roboot16/data/amp_expert/stand/stand_0p0.txt
```

同时给 `txt + pkl`，root 按 `pkl` 轨迹运动：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /root/isaaclab/roboot16/data/amp_expert/run1_2.txt \
  --pkl /root/isaaclab/roboot16/data/segments/run1_2.pkl
```

循环播放并放慢速度：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /root/isaaclab/roboot16/data/amp_expert/run1_2.txt \
  --pkl /root/isaaclab/roboot16/data/segments/run1_2.pkl \
  --loop \
  --playback-speed 0.3
```

只做数值比对，不开回放窗口：

```bash
./isaaclab.sh -p roboot16/scripts/visualize_amp_replay_isaacsim.py \
  --txt /root/isaaclab/roboot16/data/amp_expert/run1_2.txt \
  --pkl /root/isaaclab/roboot16/data/segments/run1_2.pkl \
  --compare-only
```
