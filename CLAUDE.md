# CLAUDE.md

这个文件用于告诉 Claude Code 在这个仓库里工作时应当优先了解什么。

## 仓库定位

`roboot16/` 是一个嵌套在更大 Isaac Lab 工作区里的独立 Git 仓库。

- 如果要做 `git diff`、`git restore`、`git commit` 这类操作，应当在 `roboot16/` 目录内执行。
- 大多数训练、播放、仿真相关命令，仍然是从外层 Isaac Lab 根目录启动。

## 这个仓库包含什么

这个项目围绕 Roboot16 机器人，主要包含以下几条工作流：

- AMP locomotion 训练
- 普通 RSL-RL 速度跟踪训练
- Mimic / 动作跟踪训练
- MuJoCo sim2sim 部署
- 通过 `CSVEditor/` 进行动作 CSV 可视化与编辑

## 顶层结构

```text
roboot16/
├── src/roboot16_amp_project/       # 任务注册、环境配置、资产与工具函数
├── src/amp_rsl_rl/                 # 本地 AMP 训练 runner 实现
├── scripts/reinforcement_learning/ # 训练 / 播放入口脚本
├── scripts/mimic/                  # csv<->npz 转换与回放工具
├── deploy/                         # MuJoCo 部署脚本与导出策略
├── data/                           # AMP 与 mimic 数据
├── GMR/                            # 动作重定向 / 导出辅助项目
├── CSVEditor/                      # 动作 CSV 可视化 / 编辑工具
└── logs/                           # 本地训练日志与模型输出
```

## 关键入口

### 训练 / 播放

- `scripts/reinforcement_learning/amp_rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`

### Mimic 数据处理

- `scripts/mimic/csv_to_npz.py`
- `scripts/mimic/replay_npz.py`
- `scripts/mimic/gmr_pkl_to_csv.py`

### MuJoCo 部署

- `deploy/deploy_mujoco/deploy_mujoco_roboot16.py`
- `deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py`

### 任务注册位置

任务 ID 注册都在：

```text
src/roboot16_amp_project/tasks/
```

当前主要任务分组：

- `roboot16/`：普通 flat velocity 任务
- `roboot16_amp/`：AMP flat 与 high-speed AMP
- `roboot16_mimic/`：动作跟踪 / mimic

## 当前任务 ID

- `Isaac-Roboot16-Flat-Project-v0`
- `Isaac-Roboot16-Flat-Project-Play-v0`
- `Isaac-Roboot16-AMP-Flat-Project-v0`
- `Isaac-Roboot16-AMP-Flat-Project-Play-v0`
- `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0`
- `Isaac-Roboot16-AMP-Flat-HighSpeed-Project-Play-v0`
- `Isaac-Roboot16-Mimic-Project-v0`
- `Isaac-Roboot16-Mimic-Project-Play-v0`

## 常用命令

### 训练普通 AMP

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-Project-v0 \
  --headless \
  --video \
  --video_interval 2000
```

### 训练高速 AMP

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/amp_rsl_rl/train.py \
  --task Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0 \
  --headless \
  --video \
  --video_interval 2000
```

### 训练 Mimic

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Roboot16-Mimic-Project-v0 \
  --headless \
  --video \
  --video_interval 8000
```

### 通过 play 导出 Mimic ONNX

```bash
./isaaclab.sh -p roboot16/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Roboot16-Mimic-Project-Play-v0 \
  --checkpoint /path/to/roboot16/logs/rsl_rl/roboot16_mimic_project/<run>/model_<iter>.pt
```

### 把 Mimic CSV 转成 NPZ

```bash
./isaaclab.sh -p roboot16/scripts/mimic/csv_to_npz.py \
  --input_file roboot16/data/mimic/walk1_subject1_short.csv \
  --input_fps 30 \
  --output_file roboot16/data/mimic/walk1_subject1_short_50hz.npz \
  --output_fps 50 \
  --headless
```

### 运行 MuJoCo Mimic 部署

```bash
mjpython roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py \
  --onnx roboot16/deploy/pre_train/roboot16_mimic/policy.onnx \
  --motion-npz roboot16/data/mimic/walk1_subject1_test1_30s.npz \
  --model roboot16/assets/Roboot1.6/xml/scene_1.xml
```

## 数据与产物约定

### AMP

- expert motion 文本放在 `data/amp_expert/`
- AMP 训练日志放在 `logs/amp_rsl_rl/`

### Mimic

- 输入 CSV 与导出的 NPZ 放在 `data/mimic/`
- Mimic 训练日志通常在 `logs/rsl_rl/roboot16_mimic_project/`
- `play.py` 导出的 ONNX 会放在每次 run 的 `exported/` 目录下

### Deploy

- 经典 TorchScript 策略目标位置：`deploy/pre_train/roboot16/policy.pt`
- Mimic ONNX 策略目标位置：`deploy/pre_train/roboot16_mimic/policy.onnx`

## 改代码时的注意点

- 要分清 AMP 任务、普通 velocity 任务、mimic 任务，三者入口和配置不是一套。
- 改 `tasks/` 下面的 `__init__.py` 时，要特别注意 task ID 和 config entry point 不要写错。
- 许多脚本默认假设路径是相对 Isaac Lab 根目录下的 `roboot16/...`，改路径时要保持这个习惯。
- 不要默认认为这些文件由外层 Isaac Lab 仓库跟踪；`roboot16/` 有自己独立的 `.git`。
- `CSVEditor/` 是仓库内的独立工具，依赖和主训练流程不完全一样。

## 建议优先查看的文档

- `README.md`
- `COMMON_COMMANDS.md`
- `deploy/README.md`
- `docs/README.md`
- `deploy/pre_train/roboot16/README.md`
- `deploy/pre_train/roboot16_mimic/README.md`
