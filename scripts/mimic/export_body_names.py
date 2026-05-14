from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


parser = argparse.ArgumentParser(description="Export articulation body names in Isaac Lab order.")
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to output body_names json file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from roboot16_amp_project.assets.robots import ROBOOT_CFG


@configclass
class BodyNamesSceneCfg(InteractiveSceneCfg):
    robot = ROBOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(BodyNamesSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(list(robot.body_names), indent=2))
    print(f"[INFO] Saved body names to: {args_cli.output}")
    print(f"[INFO] Body count: {len(robot.body_names)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
