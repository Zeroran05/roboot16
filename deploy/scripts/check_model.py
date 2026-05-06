from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

from common import DEFAULT_MODEL, body_names, hinge_joint_ids, joint_names, set_default_pose


def main():
    parser = argparse.ArgumentParser(description="Load Roboot1.6 in MuJoCo and print model metadata.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to MuJoCo XML model.")
    parser.add_argument("--no-viewer", action="store_true", help="Only print metadata; do not open viewer.")
    parser.add_argument("--seconds", type=float, default=None, help="Optional viewer duration. By default, run until closed.")
    parser.add_argument("--base-height", type=float, default=None, help="Initial floating-base height. Defaults to auto-grounded placement.")
    parser.add_argument("--step", action="store_true", help="Advance physics while the viewer is open.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    set_default_pose(model, data, base_height=args.base_height)

    print(f"Loaded model: {args.model}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}, nbody={model.nbody}")
    print("\nJoints:")
    for jid, name in enumerate(joint_names(model)):
        print(
            f"  {jid:02d} {name:28s} "
            f"type={int(model.jnt_type[jid])} qposadr={model.jnt_qposadr[jid]} dofadr={model.jnt_dofadr[jid]} "
            f"range={model.jnt_range[jid].tolist()}"
        )

    print("\nHinge joints used by PD/policy:")
    for jid in hinge_joint_ids(model):
        print(f"  {jid:02d} {joint_names(model)[jid]}")

    print("\nBodies:")
    for bid, name in enumerate(body_names(model)):
        print(f"  {bid:02d} {name}")
    print(f"\nDefault reset base_z={data.qpos[2]:.4f}")

    if args.no_viewer:
        return

    from mujoco import viewer as mujoco_viewer

    print("\nOpening viewer. On macOS, run this script with mjpython if the viewer fails.")
    with mujoco_viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and (args.seconds is None or time.time() - t0 < args.seconds):
            if args.step:
                mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
