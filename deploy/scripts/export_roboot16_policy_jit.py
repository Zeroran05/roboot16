from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import DEFAULT_EXPORT_POLICY, default_checkpoint


class Actor(torch.nn.Module):
    """Feed-forward actor matching the current roboot16 RSL-RL checkpoint layout."""

    def __init__(self, obs_dim: int, hidden_dims: list[int], act_dim: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.actor = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def infer_actor_layout(model_state_dict: dict[str, torch.Tensor]) -> tuple[int, list[int], int]:
    weight_shapes: list[tuple[int, int, int]] = []
    for key, value in model_state_dict.items():
        if key.startswith("actor.") and key.endswith(".weight"):
            weight_shapes.append((int(key.split(".")[1]), *value.shape))
    if not weight_shapes:
        raise RuntimeError("No actor weights found in checkpoint.")

    weight_shapes.sort(key=lambda item: item[0])
    obs_dim = weight_shapes[0][2]
    hidden_dims = [out_dim for _, out_dim, _ in weight_shapes[:-1]]
    act_dim = weight_shapes[-1][1]
    return obs_dim, hidden_dims, act_dim


def export_policy(checkpoint: Path, output: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not contain model_state_dict.")

    model_state_dict = ckpt["model_state_dict"]
    obs_dim, hidden_dims, act_dim = infer_actor_layout(model_state_dict)

    actor = Actor(obs_dim=obs_dim, hidden_dims=hidden_dims, act_dim=act_dim)
    actor.load_state_dict(
        {key: value for key, value in model_state_dict.items() if key.startswith("actor.")},
        strict=True,
    )
    actor.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(actor)
    scripted.save(str(output))

    print(f"Exported actor JIT to: {output}")
    print(f"obs_dim={obs_dim}, hidden_dims={hidden_dims}, act_dim={act_dim}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export roboot16 RSL-RL checkpoint to TorchScript actor.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Raw RSL-RL checkpoint (*.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_POLICY,
        help="Output TorchScript policy path.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint is not None else default_checkpoint()
    export_policy(checkpoint, args.output)


if __name__ == "__main__":
    main()
