# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd

from rsl_rl.networks import EmpiricalNormalization


class Discriminator(nn.Module):
    """Discriminator implements the discriminator network for the AMP algorithm.

    This network is trained to distinguish between expert and policy-generated data.
    It also provides reward signals for the policy through adversarial learning.

    Args:
        input_dim (int): Dimension of the concatenated input state (state + next state).
        hidden_layer_sizes (list): List of hidden layer sizes.
        reward_scale (float): Scale factor for the computed reward. This matches
            TienKung-Lab's ``amp_reward_coef`` semantics.
        device (str | torch.device): Device to run the model on.
        use_minibatch_std (bool): Whether to use minibatch standard deviation in the network
        empirical_normalization (bool): Whether to normalize AMP observations empirically before scoring.
    """

    def __init__(
        self,
        input_dim: int,
        amp_obs_dim: int,
        condition_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        device: str | torch.device = "cpu",
        loss_type: str = "LSGAN",
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.amp_obs_dim = amp_obs_dim
        self.condition_dim = condition_dim
        self.reward_scale = reward_scale
        layers = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        self.linear = nn.Linear(final_in_dim, 1)

        self.empirical_normalization = empirical_normalization
        if empirical_normalization:
            self.amp_normalizer = EmpiricalNormalization(shape=[amp_obs_dim])
        else:
            self.amp_normalizer = nn.Identity()

        self.to(self.device)
        self.train()
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type if loss_type is not None else "LSGAN"
        if self.loss_type != "LSGAN":
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported type is 'LSGAN'."
            )
        self.loss_fun = torch.nn.MSELoss()

    def _build_input(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.amp_normalizer(state)
        next_state = self.amp_normalizer(next_state)
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError("Conditional discriminator requires a condition tensor.")
            condition = condition.to(state.device, dtype=state.dtype).reshape(state.shape[0], self.condition_dim)
            # Map forward-speed commands from the task's nominal [0, 4] range to [-1, 1].
            condition = (condition - 2.0) / 2.0
            return torch.cat([state, next_state, condition], dim=-1)
        return torch.cat([state, next_state], dim=-1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
        next_state: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x (Tensor): Input tensor (batch_size, input_dim).

        Returns:
            Tensor: Discriminator output logits/scores.
        """
        if x is None:
            if state is None or next_state is None:
                raise ValueError("Either `x` or both `state` and `next_state` must be provided.")
            x = self._build_input(state, next_state, condition)
        elif state is not None or next_state is not None or condition is not None:
            raise ValueError("Provide either pre-concatenated `x` or structured inputs, not both.")

        h = self.trunk(x)
        if self.use_minibatch_std:
            s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)
        return self.linear(h)

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        """Mean over feature-wise std across the batch; shape (B,1)."""
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        s = h.float().std(dim=0, unbiased=False).mean()
        return s.expand(h.shape[0], 1).to(h.dtype)

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict pure AMP/style reward using TienKung-Lab's bounded reward.

        Args:
            state (Tensor): Current state tensor.
            next_state (Tensor): Next state tensor.

        Returns:
            Tensor: Computed adversarial reward.
        """
        with torch.no_grad():
            discriminator_logit = self.forward(state=state, next_state=next_state, condition=condition)
            reward = self.reward_scale * torch.clamp(
                1 - 0.25 * torch.square(discriminator_logit - 1), min=0
            )
            return reward.squeeze()

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        task_reward_lerp: float,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mixed AMP reward using TienKung-Lab's reward formulation."""
        with torch.no_grad():
            discriminator_logit = self.forward(state=state, next_state=next_state, condition=condition)
            style_reward = self.reward_scale * torch.clamp(
                1 - 0.25 * torch.square(discriminator_logit - 1), min=0
            )
            mixed_reward = (1.0 - task_reward_lerp) * style_reward + task_reward_lerp * task_reward.unsqueeze(-1)
            return mixed_reward.squeeze(), discriminator_logit

    def policy_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """Compute LSGAN loss for policy transitions with target -1."""
        expected = -1 * torch.ones_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def expert_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """Compute LSGAN loss for expert transitions with target +1."""
        expected = torch.ones_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update empirical statistics using provided AMP batches."""
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch)

    def compute_loss(
        self,
        policy_d,
        expert_d,
        sample_amp_expert,
        sample_amp_policy,
        policy_condition: torch.Tensor | None = None,
        expert_condition: torch.Tensor | None = None,
        lambda_: float = 10,
    ):

        sample_amp_expert = tuple(self.amp_normalizer(s) for s in sample_amp_expert)
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            condition=expert_condition,
            lambda_=lambda_,
        )
        expert_loss = self.expert_loss(expert_d)
        policy_loss = self.policy_loss(policy_d)
        amp_loss = 0.5 * (expert_loss + policy_loss)
        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: tuple[torch.Tensor, torch.Tensor],
        condition: torch.Tensor | None = None,
        lambda_: float = 10,
    ) -> torch.Tensor:
        """Compute TienKung-Lab style gradient penalty on expert samples.

        Args:
            expert_states (tuple[Tensor, Tensor]): A tuple containing batches of expert states and expert next states.
            lambda_ (float): Penalty coefficient.

        Returns:
            Tensor: Gradient penalty value.
        """
        # Keep the gradient penalty focused on AMP state inputs; the speed
        # condition still influences discriminator scoring in the forward path.
        expert = torch.cat(expert_states, -1)
        data = expert.detach().requires_grad_(True)
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError("Conditional discriminator requires condition for gradient penalty.")
            condition = condition.to(expert_states[0].device, dtype=expert_states[0].dtype).reshape(
                expert_states[0].shape[0], self.condition_dim
            )
            condition = (condition - 2.0) / 2.0
            trunk_input = torch.cat([data, condition.detach()], -1)
        else:
            trunk_input = data
        h = self.trunk(trunk_input)
        if self.use_minibatch_std:
            with torch.no_grad():
                s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)
        scores = self.linear(h)
        grad = autograd.grad(
            outputs=scores,
            inputs=data,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()
