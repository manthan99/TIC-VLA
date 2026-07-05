"""
Proximal Policy Optimization (PPO) algorithm implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 2e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        num_minibatches: int = 10,
        use_clipped_value_loss: bool = True,
        normalize_advantage: bool = True,
        max_log_ratio: float = 20.0,
        adv_clip: Optional[float] = 10.0,
        device: str = "cuda:0",
    ):
        """
        Initialize PPO algorithm.

        Args:
            actor_critic: Actor-critic network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of training epochs per update
            num_minibatches: Number of minibatches per epoch
            use_clipped_value_loss: Whether to clip value loss
            normalize_advantage: Whether to normalize advantages
            max_log_ratio: Clamp log-ratio to [-max_log_ratio, max_log_ratio] before exp for stability
            adv_clip: If set, clamp normalized advantages to [-adv_clip, adv_clip]
            device: Device to run on
        """
        self.actor_critic = actor_critic.to(device)
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage = normalize_advantage
        self.max_log_ratio = max_log_ratio
        self.adv_clip = adv_clip

        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "total_loss": [],
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards of shape (num_steps, num_envs)
            values: Value estimates of shape (num_steps, num_envs)
            next_values: Value estimates for the state after the last step, shape (num_envs,)
            dones: Done flags of shape (num_steps, num_envs)

        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        num_steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        # Compute GAE from back to front
        for step in reversed(range(num_steps)):
            # For the last step, use the provided next_values
            # For all other steps, use values[step + 1]
            if step == num_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = next_values  # next_values is 1D: (num_envs,)
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]  # values[step + 1] is 1D: (num_envs,)

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            advantages[step] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        obs: torch.Tensor | Dict[str, torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update the policy and value networks.

        Args:
            obs: Observations of shape (num_steps * num_envs, obs_dim)
            actions: Actions of shape (num_steps * num_envs, action_dim)
            old_log_probs: Old action log probabilities of shape (num_steps * num_envs, 1)
            advantages: Advantages of shape (num_steps * num_envs)
            returns: Returns of shape (num_steps * num_envs)
            old_values: Old value estimates of shape (num_steps * num_envs, 1)

        Returns:
            Dictionary of training statistics
        """
        # ----
        # Shape hygiene (critical!):
        # Different networks return scalars as (B,) or (B,1). If we mix (B,1) with (B,),
        # PyTorch can silently broadcast to (B,B), causing huge loss spikes.
        # Force all per-sample scalars to be 1D (B,) everywhere in PPO.
        # ----
        old_log_probs = old_log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        old_values = old_values.reshape(-1)

        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.adv_clip is not None:
            advantages = torch.clamp(advantages, -float(self.adv_clip), float(self.adv_clip))

        # Move once to device (non_blocking to leverage pinned memory if provided)
        if isinstance(obs, dict):
            moved_obs = {}
            first_tensor = None
            for k, v in obs.items():
                if k == "_kv_cache":
                    moved_obs[k] = v  # keep list on CPU; handled in minibatch loop
                else:
                    moved = v.to(self.device, non_blocking=True)
                    moved_obs[k] = moved
                    if first_tensor is None:
                        first_tensor = moved
            obs = moved_obs
            if first_tensor is None:
                raise ValueError("No tensor observations found for PPO update.")
            batch_size = first_tensor.shape[0]
        else:
            obs = obs.to(self.device, non_blocking=True)
            batch_size = obs.shape[0]

        actions = actions.to(self.device, non_blocking=True)
        old_log_probs = old_log_probs.to(self.device, non_blocking=True).reshape(-1)
        advantages = advantages.to(self.device, non_blocking=True).reshape(-1)
        returns = returns.to(self.device, non_blocking=True).reshape(-1)
        old_values = old_values.to(self.device, non_blocking=True).reshape(-1)

        # Compute minibatch size (ceil division to avoid zero)
        minibatch_size = max(1, (batch_size + self.num_minibatches - 1) // self.num_minibatches)

        # Training statistics
        policy_loss_epoch = []
        value_loss_epoch = []
        entropy_loss_epoch = []
        approx_kl_epoch = []
        clip_fraction_epoch = []
        total_loss_epoch = []

        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Shuffle indices on device to avoid host roundtrips
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                if isinstance(obs, dict):
                    batch_obs = {}
                    for k, v in obs.items():
                        if k == '_kv_cache':
                            # Handle KV cache list specially (it's a list, not a tensor)
                            if isinstance(v, list):
                                batch_obs[k] = [v[i] for i in mb_idx.cpu().tolist()]
                            else:
                                batch_obs[k] = v
                        else:
                            batch_obs[k] = v[mb_idx]
                else:
                    batch_obs = obs[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log_probs = old_log_probs[mb_idx].reshape(-1)
                batch_advantages = advantages[mb_idx].reshape(-1)
                batch_returns = returns[mb_idx].reshape(-1)
                batch_old_values = old_values[mb_idx].reshape(-1)

                # Evaluate actions
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(  # type: ignore
                    batch_obs, batch_actions
                )
                new_log_probs = new_log_probs.reshape(-1)
                new_values = new_values.reshape(-1)
                entropy = entropy.reshape(-1)

                # Compute ratio
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -float(self.max_log_ratio), float(self.max_log_ratio))
                ratio = torch.exp(log_ratio)

                # Policy loss (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.minimum(surr1, surr2).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_clipped = batch_old_values + torch.clamp(
                        new_values - batch_old_values, -self.clip_param, self.clip_param
                    )
                    value_loss1 = (new_values - batch_returns).pow(2)
                    value_loss2 = (value_clipped - batch_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Compute statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    
                    policy_loss_epoch.append(policy_loss.item())
                    value_loss_epoch.append(value_loss.item())
                    total_loss_epoch.append(loss.item())
                    entropy_loss_epoch.append(entropy_loss.item())
                    approx_kl_epoch.append(approx_kl.item())
                    clip_fraction_epoch.append(clip_fraction.item())

        # Update training statistics
        self.training_stats["policy_loss"].append(np.mean(policy_loss_epoch))
        self.training_stats["value_loss"].append(np.mean(value_loss_epoch))
        self.training_stats["entropy_loss"].append(np.mean(entropy_loss_epoch))
        self.training_stats["approx_kl"].append(np.mean(approx_kl_epoch))
        self.training_stats["clip_fraction"].append(np.mean(clip_fraction_epoch))
        self.training_stats["total_loss"].append(np.mean(total_loss_epoch))

        return {
            "policy_loss": float(np.mean(policy_loss_epoch)),
            "value_loss": float(np.mean(value_loss_epoch)),
            "entropy_loss": float(np.mean(entropy_loss_epoch)),
            "approx_kl": float(np.mean(approx_kl_epoch)),
            "clip_fraction": float(np.mean(clip_fraction_epoch)),
            "total_loss": float(np.mean(total_loss_epoch)),
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[key] = np.mean(values[-100:])  # Average over last 100 updates
        return stats

    def save(self, filepath: str):
        """Save the model."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_stats": self.training_stats,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "training_stats" in checkpoint:
            self.training_stats = checkpoint["training_stats"]

