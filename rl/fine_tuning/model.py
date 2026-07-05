"""Value model used by TIC-VLA RL fine-tuning."""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Critic head and action-std head for TIC-VLA PPO updates."""

    def __init__(
        self,
        robot_state_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 512,
        vision_hidden_dim: int = 896,
        visual_pool_output: int = 4,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.robot_state_dim = robot_state_dim

        conv_out_channels = 256
        self.spatial_compressor = nn.Sequential(
            nn.Conv1d(
                in_channels=vision_hidden_dim,
                out_channels=conv_out_channels,
                kernel_size=3,
                stride=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                stride=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                stride=3,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size=visual_pool_output),
            nn.Flatten(start_dim=1),
        ).to(self.device)

        visual_out_dim = conv_out_channels * visual_pool_output
        self.visual_out_dim = visual_out_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        ).to(self.device)

        self.fuse_trunk = nn.Sequential(
            nn.Linear(visual_out_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        ).to(self.device)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        ).to(self.device)
        self.critic = nn.Linear(hidden_dim // 2, 1).to(self.device)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim, device=self.device) * -1.5)

    def _prepare_robot_state(self, batch: dict) -> torch.Tensor:
        robot_state = batch.get("robot_state")
        if not torch.is_tensor(robot_state):
            robot_state = torch.as_tensor(robot_state)
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)
        return robot_state.to(self.device, dtype=torch.float32)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        visual_feat = batch.get("image_embeds")
        if visual_feat is None:
            raise ValueError("ValueNetwork requires 'image_embeds' in the input batch.")

        visual_feat = visual_feat.transpose(1, 2).to(torch.float32)
        visual_feat = self.spatial_compressor(visual_feat)
        state_feat = self.state_encoder(self._prepare_robot_state(batch))
        fused = torch.cat([visual_feat, state_feat], dim=-1)
        shared = self.fuse_trunk(fused)

        critic_features = self.critic_head(shared)
        action_log_std = torch.clamp(self.actor_logstd, min=-20.0, max=2.0)
        action_std = torch.exp(action_log_std)
        value = self.critic(critic_features).squeeze(-1)

        return value, action_std
