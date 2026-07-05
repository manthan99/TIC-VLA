# mix_resnet_builder.py
import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import model_builder
from torchvision.models import resnet18

class MixResnetBuilder(NetworkBuilder):
    """
    Two-branch Actor-Critic:
      - image -> ResNet18 (trainable)
      - vector -> small MLP
      - concat -> shared MLP -> policy head, value head
    Expects observation dict: {'obs': [B,D], 'images': [B,C,H,W]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Read common config fields if present (with defaults)
        net_cfg = kwargs.get('params', {}).get('network', {})
        cnn_cfg = net_cfg.get('cnn', {})
        mlp_cfg = net_cfg.get('mlp', {})

        # Optional hyperparams
        self.cnn_output_dim = int(cnn_cfg.get('out_features', 256))   # bottleneck after resnet
        self.mlp_units = list(mlp_cfg.get('units', [128, 128, 128]))
        self.activation = nn.ELU() if mlp_cfg.get('activation', 'elu') == 'elu' else nn.ReLU()

    def _make_image_encoder(self, in_channels):
        # Build ResNet18 and adapt first conv to input channels
        enc = resnet18(weights=None)          # no pretrained weights; set weights='IMAGENET1K_V1' if you want transfer
        if in_channels != 3:
            old = enc.conv1
            enc.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                  kernel_size=old.kernel_size, stride=old.stride,
                                  padding=old.padding, bias=old.bias is not None)
        # Replace final FC with identity to get a feature vector
        enc.fc = nn.Identity()

        # Wrap with a bottleneck (to control feature size)
        proj = nn.Sequential(
            enc,
            nn.Linear(512, self.cnn_output_dim),
            nn.ReLU(inplace=True),
        )
        return proj

    def _make_vector_encoder(self, in_dim):
        layers = []
        prev = in_dim
        for h in self.mlp_units:
            layers += [nn.Linear(prev, h), self.activation]
            prev = h
        return nn.Sequential(*layers), prev

    def build(self, name, **kwargs):
        action_space = kwargs['action_space']
        obs_space = kwargs['observation_space']  # Dict('obs_addons', 'obs_sensor')

        act_dim = int(action_space.shape[0])

        # Infer shapes
        vec_dim = int(obs_space['obs_addons'].shape[0])
        C, H, W = obs_space['obs_sensor'].shape

        # Encoders
        img_enc = self._make_image_encoder(C)
        vec_enc, vec_out = self._make_vector_encoder(vec_dim)

        fused_dim = self.cnn_output_dim + vec_out

        # Shared trunk
        trunk = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),       nn.ReLU(inplace=True),
        )

        # Heads
        mu = nn.Linear(256, act_dim)
        logstd = nn.Parameter(torch.zeros(act_dim))
        value = nn.Linear(256, 1)

        # Pack into a single module that rl_games can call
        class MixResnetAC(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_enc = img_enc
                self.vec_enc = vec_enc
                self.trunk = trunk
                self.mu = mu
                self.value = value
                self.logstd = logstd

            def forward(self, obs_dict):
                # obs_dict: {'obs_addons': [B,D], 'obs_sensor': [B,C,H,W]}
                v = obs_dict['obs_addons']
                x = obs_dict['obs_sensor']

                v_feat = self.vec_enc(v)
                i_feat = self.img_enc(x)

                z = torch.cat([v_feat, i_feat], dim=-1)
                z = self.trunk(z)

                mu = self.mu(z)
                std = torch.exp(self.logstd).expand_as(mu)
                val = self.value(z)
                return {'mu': mu, 'logstd': self.logstd, 'std': std, 'value': val}

        net = MixResnetAC()
        return net

# Register the builder under a name you can use in YAML
model_builder.ModelBuilder.register_builder('mix_resnet', MixResnetBuilder)
