import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def advanced_generated_commands(env: ManagerBasedRLEnv, command_name: str, max_dim: int, normalize: bool) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # Get the full command tensor
    full_command = env.command_manager.get_command(command_name)
    
    # Extract only the position command (first 2 dimensions for 2D)
    command = full_command[..., :max_dim]
    
    if not normalize:
        return command
    else:
        max_ = 5.
        dis = torch.norm(command, dim=-1, keepdim=False)
        mask = dis > max_
        if mask.sum() > 0:
            scale = max_ / dis[mask]
            command[mask] = scale.reshape(-1, 1) * command[mask]
        return command

def goal_obs(env, command_name: str) -> torch.Tensor:
    v = env.command_manager.get_command(command_name)[:, :2].clone()
    d = v.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dir2  = v / d                      # in [-1,1]
    return torch.cat([dir2, d], dim=1)  # [ux, uy, d]


def image_processed(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).to(images.dtype)
            # std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).to(images.dtype)
            images = images.permute(0, 3, 1, 2)
            # images = (images - mean) / std
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()

# # ticvla/navigation_mdp/vision_obs.py
# import torch
# import torch.nn.functional as F

# def image_processed(
#     env,
#     sensor_name: str = "camera",
#     key: str = "rgb",
#     out_h: int = 135,
#     out_w: int = 240,
#     grayscale: bool = True,
#     frame_stack: int = 4,
#     drop_alpha: bool = True,
# ):
#     """
#     Returns [E, C, H, W] float32 on the same device as the camera tensors.
#     C = frame_stack (grayscale) or 3*frame_stack (RGB).
#     """
#     cam = env.scene.sensors[sensor_name]
#     img = cam.data.output[key]           # [E,H,W,4] or [E,H,W,3], uint8
#     if drop_alpha and img.shape[-1] == 4:
#         img = img[..., :3]
#     img = img.float() / 255.0            # [0,1]
#     img = img.permute(0, 3, 1, 2)        # -> [E,3,H,W]
#     img = F.interpolate(img, size=(out_h, out_w), mode="bilinear", align_corners=False)

#     if grayscale:
#         w = torch.tensor([0.2989, 0.5870, 0.1140], device=img.device, dtype=img.dtype).view(1,3,1,1)
#         img = (img * w).sum(1, keepdim=True)  # [E,1,H,W]
#         ch_per_frame = 1
#     else:
#         ch_per_frame = 3

#     C = frame_stack * ch_per_frame
#     buf_name = "_frame_stack_buf"
#     if not hasattr(env, buf_name):
#         setattr(env, buf_name, torch.zeros((env.num_envs, C, out_h, out_w), device=img.device, dtype=img.dtype))
#     buf = getattr(env, buf_name)

#     # roll and insert latest frame
#     buf = torch.roll(buf, shifts=-ch_per_frame, dims=1)
#     buf[:, -ch_per_frame:, :, :] = img
#     setattr(env, buf_name, buf)
#     return buf  # [E, C, H, W]
