from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from collections.abc import Sequence


map_region = 30.0
def increase_moving_distance(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],    # pylint: disable=unused-argument
        command_name: str = 'pose_command',
        x_range_start=(-5.0, 5.0),
        x_range_end=(-20.0, 30.0),
        y_range_start=(-5.0, 5.0),   # Start with narrow left area
        y_range_end=(-20.0, 30.0),    # End with wide left area
        total_iterations=10,  # Increased from 8 to 10 for even more gradual progression
        num_steps_per_iteration=200000):  # Increased from 10000 to 15000 for more time per stage
    """Gradually increase the distance range for navigation goals."""
    cur_iteration = env.common_step_counter // num_steps_per_iteration
    # Option 1: Stay at max difficulty after reaching it (recommended)
    progress = min(cur_iteration / total_iterations, 1.0)
    
    # Option 2: Cycle through curriculum stages repeatedly
    # progress = (cur_iteration % total_iterations) / total_iterations
    
    # Interpolate X range from easy to hard
    start_x_min, start_x_max = x_range_start
    end_x_min, end_x_max = x_range_end
    
    current_x_min = start_x_min + (end_x_min - start_x_min) * progress
    current_x_max = start_x_max + (end_x_max - start_x_max) * progress
    
    # Interpolate Y range from narrow to wide
    start_y_min, start_y_max = y_range_start
    end_y_min, end_y_max = y_range_end
    
    current_y_min = start_y_min + (end_y_min - start_y_min) * progress
    current_y_max = start_y_max + (end_y_max - start_y_max) * progress
    
    # Update command ranges
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term.cfg, 'ranges'):
        command_term.cfg.ranges.pos_x = (current_x_min, current_x_max)  # type: ignore
        command_term.cfg.ranges.pos_y = (current_y_min, current_y_max)  # type: ignore


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    increased_moving_distance = CurrTerm(func=increase_moving_distance,
                                         params={})