# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import numpy as np
import omni.anim.navigation.core as nav
from omni.metropolis.utils.carb_util import CarbUtil
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
from omni.anim.people.scripts.utils import Utils
from isaacsim.robot.wheeled_robots.robots import WheeledRobot


class RobotNavigationManager:
    """
    Manages navigation for a robot, such as generating static obstacle free paths and publishing current and future positions for the robot so that characters can avoid it
    Follows the same structure as found in the navigation_manager.py
    Robot will only publish its position for the characters to avoid but it will not avoid the characters
    """

    def __init__(self, robot_name, robot, navmesh_enabled, dynamic_avoidance_enabled=True):
        self._navmesh = nav.acquire_interface().get_navmesh()
        self._robot_manager = GlobalCharacterPositionManager.get_instance()
        self._robot: WheeledRobot = robot
        self._robot_name = robot_name
        self.navmesh_enabled = navmesh_enabled
        self.dynamic_avoidance_enabled = dynamic_avoidance_enabled
        self._positions_over_time = []
        self._delta_time_list = []
        self.path_points = []

    def destroy(self):
        self._navmesh = None
        self._robot_manager = None
        self._robot_name = None
        self._robot = None
        self.navmesh_enabled = None
        self._positions_over_time = None
        self._delta_time_list = None
        self.path_points = None

    def set_path_points(self, path_points):
        self.path_points = path_points

    def get_path_points(self):
        return self.path_points

    def update_current_path_point(self):
        if self.path_points:
            current_next_step = self.path_points[0]
            # Here you can set the minimum distance to decide whether the robot has reached a position
            reach_point = self.check_proximity_to_point(current_next_step, 0.04)
            if reach_point:
                self.path_points.pop(0)

    def check_proximity_to_point(self, point, proximity_dist):
        robot_pos, _ = self._robot.get_world_pose()
        # This is the way Isaac wheeledrobot computes the distance
        # It is the mean of the absolute differences along each dimension
        distance = np.mean(np.abs(robot_pos[:2] - point[:2]))
        return distance < proximity_dist

    def publish_robot_position(self, delta_time, radius):
        if delta_time == 0:
            return

        robot_pos, _ = self._robot.get_world_pose()
        robot_pos_xy = carb.Float3(robot_pos[0], robot_pos[1], 0)
        num_frames = 10

        # Store the positions and dts of the num_frames last frames
        if len(self._positions_over_time) < num_frames:
            self._positions_over_time.append(robot_pos_xy)
            self._delta_time_list.append(delta_time)
        else:
            self._positions_over_time.pop(0)
            self._positions_over_time.append(robot_pos_xy)
            self._delta_time_list.pop(0)
            self._delta_time_list.append(delta_time)

        # the estimated velocity (per sec) of the current obstacle
        if len(self._positions_over_time) > 0 and len(self._delta_time_list) > 0:
            self.velocity_vec = CarbUtil.scale3(
                CarbUtil.sub3(self._positions_over_time[-1], self._positions_over_time[0]),
                1 / sum(self._delta_time_list),
            )
        else:
            self.velocity_vec = carb.Float3(0, 0, 0)
        self._robot_manager.set_character_current_pos(self._robot_name, robot_pos_xy)
        # Here you can set the number of frames to buffer for predicting future positions
        self._robot_manager.set_character_future_pos(
            self._robot_name, CarbUtil.add3(robot_pos_xy, CarbUtil.scale3(self.velocity_vec, 10))
        )
        self._robot_manager.set_character_radius(self._robot_name, radius)

    def generate_path(self, coords):
        prev_point = coords[0]
        path = []
        if not self.navmesh_enabled:
            path.append(prev_point)
        for point in coords[1:]:
            if self.navmesh_enabled:
                generated_path = self._navmesh.query_shortest_path(prev_point, point)
                if generated_path is None:
                    carb.log_error(
                        "There is no valid path between point position : "
                        + str(prev_point)
                        + " and "
                        + "position : "
                        + str(point)
                    )
                    return
                points = generated_path.get_points()
                path.extend(points)
                prev_point = point
            else:
                path.append(point)
        self.path_points = path

    def generate_goto_path(self, coords):
        if len(coords) < 3 or len(coords) % 3 != 0:
            raise ValueError(
                "Invalid coordinate list for path generation. Coordinate list must be a sequence of x,y,z with the last cooridnate also specifying the ending rotation."
            )
        robot_pos, _ = self._robot.get_world_pose()
        start_pos = carb.Float3(robot_pos[0], robot_pos[1], 0)
        path = []

        for i in range(0, len(coords) // 3):
            curr_point = carb.Float3(float(coords[i * 3]), float(coords[i * 3 + 1]), float(coords[i * 3 + 2]))
            path.append(curr_point)

        path.insert(0, start_pos)
        self.generate_path(path)
