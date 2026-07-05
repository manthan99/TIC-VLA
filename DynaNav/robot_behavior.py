# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import omni
from omni.anim.people.settings import PeopleSettings, AgentEvent
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from omni.kit.scripting import BehaviorScript
from isaacsim.replicator.agent.core.robot.commands.base_command import Command
from isaacsim.replicator.agent.core.robot.robot_navigation_manager import RobotNavigationManager


class RobotBehavior(BehaviorScript):
    def on_init(self):
        self._robot_path = str(self.prim_path)
        self._robot_name = self._robot_path.split("/")[-1]
        self.commands = []

        # legacy settings from omni.anim.people
        self.setting = carb.settings.get_settings()
        self.command_path = self.setting.get(PeopleSettings.ROBOT_COMMAND_FILE_PATH)
        self._navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self._avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)
        self._navigation_manager: RobotNavigationManager = None

        self.current_command: Command = None

        self._wheeled_robot: WheeledRobot = None
        self._controller: WheelBasePoseController = None
        self._initialized: bool = False

        # variable related to stage event
        self._bus = None
        self._agent_registered_event = None

    def set_up_robot(self, wheel_joints, wheel_radius, wheel_base):
        """
        Create the wheeled robot and its controller
            wheel_joints, wheel_radius, and wheel_base differ for different robot types
        """
        self._wheeled_robot = WheeledRobot(
            prim_path=self._robot_path, name=self._robot_path, wheel_dof_names=wheel_joints, create_robot=False
        )
        self._controller = WheelBasePoseController(
            name=self._robot_name + "_wheeled_base_pose_controller",
            open_loop_wheel_controller=DifferentialController(
                name=self._robot_name + "_differential_controller", wheel_radius=wheel_radius, wheel_base=wheel_base
            ),
            is_holonomic=False,
        )

    # Clean the navigation manager and navigation manager when removing the robot from the scene
    def on_destroy(self):
        if self._navigation_manager is not None:
            self._navigation_manager.destroy()
            self._navigation_manager = None

        # reset variable related to stage event
        self._bus = None
        self._agent_registered_event = None

    def on_update(self, current_time: float, delta_time: float):
        if not self._initialized:
            self._wheeled_robot.initialize()
            self._initialized = True

        if self._navigation_manager is None:
            err_msg = self.name + "has no navigation manager. No action will be executed."
            carb.log_error(err_msg)
            return

        if self.commands:
            self.execute_command(self.commands, delta_time)

        # Publish the position of the robot after it's been updated
        if self._avoidanceOn:
            delta_time = 1 / 30
            radius = 1.0
            self._navigation_manager.publish_robot_position(delta_time, radius)

    # Must clean the robot physics callback and current command when stop playing
    def on_stop(self):
        if self._navigation_manager is not None:
            self._navigation_manager = None

        self.current_command = None
        self._initialized = False

        # variable related to stage event
        self._bus = None
        self._agent_registered_event = None

    def on_play(self):
        self.setup_for_play()
        # inform the manager to add register agent
        self.register_to_agent_manager()
        return

    def register_to_agent_manager(self):
        """
        Passing the agent data to AgentManager so it can be registered
        """
        agent_name = self.get_agent_name()
        command_info = {"agent_name": str(agent_name), "prim_path": str(self.prim_path)}
        self._bus.push(self._agent_registered_event, payload=command_info)

    def setup_for_play(self):
        # Commands and Navigation settings may be changed every time before playing, hence need to be set before play
        self.command_path = self.setting.get(PeopleSettings.ROBOT_COMMAND_FILE_PATH)
        self.commands = self.get_simulation_commands()
        self._navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self._avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)

        self._navigation_manager = RobotNavigationManager(
            self._robot_path, self._wheeled_robot, self._navmeshEnabled, self._avoidanceOn
        )

        # event for agent's register
        self._agent_registered_event = carb.events.type_from_string(AgentEvent.AgentRegistered)
        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()
        return

    def execute_command(self, commands, delta_time):
        """
        Executes commands in commands list in sequence. Removes a command once completed.

        :param list[list] commands: list of commands.
        :param float delta_time: time elapsed since last execution.
        """
        while not self.current_command:
            if not commands:
                return
            else:
                next_cmd = self.get_command(commands[0])
                if next_cmd:
                    self.current_command = self.get_command(commands[0])
                # skip invalid command
                else:
                    carb.log_error(str(commands[0], " is not a valid command"))
                    commands.pop(0)
        try:
            if self.current_command.execute(delta_time):
                commands.pop(0)
                self.current_command = None
        except:
            carb.log_error(
                "{}: invalid command. Abort this execution.".format(
                    self.get_origin_command_string(self.current_command.command)
                )
            )
            self.current_command.exit_command()
            commands.pop(0)
            self.current_command = None

    """
        The following 4 methods are related to command loading
    """

    def read_commands_from_file(self):
        """
        Reads commands from file pointed by self.command_path. Creates a Queue using queue manager if a queue is specified.
        :return: List of commands.
        :rtype: python list
        """
        result, version, context = omni.client.read_file(self.command_path)
        if result != omni.client.Result.OK:
            return []

        cmd_lines = memoryview(context).tobytes().decode("utf-8").splitlines()
        return cmd_lines

    def get_command(self, command):
        """
        Needs to be implemented in child classes
        """
        carb.log_error("get_command() not implemented for " + self._robot_path)

    def convert_str_to_command(self, cmd_line):
        if not cmd_line:
            return None
        words = cmd_line.strip().split(" ")
        if words[0] == self._robot_name:
            command = []
            command = [str(word) for word in words[1:] if word != ""]
            return command
        if words[0][0] == "#":
            return None

    """
        Following is Command injection related methods
    """

    def get_simulation_commands(self):
        cmd_lines = []
        # Get commands from cmd_file
        cmd_lines.extend(self.read_commands_from_file())
        commands = []
        for cmd_line in cmd_lines:
            command = self.convert_str_to_command(cmd_line)
            if command is not None:
                commands.append(command)
        return commands

    # inject commands to robot's command list
    def inject_command(self, command_list, executeImmediately=False):
        cmd_array = []
        for command_line in command_list:
            listed_cmd = self.convert_str_to_command(command_line)
            if listed_cmd is not None:
                cmd_array.append(listed_cmd)
        if self.commands and cmd_array:
            self.commands[1:1] = cmd_array
        else:
            self.commands[0:0] = cmd_array

    def get_agent_name(self):
        return self._robot_name

    # force the agent to end current command
    def end_current_command(self):
        if self.current_command is not None:
            self.current_command.force_quit_command()
