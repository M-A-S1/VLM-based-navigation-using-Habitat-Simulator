#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the articulated agent to interact with the environment with optional LLM-driven navigation. Run as
```
python examples/robot_control_script.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

Controls:
- For velocity control
    - 1-7 to increase the motor target for the articulated agent arm joints
    - Q-U to decrease the motor target for the articulated agent arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the articulated agent base around
- PERIOD to print the current world coordinates of the articulated agent base
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- C to toggle between the default third-person view (third_rgb) and the robot's head view (head_rgb)
- X to change the articulated agent that is being controlled (if there are multiple articulated agents)
- O: Grasp an object
- P: Release a grasped object
- M: End episode
- Q: Reset to a fixed starting position
- ESC: Exit
"""

import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import magnum as mn
import numpy as np
import cv2
import pygame

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
    HeadRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import overlay_frame
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.nav import NavMeshSettings

# Constants
IMAGE_DIR = "images"
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "habitat-lab/habitat/config/benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
FIXED_START_POSITION = np.array([0.0, 0.0, 0.0])  # Fixed starting position after reset

def initialize_topdown_map(sim, map_resolution=0.02, draw_border=True):
    """Generate a top-down occupancy map from the navmesh and include initial agent position."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.include_static_objects = True
    navmesh_settings.agent_height = 1.5
    navmesh_settings.agent_radius = 0.3
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    bounds = sim.pathfinder.get_bounds()
    min_bound = np.array([bounds[0][0], bounds[0][2]])
    max_bound = np.array([bounds[1][0], bounds[1][2]])
    map_size = (max_bound - min_bound) / map_resolution
    map_size = np.ceil(map_size).astype(int)

    topdown_map = np.zeros((map_size[1], map_size[0]), dtype=np.uint8)
    navigable_count = 0
    non_navigable_count = 0
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            world_x = min_bound[0] + x * map_resolution
            world_z = min_bound[1] + y * map_resolution
            if sim.pathfinder.is_navigable([world_x, bounds[0][1], world_z]):
                topdown_map[y, x] = 255
                navigable_count += 1
            else:
                non_navigable_count += 1

    if draw_border:
        topdown_map[0, :] = 128
        topdown_map[-1, :] = 128
        topdown_map[:, 0] = 128
        topdown_map[:, -1] = 128

    logger.info(f"Navigable points: {navigable_count}, Non-navigable points: {non_navigable_count}")
    logger.info(f"Map bounds: min={min_bound}, max={max_bound}")

    # Draw initial agent and goal positions
    agent_pos = sim.articulated_agent.sim_obj.translation
    goal_pos = np.array([3.0, 0.0, 4.0])  # Updated goal position
    if not sim.pathfinder.is_navigable([goal_pos[0], goal_pos[1], goal_pos[2]]):
        logger.warning(f"Goal position {goal_pos} is not navigable, adjusting to nearest valid point")
        goal_pos[2] = 3.5  # Adjust z to be within navigable area
    topdown_map_with_positions = draw_positions(
        topdown_map, agent_pos, goal_pos, min_bound, map_resolution, [], None, sim
    )
    return topdown_map, min_bound, map_resolution

def world_to_map(pos, min_bound, map_resolution):
    """Convert world [x, z] to map [row, col]."""
    map_pos = (pos - min_bound) / map_resolution
    return np.array([map_pos[1], map_pos[0]], dtype=int)

def draw_positions(map_img, agent_pos, goal_pos, min_bound, map_resolution, robot_path, start_position, sim):
    """Draw robot, goal, start, path, and an arrow indicating robot's facing direction on the map with labels."""
    map_img_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

    # Extract x and z from magnum.Vector3 for 2D mapping
    agent_pos_2d = np.array([agent_pos[0], agent_pos[2]])
    goal_pos_2d = np.array([goal_pos[0], goal_pos[2]])
    agent_map_pos = world_to_map(agent_pos_2d, min_bound, map_resolution)
    goal_map_pos = world_to_map(goal_pos_2d, min_bound, map_resolution)

    # Draw start position if in LLM mode
    if start_position is not None:
        start_pos_2d = np.array([start_position[0], start_position[2]])
        start_map_pos = world_to_map(start_pos_2d, min_bound, map_resolution)
        if 0 <= start_map_pos[0] < map_img_color.shape[0] and 0 <= start_map_pos[1] < map_img_color.shape[1]:
            cv2.circle(map_img_color, tuple(start_map_pos[::-1]), 6, (0, 255, 0), -1)
            cv2.putText(map_img_color, "S", tuple(start_map_pos[::-1] + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw agent position
    if 0 <= agent_map_pos[0] < map_img_color.shape[0] and 0 <= agent_map_pos[1] < map_img_color.shape[1]:
        cv2.circle(map_img_color, tuple(agent_map_pos[::-1]), 8, (255, 0, 0), -1)
        cv2.putText(map_img_color, "Robot", tuple(agent_map_pos[::-1] + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw arrow to indicate robot's facing direction
        # Get the robot's rotation quaternion
        rotation = sim.articulated_agent.sim_obj.rotation
        # Convert quaternion to yaw angle (rotation around y-axis)
        quat = mn.Quaternion(rotation.vector, rotation.scalar)
        # Compute yaw (in radians) from quaternion
        yaw = np.arctan2(2.0 * (quat.vector.x * quat.vector.z + quat.scalar * quat.vector.y),
                         1.0 - 2.0 * (quat.vector.y**2 + quat.vector.z**2))
        # Invert yaw to match clockwise rotation
        yaw = -yaw
        # Adjust yaw by -90 degrees (π/2 radians) to align forward (z-axis) with 12 o'clock
        yaw -= np.pi / 2
        
        # Define arrow length (in map pixels)
        arrow_length = 20
        # Compute the endpoint of the arrow based on yaw
        arrow_end = agent_map_pos + np.array([np.cos(yaw) * arrow_length, -np.sin(yaw) * arrow_length])
        arrow_end = arrow_end.astype(int)[::-1]  # Reverse for OpenCV (x, y)

        # Ensure arrow endpoint is within bounds
        if (0 <= arrow_end[0] < map_img_color.shape[1] and 0 <= arrow_end[1] < map_img_color.shape[0]):
            cv2.arrowedLine(map_img_color, tuple(agent_map_pos[::-1]), tuple(arrow_end),
                            (255, 0, 0), 2, tipLength=0.3)

    # Draw goal position
    if 0 <= goal_map_pos[0] < map_img_color.shape[0] and 0 <= goal_map_pos[1] < map_img_color.shape[1]:
        cv2.circle(map_img_color, tuple(goal_map_pos[::-1]), 8, (0, 0, 255), -1)
        cv2.putText(map_img_color, "G", tuple(goal_map_pos[::-1] + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        logger.warning(f"Goal position {goal_pos_2d} out of map bounds, adjusting to nearest valid position")
        goal_map_pos = np.clip(goal_map_pos, [0, 0], [map_img_color.shape[0] - 1, map_img_color.shape[1] - 1])
        cv2.circle(map_img_color, tuple(goal_map_pos[::-1]), 8, (0, 0, 255), -1)
        cv2.putText(map_img_color, "G", tuple(goal_map_pos[::-1] + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return map_img_color

def read_llm_command(command_file):
    """Read LLM command from file."""
    if not os.path.exists(command_file):
        return None
    try:
        with open(command_file, "r") as f:
            command = f.read().strip().lower()
        return command
    except Exception as e:
        logger.info(f"Error reading command file: {e}")
        return None

def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})

def get_input_vel_ctlr(
    skip_pygame,
    cfg,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    control_humanoid,
    humanoid_controller,
):
    if skip_pygame:
        pygame.event.get()
        return step_env(env, "empty", {}), None, False, False
    else:
        pygame.event.pump()

    multi_agent = len(env._sim.agents_mgr) > 1
    if multi_agent:
        agent_k = f"agent_{agent_to_control}_"
    else:
        agent_k = ""
    arm_action_name = f"{agent_k}arm_action"

    if control_humanoid:
        base_action_name = f"{agent_k}humanoidjoint_action"
        base_key = "human_joints_trans"
    else:
        if "spot" in cfg:
            base_action_name = f"{agent_k}base_velocity_non_cylinder"
        else:
            base_action_name = f"{agent_k}base_velocity"
        arm_key = "arm_action"
        grip_key = "grip_action"
        base_key = "base_vel"

    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[arm_key]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
        base_action = None
    elif "stretch" in cfg:
        arm_action_space = np.zeros(10)
        arm_ctrlr = None
        base_action = [0, 0]
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None
    direction = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None, False, False
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_q]:
        end_ep = False  # Do not end episode, just reset position

    if not_block_input:
        # Base control
        if keys[pygame.K_j]:
            base_action = [0, 1]  # Left
            direction = "Left"
            logger.info(f"Key pressed: J, Direction: {direction}")
        elif keys[pygame.K_l]:
            base_action = [0, -1]  # Right
            direction = "Right"
            logger.info(f"Key pressed: L, Direction: {direction}")
        elif keys[pygame.K_k]:
            base_action = [-1, 0]  # Back
            direction = "Backward"
            logger.info(f"Key pressed: K, Direction: {direction}")
        elif keys[pygame.K_i]:
            base_action = [1, 0]  # Forward
            direction = "Forward"
            logger.info(f"Key pressed: I, Direction: {direction}")

        if arm_action_space.shape[0] == 7:
            # Velocity control
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
                direction = "Arm Joint 0 Increase"
                logger.info(f"Key pressed: Q, Direction: {direction}")
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0
                direction = "Arm Joint 0 Decrease"
                logger.info(f"Key pressed: 1, Direction: {direction}")
            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
                direction = "Arm Joint 1 Increase"
                logger.info(f"Key pressed: W, Direction: {direction}")
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0
                direction = "Arm Joint 1 Decrease"
                logger.info(f"Key pressed: 2, Direction: {direction}")
            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
                direction = "Arm Joint 2 Increase"
                logger.info(f"Key pressed: E, Direction: {direction}")
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0
                direction = "Arm Joint 2 Decrease"
                logger.info(f"Key pressed: 3, Direction: {direction}")
            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
                direction = "Arm Joint 3 Increase"
                logger.info(f"Key pressed: R, Direction: {direction}")
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0
                direction = "Arm Joint 3 Decrease"
                logger.info(f"Key pressed: 4, Direction: {direction}")
            elif keys[pygame.K_t]:
                arm_action[4] = 1.0
                direction = "Arm Joint 4 Increase"
                logger.info(f"Key pressed: T, Direction: {direction}")
            elif keys[pygame.K_5]:
                arm_action[4] = -1.0
                direction = "Arm Joint 4 Decrease"
                logger.info(f"Key pressed: 5, Direction: {direction}")
            elif keys[pygame.K_y]:
                arm_action[5] = 1.0
                direction = "Arm Joint 5 Increase"
                logger.info(f"Key pressed: Y, Direction: {direction}")
            elif keys[pygame.K_6]:
                arm_action[5] = -1.0
                direction = "Arm Joint 5 Decrease"
                logger.info(f"Key pressed: 6, Direction: {direction}")
            elif keys[pygame.K_u]:
                arm_action[6] = 1.0
                direction = "Arm Joint 6 Increase"
                logger.info(f"Key pressed: U, Direction: {direction}")
            elif keys[pygame.K_7]:
                arm_action[6] = -1.0
                direction = "Arm Joint 6 Decrease"
                logger.info(f"Key pressed: 7, Direction: {direction}")
        elif arm_action_space.shape[0] == 4:
            # Velocity control for Spot robot
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
                direction = "Arm Joint 0 Increase"
                logger.info(f"Key pressed: Q, Direction: {direction}")
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0
                direction = "Arm Joint 0 Decrease"
                logger.info(f"Key pressed: 1, Direction: {direction}")
            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
                direction = "Arm Joint 1 Increase"
                logger.info(f"Key pressed: W, Direction: {direction}")
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0
                direction = "Arm Joint 1 Decrease"
                logger.info(f"Key pressed: 2, Direction: {direction}")
            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
                direction = "Arm Joint 2 Increase"
                logger.info(f"Key pressed: E, Direction: {direction}")
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0
                direction = "Arm Joint 2 Decrease"
                logger.info(f"Key pressed: 3, Direction: {direction}")
            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
                direction = "Arm Joint 3 Increase"
                logger.info(f"Key pressed: R, Direction: {direction}")
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0
                direction = "Arm Joint 3 Decrease"
                logger.info(f"Key pressed: 4, Direction: {direction}")
        elif arm_action_space.shape[0] == 10:
            # Velocity control for Stretch robot
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
                direction = "Arm Joint 0 Increase"
                logger.info(f"Key pressed: Q, Direction: {direction}")
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0
                direction = "Arm Joint 0 Decrease"
                logger.info(f"Key pressed: 1, Direction: {direction}")
            elif keys[pygame.K_w]:
                arm_action[4] = 1.0
                direction = "Arm Joint 4 Increase"
                logger.info(f"Key pressed: W, Direction: {direction}")
            elif keys[pygame.K_2]:
                arm_action[4] = -1.0
                direction = "Arm Joint 4 Decrease"
                logger.info(f"Key pressed: 2, Direction: {direction}")
            elif keys[pygame.K_e]:
                arm_action[5] = 1.0
                direction = "Arm Joint 5 Increase"
                logger.info(f"Key pressed: E, Direction: {direction}")
            elif keys[pygame.K_3]:
                arm_action[5] = -1.0
                direction = "Arm Joint 5 Decrease"
                logger.info(f"Key pressed: 3, Direction: {direction}")
            elif keys[pygame.K_r]:
                arm_action[6] = 1.0
                direction = "Arm Joint 6 Increase"
                logger.info(f"Key pressed: R, Direction: {direction}")
            elif keys[pygame.K_4]:
                arm_action[6] = -1.0
                direction = "Arm Joint 6 Decrease"
                logger.info(f"Key pressed: 4, Direction: {direction}")
            elif keys[pygame.K_t]:
                arm_action[7] = 1.0
                direction = "Arm Joint 7 Increase"
                logger.info(f"Key pressed: T, Direction: {direction}")
            elif keys[pygame.K_5]:
                arm_action[7] = -1.0
                direction = "Arm Joint 7 Decrease"
                logger.info(f"Key pressed: 5, Direction: {direction}")
            elif keys[pygame.K_y]:
                arm_action[8] = 1.0
                direction = "Arm Joint 8 Increase"
                logger.info(f"Key pressed: Y, Direction: {direction}")
            elif keys[pygame.K_6]:
                arm_action[8] = -1.0
                direction = "Arm Joint 8 Decrease"
                logger.info(f"Key pressed: 6, Direction: {direction}")
            elif keys[pygame.K_u]:
                arm_action[9] = 1.0
                direction = "Arm Joint 9 Increase"
                logger.info(f"Key pressed: U, Direction: {direction}")
            elif keys[pygame.K_7]:
                arm_action[9] = -1.0
                direction = "Arm Joint 9 Decrease"
                logger.info(f"Key pressed: 7, Direction: {direction}")
        elif isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if keys[pygame.K_d]:
                arm_action[1] -= EE_FACTOR
                direction = "End Effector Right"
                logger.info(f"Key pressed: D, Direction: {direction}")
            elif keys[pygame.K_a]:
                arm_action[1] += EE_FACTOR
                direction = "End Effector Left"
                logger.info(f"Key pressed: A, Direction: {direction}")
            elif keys[pygame.K_w]:
                arm_action[0] += EE_FACTOR
                direction = "End Effector Up"
                logger.info(f"Key pressed: W, Direction: {direction}")
            elif keys[pygame.K_s]:
                arm_action[0] -= EE_FACTOR
                direction = "End Effector Down"
                logger.info(f"Key pressed: S, Direction: {direction}")
            elif keys[pygame.K_q]:
                arm_action[2] += EE_FACTOR
                direction = "End Effector Rotate CW"
                logger.info(f"Key pressed: Q, Direction: {direction}")
            elif keys[pygame.K_e]:
                arm_action[2] -= EE_FACTOR
                direction = "End Effector Rotate CCW"
                logger.info(f"Key pressed: E, Direction: {direction}")
        else:
            raise ValueError("Unrecognized arm action space")

        if keys[pygame.K_p]:
            logger.info("[robot_control_script.py]: Unsnapping")
            magic_grasp = -1
            direction = "Release"
            logger.info(f"Key pressed: P, Direction: {direction}")
        elif keys[pygame.K_o]:
            logger.info("[robot_control_script.py]: Snapping")
            magic_grasp = 1
            direction = "Grasp"
            logger.info(f"Key pressed: O, Direction: {direction}")

    if control_humanoid:
        if humanoid_controller is None:
            (
                joint_trans,
                root_trans,
            ) = env._sim.articulated_agent.get_joint_transform()
            num_joints = len(joint_trans) // 4
            root_trans = np.array(root_trans)
            index_arms_start = 10
            joint_trans_quat = [
                mn.Quaternion(
                    mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                    joint_trans[4 * index + 3],
                )
                for index in range(num_joints)
            ]
            rotated_joints_quat = []
            for index, joint_quat in enumerate(joint_trans_quat):
                random_vec = np.random.rand(3)
                random_angle = np.random.rand() * 10
                rotation_quat = mn.Quaternion.rotation(
                    mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
                )
                if index > index_arms_start:
                    joint_quat *= rotation_quat
                rotated_joints_quat.append(joint_quat)
            joint_trans = np.concatenate(
                [
                    np.array(list(quat.vector) + [quat.scalar])
                    for quat in rotated_joints_quat
                ]
            )
            base_action = np.concatenate(
                [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
            )
        else:
            relative_pos = mn.Vector3(base_action[0], 0, base_action[1])
            humanoid_controller.calculate_walk_pose(relative_pos)
            base_action = humanoid_controller.get_pose()

    if keys[pygame.K_PERIOD]:
        pos = [
            float("%.3f" % x)
            for x in env._sim.articulated_agent.sim_obj.translation
        ]
        rot = env._sim.articulated_agent.sim_obj.rotation
        ee_pos = env._sim.articulated_agent.ee_transform().translation
        logger.info(
            f"Agent state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    elif keys[pygame.K_COMMA]:
        joint_state = [
            float("%.3f" % x) for x in env._sim.articulated_agent.arm_joint_pos
        ]
        logger.info(f"Agent arm joint state: {joint_state}")

    args: Dict[str, Any] = {}

    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}
    else:
        name = arm_action_name
        if given_arm_action:
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    # Any key press will trigger image saving
    any_key_pressed = any(keys)

    return step_env(env, name, args), arm_action, end_ep, any_key_pressed

def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)
    return None

class FreeCam:
    def __init__(self):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, update_idx):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_z] and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if keys[pygame.K_u]:
                offset_rpy[1] += 1
            elif keys[pygame.K_o]:
                offset_rpy[1] -= 1
            elif keys[pygame.K_i]:
                offset_rpy[2] += 1
            elif keys[pygame.K_k]:
                offset_rpy[2] -= 1
            elif keys[pygame.K_j]:
                offset_rpy[0] += 1
            elif keys[pygame.K_l]:
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if keys[pygame.K_q]:
                offset_xyz[1] += 1
            elif keys[pygame.K_e]:
                offset_xyz[1] -= 1
            elif keys[pygame.K_w]:
                offset_xyz[2] += 1
            elif keys[pygame.K_s]:
                offset_xyz[2] -= 1
            elif keys[pygame.K_a]:
                offset_xyz[0] += 1
            elif keys[pygame.K_d]:
                offset_xyz[0] -= 1
            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz
            if keys[pygame.K_b]:
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)

            quat = euler_to_quat(self._free_rpy)
            trans = mn.Matrix4.from_(
                quat.to_matrix(), mn.Vector3(*self._free_xyz)
            )
            env._sim._sensors[
                "third_rgb"
            ]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result
        return step_result

class CameraToggle:
    def __init__(self):
        self._use_head_view = False
        self._last_pressed = 0

    @property
    def use_head_view(self):
        return self._use_head_view

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_c] and (pygame.time.get_ticks() - self._last_pressed) > 200:
            self._use_head_view = not self._use_head_view
            logger.info(f"Switching to {'head_rgb (robot head view)' if self._use_head_view else 'third_rgb (default view)'}")
            self._last_pressed = pygame.time.get_ticks()

def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()
    agent = env._sim.get_agent(0)

    # Set and capture goal image with retry if black
    goal_pos = np.array(args.goal_pos, dtype=np.float32)
    agent_state = agent.get_state()
    agent_state.position = goal_pos
    agent.set_state(agent_state)
    goal_obs = env._sim.get_sensor_observations()
    goal_image = goal_obs.get("head_rgb")  # Use head_rgb for goal image to match VLM perspective
    if goal_image is None or np.all(goal_image == 0):
        logger.warning("Goal image (head_rgb) is black or None, attempting to reposition and retry")
        agent_state.position = np.clip(goal_pos, [-2.66, -4.76, 0], [4.6, 8.17, 5])
        agent.set_state(agent_state)
        goal_obs = env._sim.get_sensor_observations()
        goal_image = goal_obs.get("head_rgb")
        if goal_image is None or np.all(goal_image == 0):
            logger.error("Failed to capture valid goal image from head_rgb, falling back to third_rgb")
            goal_image = goal_obs.get("third_rgb", np.zeros((args.play_cam_res, args.play_cam_res, 3), dtype=np.uint8))
            goal_image = goal_image[:, :, :3]
        else:
            goal_image = goal_image[:, :, :3]
    else:
        goal_image = goal_image[:, :, :3]
    cv2.imwrite(osp.join(IMAGE_DIR, "goal_image.png"), cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR))
    logger.info(f"Goal image saved to {osp.join(IMAGE_DIR, 'goal_image.png')}")

    # Reset to initial position
    obs = env.reset()

    # Initialize top-down map
    topdown_map, min_bound, map_resolution = initialize_topdown_map(env._sim)

    if not args.no_render:
        draw_ob = observations_to_image(obs, "third_rgb")
        logger.info(f"Main view dimensions: {draw_ob.shape}")
        # Set window size based on draw_ob, fallback to default if invalid
        window_width = draw_ob.shape[1] if draw_ob.shape[1] > 0 else 640
        window_height = draw_ob.shape[0] if draw_ob.shape[0] > 0 else 512
        screen = pygame.display.set_mode([window_width, window_height])
        pygame.display.set_caption("Robot Control")

    update_idx = 0
    target_fps = 30.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions: List[float] = []
    agent_to_control = 0
    start_position = None
    robot_path = []
    is_llm_mode = False
    count_steps = 0

    free_cam = FreeCam()
    camera_toggle = CameraToggle()
    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.agents_mgr) > 1

    humanoid_controller = None
    if args.use_humanoid_controller:
        humanoid_controller = HumanoidRearrangeController(args.walk_pose_path)
        humanoid_controller.reset(env._sim.articulated_agent.base_pos)

    command_file = osp.join(IMAGE_DIR, "command.txt")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    while True:
        if (
            args.save_actions
            and len(all_arm_actions) > args.save_actions_count
        ):
            break
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if args.no_render:
            keys = defaultdict(lambda: False)
        else:
            keys = pygame.key.get_pressed()

        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.agents_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        # Check for LLM command
        command = read_llm_command(command_file)
        if command == "start_llm":
            logger.info("Received 'start_llm' command. Switching to LLM navigation mode.")
            is_llm_mode = True
            start_position = np.array(env._sim.articulated_agent.sim_obj.translation)
            robot_path = [start_position.copy()]
            obs = env._sim.get_sensor_observations()
            current_image = obs.get("head_rgb" if camera_toggle.use_head_view else "third_rgb")
            if current_image is None:
                logger.warning(f"{'head_rgb' if camera_toggle.use_head_view else 'third_rgb'} sensor not found for start image")
                current_image = np.zeros((args.play_cam_res, args.play_cam_res, 3), dtype=np.uint8)
            else:
                current_image = current_image[:, :, :3]
            try:
                cv2.imwrite(osp.join(IMAGE_DIR, "start_image.png"), cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(IMAGE_DIR, "current_image.png"), cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Start and current images saved to {IMAGE_DIR}")
            except Exception as e:
                logger.info(f"Error saving start/current images: {e}")
            if os.path.exists(command_file):
                os.remove(command_file)
            continue

        # Handle LLM mode
        any_key_pressed = False
        if is_llm_mode and command in ["forward", "left", "right", "stop"]:
            action_name = f"agent_{agent_to_control}_base_velocity" if is_multi_agent else "base_velocity"
            action_args = {}
            if command == "forward":
                action_args = {"base_vel": [1.0, 0.0]}
                logger.info(f"LLM command: {command.upper()}, Direction: Forward")
            elif command == "left":
                action_args = {"base_vel": [0.0, 1.0]}
                logger.info(f"LLM command: {command.upper()}, Direction: Left")
            elif command == "right":
                action_args = {"base_vel": [0.0, -1.0]}
                logger.info(f"LLM command: {command.upper()}, Direction: Right")
            elif command == "stop":
                action_args = {"base_vel": [0.0, 0.0]}
                logger.info(f"LLM command: {command.upper()}, Direction: Stop")
                env.episode_over = True
            if os.path.exists(command_file):
                os.remove(command_file)
            step_result = step_env(env, action_name, action_args)
            any_key_pressed = True  # Treat LLM commands as key presses
        else:
            # Manual mode
            step_result, arm_action, end_ep, any_key_pressed = get_input_vel_ctlr(
                args.no_render,
                args.cfg,
                use_arm_actions[update_idx] if use_arm_actions is not None else None,
                env,
                not free_cam.is_free_cam_mode,
                agent_to_control,
                args.control_humanoid,
                humanoid_controller=humanoid_controller,
            )
            if step_result is None:
                break
            if end_ep:
                total_reward = 0
                if gfx_measure is not None:
                    gfx_measure.get_metric(level_get=True)
                obs = env.reset()
                agent = env._sim.get_agent(0)
                agent_state = agent.get_state()
                agent_state.position = FIXED_START_POSITION
                if not env._sim.pathfinder.is_navigable(FIXED_START_POSITION):
                    logger.warning(f"Fixed start position {FIXED_START_POSITION} is not navigable, finding nearest navigable point")
                    nearest_pos = env._sim.pathfinder.get_random_navigable_point_near(
                        FIXED_START_POSITION, radius=2.0
                    )
                    agent_state.position = nearest_pos
                agent.set_state(agent_state)
                logger.info(f"Agent reset to position: {agent_state.position}")
                topdown_map, min_bound, map_resolution = initialize_topdown_map(env._sim)
                start_position = None
                robot_path = []
                is_llm_mode = False
                count_steps = 0
                continue
            elif keys[pygame.K_q]:
                # Handle Q key reset separately
                agent = env._sim.get_agent(0)
                agent_state = agent.get_state()
                agent_state.position = FIXED_START_POSITION
                if not env._sim.pathfinder.is_navigable(FIXED_START_POSITION):
                    logger.warning(f"Fixed start position {FIXED_START_POSITION} is not navigable, finding nearest navigable point")
                    nearest_pos = env._sim.pathfinder.get_random_navigable_point_near(
                        FIXED_START_POSITION, radius=2.0
                    )
                    agent_state.position = nearest_pos
                agent.set_state(agent_state)
                logger.info(f"Agent reset to position: {agent_state.position}")
                obs = env._sim.get_sensor_observations()

        if is_llm_mode and any_key_pressed:
            count_steps += 1
            current_pos = np.array(env._sim.articulated_agent.sim_obj.translation)
            robot_path.append(current_pos)
            logger.info(f"Step {count_steps}: Position: {current_pos}")

        if not args.no_render:
            free_cam.update(env, step_result, update_idx)
            camera_toggle.update()

        all_arm_actions.append(arm_action if arm_action is not None else np.zeros(arm_action_space.shape[0]))
        update_idx += 1
        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        agent_pos = env._sim.articulated_agent.sim_obj.translation

        # Check goal distance in LLM mode
        if is_llm_mode:
            distance_to_goal = np.linalg.norm(agent_pos - goal_pos)
            logger.info(f"Distance to goal: {distance_to_goal:.3f}")
            if distance_to_goal < 0.2:
                logger.info("Successfully navigated to the destination point")
                env.episode_over = True

        # Update top-down map
        topdown_map_updated = draw_positions(
            topdown_map, agent_pos, goal_pos, min_bound, map_resolution, robot_path, start_position, env._sim
        )
        cv2.imwrite(osp.join(IMAGE_DIR, "topdown_map.png"), topdown_map_updated)

        # Save image whenever any key is pressed or LLM command is executed
        if any_key_pressed:
            current_image = obs.get("head_rgb" if camera_toggle.use_head_view else "third_rgb")
            if current_image is not None:
                current_image = np.ascontiguousarray(current_image[:, :, :3], dtype=np.uint8)
                cv2.imwrite(
                    osp.join(IMAGE_DIR, "current_image.png"),
                    cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                )
                logger.info(f"Current image saved to {osp.join(IMAGE_DIR, 'current_image.png')}")
            else:
                logger.warning(f"{'head_rgb' if camera_toggle.use_head_view else 'third_rgb'} sensor not found for current image")

        if not args.no_render:
            if free_cam.is_free_cam_mode:
                cam = obs.get("third_rgb")
                if cam is None:
                    logger.warning("third_rgb observation is None in free cam mode")
                    use_ob = np.zeros((args.play_cam_res, args.play_cam_res, 3), dtype=np.uint8)
                else:
                    use_ob = np.ascontiguousarray(cam[:, :, :3], dtype=np.uint8)
            else:
                use_ob = observations_to_image(obs, "head_rgb" if camera_toggle.use_head_view else "third_rgb")
                if not args.skip_render_text and use_ob is not None:
                    # Validate use_ob before overlay_frame
                    if use_ob.shape[-1] == 3 and use_ob.dtype == np.uint8 and use_ob.flags['C_CONTIGUOUS']:
                        try:
                            use_ob = overlay_frame(use_ob, info)
                        except Exception as e:
                            logger.warning(f"Failed to overlay text on frame: {e}")
                    else:
                        logger.warning(
                            f"Skipping overlay_frame due to invalid use_ob - shape: {use_ob.shape}, "
                            f"dtype: {use_ob.dtype}, contiguous: {use_ob.flags['C_CONTIGUOUS']}"
                        )

            draw_ob = use_ob.copy() if use_ob is not None else np.zeros((args.play_cam_res, args.play_cam_res, 3), dtype=np.uint8)
            if update_idx == 0:
                logger.info(f"Normal mode: use_ob shape = {use_ob.shape if use_ob is not None else 'None'}")
                logger.info(f"draw_ob shape = {draw_ob.shape}")
                logger.info(f"topdown_map_updated shape = {topdown_map_updated.shape}")

            draw_ob_transposed = np.transpose(draw_ob, (1, 0, 2))
            draw_ob_surface = pygame.surfarray.make_surface(draw_ob_transposed)
            screen.blit(draw_ob_surface, (0, 0))
            pygame.display.flip()

            cv2.imshow("Top-Down Map", topdown_map_updated)
            cv2.waitKey(20)

        if args.save_obs:
            screen_width = draw_ob.shape[1] + topdown_map_updated.shape[1]
            screen_height = max(draw_ob.shape[0], topdown_map_updated.shape[0])
            combined_ob = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            combined_ob[:draw_ob.shape[0], :draw_ob.shape[1]] = draw_ob
            combined_ob[:topdown_map_updated.shape[0], draw_ob.shape[1]:] = topdown_map_updated
            all_obs.append(combined_ob)

        if env.episode_over:
            if is_llm_mode:
                logger.info(f"Episode finished after {count_steps} steps.")
            total_reward = 0
            obs = env.reset()
            agent = env._sim.get_agent(0)
            agent_state = agent.get_state()
            agent_state.position = FIXED_START_POSITION
            if not env._sim.pathfinder.is_navigable(FIXED_START_POSITION):
                logger.warning(f"Fixed start position {FIXED_START_POSITION} is not navigable, finding nearest navigable point")
                nearest_pos = env._sim.pathfinder.get_random_navigable_point_near(
                    FIXED_START_POSITION, radius=2.0
                )
                agent_state.position = nearest_pos
            agent.set_state(agent_state)
            logger.info(f"Agent reset to position: {agent_state.position}")
            topdown_map, min_bound, map_resolution = initialize_topdown_map(env._sim)
            start_position = None
            robot_path = []
            is_llm_mode = False
            count_steps = 0

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = all_arm_actions[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        logger.info(f"Saved actions to {save_path}")
        pygame.quit()
        cv2.destroyAllWindows()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(level_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        pygame.quit()
        cv2.destroyAllWindows()

def has_pygame():
    return pygame is not None

def observations_to_image(obs, sensor_name="third_rgb"):
    """Convert observations to image for rendering with OpenCV compatibility."""
    default_shape = (512, 640, 3)
    image = obs.get(sensor_name, np.zeros(default_shape, dtype=np.uint8))
    # Print shape info only once
    if not hasattr(observations_to_image, 'shape_printed'):
        logger.info(f"Raw {sensor_name} shape: {image.shape}, dtype: {image.dtype}, contiguous: {image.flags['C_CONTIGUOUS']}")
        observations_to_image.shape_printed = True
    
    # Ensure the image has the correct shape and type
    if image.shape[-1] == 3:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
    else:
        logger.warning(f"Unexpected {sensor_name} shape: {image.shape}. Resetting to default shape.")
        image = np.zeros(default_shape, dtype=np.uint8)
    
    if not hasattr(observations_to_image, 'shape_printed'):
        logger.info(f"Processed {sensor_name} shape: {image.shape}, dtype: {image.dtype}, contiguous: {image.flags['C_CONTIGUOUS']}")
        observations_to_image.shape_printed = True
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="The number of steps the saved action trajectory is clipped to."
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )
    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )
    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )
    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--goal-pos",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[-1.3, 0.0, 0.5],  # Updated goal position
        help="Goal position as x,y,z (e.g., --goal-pos 3.0,0.0,4.0)"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    ),
                    "head_rgb_sensor": HeadRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics."
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()

    pygame.init()
    with habitat.Env(config=config) as env:
        play_env(env, args, config)