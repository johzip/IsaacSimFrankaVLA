# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import subprocess
import json

import torch

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Franka Cabinet Problem")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import and setup the Franka Cabinet Environment
import franka_cabinet_env as franka_cabinet_env

env_cfg = franka_cabinet_env.FrankaCabinetEnvCfg()
env = franka_cabinet_env.FrankaCabinetEnv(env_cfg)

#TODO actionAPI
#prompt = "In: What action should the robot take to open the top drawer?\nOut:"
#
#result = subprocess.run(
#    [
#        "/home/zipfelj/minicondaStorage/envs/openvla/bin/python",  # path to openVLA env's python
#        "openvla/scripts/openvlaActionController.py",
#        prompt
#    ],
#    capture_output=True,
#    text=True
#)
#simulation_app.update()-+

#import roboticstoolbox as rtb
#robot = rtb.models.Panda()
#print(robot)
from carb.input import KeyboardEventType

joint_positions = env._robot.data.joint_pos.clone()
step = 0.05

#Controller right now all joints are extended or shortend all at the same time uppon input
def on_keyboard_input(event):
    if event.type == KeyboardEventType.KEY_PRESS or event.type == KeyboardEventType.KEY_REPEAT:
        if event.input == carb.input.KeyboardInput.UP:
            # Move end-effector up by adjusting shoulder and elbow joints
            joint_positions[:, 1] += step  # shoulder lift
            joint_positions[:, 3] -= step  # elbow
            joint_positions[:, 5] += step  # wrist
        elif event.input == carb.input.KeyboardInput.DOWN:
            # Move end-effector down
            joint_positions[:, 1] -= step  # shoulder lift
            joint_positions[:, 3] += step  # elbow
            joint_positions[:, 5] -= step  # wrist
        elif event.input == carb.input.KeyboardInput.LEFT:
            # Rotate base joint to move left
            joint_positions[:, 0] -= step  # base rotation
        elif event.input == carb.input.KeyboardInput.RIGHT:
            # Rotate base joint to move right
            joint_positions[:, 0] += step  # base rotation
        elif event.input == carb.input.KeyboardInput.N:
            # Close gripper
            joint_positions[:, 7] -= step
            joint_positions[:, 8] -= step
        elif event.input == carb.input.KeyboardInput.M:
            # Open gripper
            joint_positions[:, 7] += step
            joint_positions[:, 8] += step

import omni.appwindow
import carb.input

app_window = omni.appwindow.get_default_app_window()
keyboard = app_window.get_keyboard()
input = carb.input.acquire_input_interface()
keyboard_sub_id = input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)



print("############")
print(joint_positions)
print("############")

# Execute
obs = env.reset()
while simulation_app.is_running():
    # Clamp joint positions to robot limits if needed
    joint_positions = torch.clamp(
        joint_positions,
        env.robot_dof_lower_limits,
        env.robot_dof_upper_limits
    )
    
    # Send joint positions as action (assuming position control)
    actions = joint_positions

    #actions = 2 * torch.rand((env.num_envs, env_cfg.action_space), device=env.device) - 1
    result = env.step(actions)
    #print("querry result:")
    #print(result)
    #result = env.step(result)
    simulation_app.update()