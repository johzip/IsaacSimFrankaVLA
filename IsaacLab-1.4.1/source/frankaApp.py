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

import roboticstoolbox as rtb
robot = rtb.models.Panda()
print(robot)

# Execute
obs = env.reset()
while simulation_app.is_running():
    actions = 2 * torch.rand((env.num_envs, env_cfg.action_space), device=env.device) - 1
    result = env.step(actions)
    #print("querry result:")
    #print(result)
    #result = env.step(result)
    simulation_app.update()