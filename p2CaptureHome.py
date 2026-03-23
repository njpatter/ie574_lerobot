from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from p2Config import *

import json 


robot = SO101Follower(robot_config)

robot.connect()
input("Robot is now at the home pose. Press ENTER to save the home pose.")

obs = robot.get_observation()
home_action = {k: v for k, v in obs.items() if k.endswith(".pos")}

with open(home_config_file, "w") as f:
    json.dump(home_action, f, indent=2)

print(f"Saved {home_config_file} with the current robot pose as home.")
robot.disconnect()