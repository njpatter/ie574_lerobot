from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from p2Config import robot_config, teleop_config, camera_config

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)