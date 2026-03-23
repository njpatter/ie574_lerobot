from datetime import datetime
import os
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig  

# Data collection parameters
NUM_EPISODES = 50  
metadata_buffer_size = min(10, NUM_EPISODES - 1) 
FPS = 30
EPISODE_TIME_SEC = 12  
RESET_TIME_SEC = 8
TASK_DESCRIPTION = "LCtest"

# Add date and time (to minute) for unique dataset directory 
datasets_dir = os.path.join(os.getcwd(), "datasets")
# If datasets_dir doesn't exist, create it
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)
HF_DATASET_ID = os.path.join(datasets_dir, datetime.now().strftime("%Y-%m-%d_%H-%M"))


# Home configuration file
home_config_file = "single_arm_home.json"
# If home_config does't exist, create an empty one
if not os.path.exists(home_config_file):
    with open(home_config_file, "w") as f:
        f.write("{}")

# Training epoch count and checkpoint path
TRAINING_COUNT=20000 
LOCAL_CKPT_PATH = os.path.join(os.getcwd(), "outputs/checkpoints/060000/pretrained_model")  

# Camera config used in all of the other files
camera_config = {
    "front": OpenCVCameraConfig(
        index_or_path=0,
        width=640,
        height=480,
        fps=FPS
    )
}


teleop_config = SO101LeaderConfig(
    port="COM3", #"/dev/ttyACM1",
    id="leader_njp",
)

robot_config = SO101FollowerConfig(
    port="COM5", #"/dev/ttyACM0",
    id="follower_njp",
    cameras=camera_config,
)

