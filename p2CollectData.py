import os
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from p2Config import *

# If windows, import winsound
if os.name == 'nt':
    import winsound


# ===================== Init Robot & Teleop =====================
robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

# ===================== Dataset Features =====================
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}


dataset = LeRobotDataset.create(
    repo_id= HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
    metadata_buffer_size=metadata_buffer_size, # smaller than episode number
)

# ===================== UI & Processors =====================
_, events = init_keyboard_listener()
init_rerun(session_name="single_recording")

robot.connect()
teleop.connect()

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# ===================== Record Loop =====================
episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Episode {episode_idx + 1}")
    print(f"Episode {episode_idx + 1}")
    if os.name == 'nt':
        # Extra sound for start to differentiate from reset
        winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=False,
    )

    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        if os.name == 'nt':
            # Extra sound for reset to differentiate from start
            winsound.Beep(2000, 200)  
            winsound.Beep(500, 400)  
            winsound.Beep(2000, 200)   
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=False,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# ===================== Cleanup =====================
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()