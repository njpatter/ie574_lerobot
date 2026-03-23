from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from huggingface_hub import snapshot_download

# Don't use this yet, I can't get it to work!!!! 
print.error("Don't use this yet, I can't get it to work!!!!")

path1 = "C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/2026-03-22_11-26"
path2 = "C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/2026-03-22_11-28"
path3 = "C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/2026-03-22_11-35"
path4 = "C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/2026-03-22_11-42"
path5 = "C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/2026-03-22_11-46"

# Instantiate MultiLeRobotDataset with the list of local paths
merged_dataset = MultiLeRobotDataset(
    repo_ids=[path1, path2, path3, path4, path5]
)

# Save the merged dataset to a new local directory
merged_dataset.save_local("C:/Users/patterson/Documents/GitHub/ie574_lerobot/datasets/a_test_merged")
