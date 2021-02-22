import argparse
import os
import shutil
from pyquaternion import Quaternion
from tqdm import tqdm

from anno_tool_api import LyftDatasetAnno


def parse_args():
    parser = argparse.ArgumentParser("Precompute top views and object annotations.")
    parser.add_argument(
        "--lyft_root",
        type=str,
        help="Root directory for the nuscenes dataset, should contain subdirectories 'maps', 'samples', 'sweeps'.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory under which all the top down views and infos will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lyft_root = args.lyft_root
    save_dir = args.save_dir

    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    lyft = LyftDatasetAnno(
        data_path=lyft_root,
        json_path=os.path.join(lyft_root, "data/"),
        map_resolution=0.2,
        verbose=True,
    )

    scene_recs = lyft.scene
    for scene_rec in scene_recs:
        scene_token = scene_rec["token"]
        print(f"\nRenaming scene {scene_token}")
        log_token = scene_rec["log_token"]
        name = scene_rec["name"]
        nbr_samples = scene_rec["nbr_samples"]

        first_sample_token = scene_rec["first_sample_token"]
        last_sample_token = scene_rec["last_sample_token"]

        first_sample_rec = lyft.get("sample", first_sample_token)
        last_sample_rec = lyft.get("sample", last_sample_token)

        # Iterate through all samples, collect all sample tokens
        sample_rec = first_sample_rec
        sample_tokens = []
        while sample_rec["next"]:
            sample_token = sample_rec["token"]
            sample_rec = lyft.get("sample", sample_rec["next"])
            sample_tokens.append(sample_token)

        if not os.path.exists(os.path.join(save_dir, scene_token)):
            os.makedirs(os.path.join(save_dir, scene_token))

        for i, sample_token in tqdm(enumerate(sample_tokens)):
            frontal_orig_path = (
                os.path.join(save_dir, scene_token, f"{sample_token}.jpg"),
            )
            top_down_orig_path = (
                os.path.join(save_dir, scene_token, f"{sample_token}.png"),
            )
            data_orig_path = os.path.join(save_dir, scene_token, f"{sample_token}.json")

            frontal_new_path = (
                os.path.join(save_dir, scene_token, f"frontal_{i}.jpg"),
            )
            top_down_new_path = (
                os.path.join(save_dir, scene_token, f"top_down_{i}.png"),
            )
            data_new_path = os.path.join(save_dir, scene_token, f"frame_{i}_data.json")

            shutil.move(frontal_orig_path, frontal_new_path)
            shutil.move(top_down_orig_path, top_down_new_path)
            shutil.move(data_orig_path, data_new_path)