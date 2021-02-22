import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Precompute top views and object annotations.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory under which all the top down views and infos are saved.",
    )
    parser.add_argument(
        "--save_path", type=str, help="Path under which the input list is saved."
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        help="Path on Amazon S3.",
    )
    return parser.parse_args()


# {
#       "frontal": "https://lyft-path.s3.eu-west-3.amazonaws.com/tmp/frontal_ix.jpg",
#       "top-down": "https://lyft-path.s3.eu-west-3.amazonaws.com/tmp/top_down_ix.png",
#       "frame_data_url": "https://lyft-path.s3.eu-west-3.amazonaws.com/tmp/frame_ix_data.json",
#       "video_data": { "max_frame_ind": 99 },
#     }


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir
    save_path = args.save_path
    file_prefix = args.file_prefix

    scene_dicts = []

    scenes = os.listdir(save_dir)
    for scene in scenes:
        frontals = [
            item
            for item in os.listdir(os.path.join(save_dir, scene))
            if item.endswith(".jpg")
        ]
        top_down = [
            item
            for item in os.listdir(os.path.join(save_dir, scene))
            if item.endswith(".png")
        ]
        frame_data = [
            item
            for item in os.listdir(os.path.join(save_dir, scene))
            if item.endswith(".json")
        ]
        assert len(frontals) == len(top_down) == len(frame_data), "Missing files..."

        scene_dict = {}
        scene_dict["video_data"] = {}
        scene_dict["video_data"]["max_frame_ind"] = len(frontals)

        scene_dict["frontal"] = os.path.join(file_prefix, scene, "frontal_ix.jpg")
        scene_dict["top-down"] = os.path.join(file_prefix, scene, "top_down_ix.png")
        scene_dict["frame_data_url"] = os.path.join(
            file_prefix, scene, "frame_ix_data.json"
        )

        scene_dicts.append(scene_dict)

    with open(save_path, "w") as f:
        json.dump(scene_dicts, f)
