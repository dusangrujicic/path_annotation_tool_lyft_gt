import argparse
import json
import os
import sys
import shutil

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


def main():
    args = parse_args()

    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))

    lyft = LyftDatasetAnno(
        data_path=args.lyft_root,
        json_path=os.path.join(args.lyft_root, "data/"),
        map_resolution=0.2,
        verbose=True,
    )

    sample_tokens = [
        "24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8",
        "2346756c83f6ae8c4d1adec62b4d0d31b62116d2e1819e96e9512667d15e7cec",
    ]

    for i, sample_token in tqdm(enumerate(sample_tokens)):
        sample = lyft.get("sample", sample_token)
        sample_data_camera = lyft.get("sample_data", sample["data"]["CAM_FRONT"])
        sample_data_lidar = lyft.get("sample_data", sample["data"]["LIDAR_TOP"])
        sample_data_token_camera = sample_data_camera["token"]
        sample_data_token_lidar = sample_data_lidar["token"]

        (
            image_objects_bbox,
            camera_intrinsic,
            camera_translation,
            camera_rotation,
        ) = lyft.explorer.render_sample_camera_data_custom(
            sample_data_token=sample_data_token_camera,
            out_path=os.path.join(args.save_dir, f"frontal_{i}.jpg"),
            with_annotations=False,
        )

        (
            ego_bbox,
            ego_translation,
            ego_rotation,
            map_objects_center,
            map_objects_elevation,
            map_objects_bbox,
            objects_type,
            objects_token,
            map_patch,
        ) = lyft.explorer.render_sample_lidar_data_custom(
            sample_data_token=sample_data_token_lidar,
            out_path=os.path.join(args.save_dir, f"top_down_{i}.png"),
            with_annotations=False,
        )

        # Remove objects not visible in camera view
        map_objects_center = [
            item
            for i, item in enumerate(map_objects_center)
            if image_objects_bbox[i] != None
        ]
        map_objects_elevation = [
            item
            for i, item in enumerate(map_objects_elevation)
            if image_objects_bbox[i] != None
        ]
        map_objects_bbox = [
            item
            for i, item in enumerate(map_objects_bbox)
            if image_objects_bbox[i] != None
        ]
        objects_type = [
            item for i, item in enumerate(objects_type) if image_objects_bbox[i] != None
        ]
        objects_token = [
            item
            for i, item in enumerate(objects_token)
            if image_objects_bbox[i] != None
        ]
        image_objects_bbox = [item for item in image_objects_bbox if item != None]

        out_dict = {
            "sample_token": sample_token,
            "map_patch": map_patch,
            "egobbox": ego_bbox,
            "map_objects_center": map_objects_center,
            "map_objects_elevation": map_objects_elevation,
            "map_objects_bbox": map_objects_bbox,
            "image_objects_bbox": image_objects_bbox,
            "objects_token": objects_token,
            "objects_type": objects_type,
            "cam_intrinsic": camera_intrinsic,
            "ego_translation": ego_translation,
            "ego_rotation": ego_rotation,
            "cam_translation": camera_translation,
            "cam_rotation": camera_rotation,
        }

        with open(os.path.join(args.save_dir, f"frame_{i}_data.json"), "w") as f:
            json.dump(out_dict, f)


if __name__ == "__main__":
    main()