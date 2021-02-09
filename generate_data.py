import argparse
import json
import os
import shutil
import sys

import numpy as np
from lyft_dataset_sdk.utils.data_classes import (
    Box,
    LidarPointCloud,  # NOQA
    RadarPointCloud,
)
from lyft_dataset_sdk.utils.geometry_utils import (
    BoxVisibility,  # NOQA
    box_in_image,
    view_points,
)
from pyquaternion import Quaternion
from tqdm import tqdm

from anno_tool_api import LyftDatasetAnno

save_dir = "tmp/"
lyft_root = "/esat/izar/dgrujici/Datasets/LyftPerception/sample_dataset"

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

    for i, sample_token in enumerate(sample_tokens):
        name = scene_token + "_" + sample_token
        subsequent_sample_tokens = sample_tokens[(i + 1) :]
        if i == 5:
            break
        subsequent_egoposes = []
        for subsequent_sample_token in subsequent_sample_tokens:
            subsequent_sample_rec = lyft.get("sample", subsequent_sample_token)
            subsequent_sample_data = lyft.get(
                "sample_data", subsequent_sample_rec["data"]["LIDAR_TOP"]
            )
            subsequent_egopose = lyft.get(
                "ego_pose", subsequent_sample_data["ego_pose_token"]
            )
            subsequent_egoposes.append(subsequent_egopose)

        sample_rec = lyft.get("sample", sample_token)
        sample_data_camera = lyft.get("sample_data", sample_rec["data"]["CAM_FRONT"])
        sample_data_lidar = lyft.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        current_egopose = lyft.get("ego_pose", sample_data_lidar["ego_pose_token"])

        subsequent_egocenters = []
        subsequent_egoboxes = []
        for subsequent_egopose in subsequent_egoposes:
            subsequent_egobox = Box(
                center=subsequent_egopose["translation"],
                size=[1.730, 4.084, 1.562],
                orientation=Quaternion(subsequent_egopose["rotation"]),
            )
            ypr = Quaternion(current_egopose["rotation"]).yaw_pitch_roll
            yaw = ypr[0]

            subsequent_egobox.translate(-np.array(current_egopose["translation"]))
            subsequent_egobox.rotate_around_origin(
                Quaternion(
                    scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
                ).inverse
            )
            subsequent_egocenter = subsequent_egobox.center[:2].tolist()
            subsequent_egobox = (
                subsequent_egobox.bottom_corners()[:2, :].transpose().tolist()
            )
            subsequent_egocenters.append(subsequent_egocenter)
            subsequent_egoboxes.append(subsequent_egobox)

        anchor_egocenter = subsequent_egocenters[0]
        filtered_subsequent_egocenter = []
        filtered_subsequent_egoboxes = []
        for subsequent_egocenter, subsequent_egobox in zip(
            subsequent_egocenters, subsequent_egoboxes
        ):
            anchor_egocenter = np.array(subsequent_egocenter)

        (
            image_objects_bbox,
            camera_intrinsic,
            camera_translation,
            camera_rotation,
        ) = lyft.explorer.render_sample_camera_data_custom(
            sample_data_token=sample_data_camera["token"],
            out_path=os.path.join(save_dir, f"frontal_{i}.jpg"),
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
            sample_data_token=sample_data_lidar["token"],
            out_path=os.path.join(save_dir, f"top_down_{i}.png"),
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
            "scene_token": scene_token,
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
            "subsequent_egocenters": subsequent_egocenters,
        }

        with open(os.path.join(save_dir, f"frame_{i}_data.json"), "w") as f:
            json.dump(out_dict, f)
