import argparse
import json
import os

import numpy as np
from lyft_dataset_sdk.utils.data_classes import (
    Box,
)
from lyft_dataset_sdk.utils.geometry_utils import (
    BoxVisibility,
    box_in_image,
    view_points,
)
from pyquaternion import Quaternion
from tqdm import tqdm

from anno_tool_api import LyftDatasetAnno
from shapely.geometry import LineString
from typing import List, Tuple, Union


def return_side_points(
    cur_point: Union[Tuple, List],
    prev_point: Union[Tuple, List, None] = None,
    thickness=2.0,
):
    line = LineString([cur_point, prev_point])
    left = line.parallel_offset(thickness / 2, "left")
    right = line.parallel_offset(thickness / 2, "right")
    return [left.boundary[1].x, left.boundary[1].y], [
        right.boundary[0].x,
        right.boundary[0].y,
    ]


def compute_polygon_from_path(path_nodes: List[List], thickness=2.0):
    forward = []
    backward = []
    for i in range(1, len(path_nodes)):
        left, right = return_side_points(path_nodes[i], path_nodes[i - 1], thickness)
        forward.append(left)
        backward.append(right)
    backward = backward[::-1]
    return forward + backward


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
    parser.add_argument(
        "--first_ind",
        type=int,
        default=-1,
        help="Index of the first scene that is processed.",
    )
    parser.add_argument(
        "--last_ind",
        type=int,
        default=-1,
        help="Index of the last scene that is processed.",
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

    first_ind = args.first_ind if args.first_ind > -1 else 0
    last_ind = args.last_ind if args.last_ind > -1 else len(scene_recs)
    print(f"Extracting from scene {first_ind} to scene {last_ind}...")
    for scene_rec in scene_recs[first_ind:last_ind]:
        scene_token = scene_rec["token"]
        print(f"\tExtracting scene {scene_token}")
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

        for i, sample_token in enumerate(sample_tokens):
            subsequent_sample_tokens = sample_tokens[(i + 1) :]
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
            sample_data_camera = lyft.get(
                "sample_data", sample_rec["data"]["CAM_FRONT"]
            )
            sample_data_lidar = lyft.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
            current_egopose = lyft.get("ego_pose", sample_data_lidar["ego_pose_token"])

            (
                image_objects_bbox,
                camera_intrinsic,
                camera_translation,
                camera_rotation,
            ) = lyft.explorer.render_sample_camera_data_custom(
                sample_data_token=sample_data_camera["token"],
                out_path=os.path.join(save_dir, scene_token, f"{sample_token}.jpg"),
                with_annotations=False,
            )

            ################################
            sd_record = lyft.get("sample_data", sample_data_camera["token"])
            cs_record = lyft.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            image_size = (sd_record["width"], sd_record["height"])

            subsequent_egocenters = []
            for subsequent_egopose in subsequent_egoposes:
                subsequent_egobox = Box(
                    center=subsequent_egopose["translation"],
                    size=[1.730, 4.084, 1.562],
                    orientation=Quaternion(subsequent_egopose["rotation"]),
                )

                # Move box to ego vehicle coord system
                subsequent_egobox.translate(-np.array(current_egopose["translation"]))
                subsequent_egobox.rotate_around_origin(
                    Quaternion(current_egopose["rotation"]).inverse
                )

                #  Move box to sensor coord system
                subsequent_egobox.translate(-np.array(cs_record["translation"]))
                subsequent_egobox.rotate_around_origin(
                    Quaternion(cs_record["rotation"]).inverse
                )

                if box_in_image(
                    subsequent_egobox,
                    cam_intrinsic,
                    image_size,
                    vis_level=BoxVisibility.ANY,
                ):
                    subsequent_egocenter = (
                        subsequent_egobox.bottom_corners()
                        .mean(axis=1)
                        .transpose()
                        .tolist()
                    )
                    subsequent_egocenters.append(subsequent_egocenter)

            # Project subsequent egocenters to scenes
            if len(subsequent_egocenters):
                # Draw path width

                subsequent_egocenters = np.array(subsequent_egocenters)
                subsequent_egocenters_x = subsequent_egocenters[:, 0]
                subsequent_egocenters_y = subsequent_egocenters[:, 2]
                subsequent_egocenters_z = subsequent_egocenters[:, 1]
                subsequent_egocenters_xy = np.concatenate(
                    (
                        subsequent_egocenters_x[:, None],
                        subsequent_egocenters_y[:, None],
                    ),
                    axis=1,
                )

                subsequent_egocenters_xy = compute_polygon_from_path(
                    subsequent_egocenters_xy.tolist(), thickness=3.5
                )
                subsequent_egocenters_xy = np.array(subsequent_egocenters_xy)
                subsequent_egocenters_x = subsequent_egocenters_xy[:, 0]
                subsequent_egocenters_y = subsequent_egocenters_xy[:, 1]
                subsequent_egocenters_z = np.concatenate(
                    (subsequent_egocenters_z[1:], subsequent_egocenters_z[1:][::-1])
                )
                subsequent_egocenters = np.concatenate(
                    (
                        subsequent_egocenters_x[:, None],
                        subsequent_egocenters_z[:, None],
                        subsequent_egocenters_y[:, None],
                    ),
                    axis=1,
                )

                subsequent_egocenters = np.array(subsequent_egocenters).transpose()
                subsequent_egocenters_frontal = (
                    view_points(
                        subsequent_egocenters, view=cam_intrinsic, normalize=True
                    )[:2, :]
                    .transpose()
                    .astype(int)
                    .tolist()
                )
            else:
                subsequent_egocenters_frontal = []

            # image = cv2.imread(
            #     os.path.join(save_dir, scene_token, f"{sample_token}.jpg")
            # )
            # start_egocenter = subsequent_egocenters_frontal[0]
            # for egocenter in subsequent_egocenters_frontal[1:]:
            #     image = cv2.line(
            #         image,
            #         tuple(start_egocenter),
            #         tuple(egocenter),
            #         color=(0, 255, 0),
            #         thickness=3,
            #     )
            #     start_egocenter = egocenter

            # os.makedirs(os.path.join(save_dir, "jaja"), exist_ok=True)
            # cv2.imwrite(os.path.join(save_dir, "jaja", f"path_{i}.jpg"), image)

            out_dict = json.load(
                open(os.path.join(save_dir, scene_token, f"frame_{i}_data.json"), "r")
            )
            out_dict["path_points_frontal"] = subsequent_egocenters_frontal

            with open(
                os.path.join(save_dir, scene_token, f"frame_{i}_data.json"), "w"
            ) as f:
                json.dump(out_dict, f)
