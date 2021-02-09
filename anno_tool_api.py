# Lyft Dataset SDK.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import (
    Box,
    LidarPointCloud,
    RadarPointCloud,
)  # NOQA
from lyft_dataset_sdk.utils.geometry_utils import (
    BoxVisibility,
    box_in_image,
    view_points,
)  # NOQA

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("LyftDataset sdk only supports Python version 3.")


class LyftDatasetAnno(LyftDataset):
    """Database class for Lyft Dataset to help query and retrieve information from the database."""

    def __init__(
        self,
        data_path: str,
        json_path: str,
        verbose: bool = True,
        map_resolution: float = 0.1,
    ):
        """Loads database and creates reverse indexes and shortcuts.

        Args:
            data_path: Path to the tables and data.
            json_path: Path to the folder with json files
            verbose: Whether to print status messages during load.
            map_resolution: Resolution of maps (meters).
        """
        super().__init__(data_path, json_path, verbose, map_resolution)

        # Initialize LyftDatasetExplorer class
        self.explorer = LyftDatasetExplorerAnno(self)

    def get_sample_data(
        self,
        sample_data_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        selected_anntokens: List[str] = None,
        flat_vehicle_coordinates: bool = False,
    ) -> Tuple[Path, List[Box], np.array]:
        """Returns the data path as well as all annotations related to that sample_data.
        The boxes are transformed into the current sensor's coordinate frame.

        Args:
            sample_data_token: Sample_data token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            selected_anntokens: If provided only return the selected annotation.
            flat_vehicle_coordinates: Instead of current sensor's coordinate frame, use vehicle frame which is
        aligned to z-plane in world

        Returns: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)

        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", cs_record["sensor_token"])
        pose_record = self.get("ego_pose", sd_record["ego_pose_token"])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            image_size = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            image_size = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane
                ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
                yaw = ypr[0]

                box.translate(-np.array(pose_record["translation"]))
                box.rotate_around_origin(
                    Quaternion(
                        scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
                    ).inverse
                )

            else:
                # Move box to ego vehicle coord system
                box.translate(-np.array(pose_record["translation"]))
                box.rotate_around_origin(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system
                box.translate(-np.array(cs_record["translation"]))
                box.rotate_around_origin(Quaternion(cs_record["rotation"]).inverse)

            if sensor_record["modality"] == "camera" and not box_in_image(
                box, cam_intrinsic, image_size, vis_level=box_vis_level
            ):
                box = None

            box_list.append(box)

        return data_path, box_list, cam_intrinsic


class LyftDatasetExplorerAnno(LyftDatasetExplorer):
    """Helper class to list and visualize Lyft Dataset data. These are meant to serve as tutorials and templates for
    working with the data."""

    def __init__(self, lyftd: LyftDataset):
        super().__init__(lyftd)
        self.lyftd = lyftd

    def render_sample(
        self,
        token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        nsweeps: int = 1,
        out_path: str = None,
    ) -> None:
        """Render all LIDAR and camera sample_data in sample along with annotations.

        Args:
            token: Sample token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            nsweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.

        Returns:

        """
        record = self.lyftd.get("sample", token)

        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        nonradar_data = {}
        for channel, token in record["data"].items():
            sd_record = self.lyftd.get("sample_data", token)
            sensor_modality = sd_record["sensor_modality"]
            if sensor_modality in ["lidar", "camera"]:
                nonradar_data[channel] = token
            else:
                radar_data[channel] = token

        num_radar_plots = 1 if len(radar_data) > 0 else 0

        # Create plots.
        n = num_radar_plots + len(nonradar_data)
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        if len(radar_data) > 0:
            # Plot radar into a single subplot.
            ax = axes[0, 0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data(
                    sd_token,
                    with_annotations=i == 0,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    num_sweeps=nsweeps,
                )
            ax.set_title("Fused RADARs")

        # Plot camera and lidar in separate subplots.
        for (_, sd_token), ax in zip(
            nonradar_data.items(), axes.flatten()[num_radar_plots:]
        ):
            self.render_sample_data(
                sd_token,
                box_vis_level=box_vis_level,
                ax=ax,
                num_sweeps=nsweeps,
            )

        axes.flatten()[-1].axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if out_path is not None:
            plt.savefig(out_path)

    def render_ego_centric_map(
        self, sample_data_token: str, axes_limit: float = 40, ax: Axes = None
    ) -> None:
        """Render map centered around the associated ego pose.

        Args:
            sample_data_token: Sample_data token.
            axes_limit: Axes limit measured in meters.
            ax: Axes onto which to render.

        """

        def crop_image(
            image: np.array, x_px: int, y_px: int, axes_limit_px: int
        ) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        sd_record = self.lyftd.get("sample_data", sample_data_token)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        sample = self.lyftd.get("sample", sd_record["sample_token"])
        scene = self.lyftd.get("scene", sample["scene_token"])
        log = self.lyftd.get("log", scene["log_token"])
        map = self.lyftd.get("map", log["map_token"])
        map_mask = map["mask"]

        pose = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
        pixel_coords = map_mask.to_pixel_coords(
            pose["translation"][0], pose["translation"][1]
        )

        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()

        cropped = crop_image(
            mask_raster,
            pixel_coords[0],
            pixel_coords[1],
            int(scaled_limit_px * math.sqrt(2)),
        )

        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        ego_centric_map = crop_image(
            rotated_cropped,
            rotated_cropped.shape[1] / 2,
            rotated_cropped.shape[0] / 2,
            scaled_limit_px,
        )
        ax.imshow(
            ego_centric_map,
            extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
            cmap="gray",
            vmin=0,
            vmax=150,
        )

    def render_sample_data(
        self,
        sample_data_token: str,
        with_annotations: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Axes = None,
        num_sweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = False,
    ):
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            with_annotations: Whether to draw annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            axes_limit: Axes limit for lidar and radar (measured in meters).
            ax: Axes onto which to render.
            num_sweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.
            underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.

        """

        # Get sensor modality.
        sd_record = self.lyftd.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        if sensor_modality == "lidar":
            # Get boxes in lidar frame.
            _, boxes, _ = self.lyftd.get_sample_data(
                sample_data_token,
                box_vis_level=box_vis_level,
                flat_vehicle_coordinates=True,
            )

            # Get aggregated point cloud in lidar frame.
            sample_rec = self.lyftd.get("sample", sd_record["sample_token"])
            chan = sd_record["channel"]
            ref_chan = "LIDAR_TOP"
            pc, times = LidarPointCloud.from_file_multisweep(
                self.lyftd, sample_rec, chan, ref_chan, num_sweeps=num_sweeps
            )

            # Compute transformation matrices for lidar point cloud
            cs_record = self.lyftd.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )
            pose_record = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
            vehicle_from_sensor = np.eye(4)
            vehicle_from_sensor[:3, :3] = Quaternion(
                cs_record["rotation"]
            ).rotation_matrix
            vehicle_from_sensor[:3, 3] = cs_record["translation"]

            ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            rot_vehicle_flat_from_vehicle = np.dot(
                Quaternion(
                    scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]
                ).rotation_matrix,
                Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
            )

            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            if underlay_map:
                self.render_ego_centric_map(
                    sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax
                )

            # Show point cloud.
            points = view_points(
                pc.points[:3, :],
                np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor),
                normalize=False,
            )
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="red")

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == "radar":
            # Get boxes in lidar frame.
            sample_rec = self.lyftd.get("sample", sd_record["sample_token"])
            lidar_token = sample_rec["data"]["LIDAR_TOP"]
            _, boxes, _ = self.lyftd.get_sample_data(
                lidar_token, box_vis_level=box_vis_level
            )

            # Get aggregated point cloud in lidar frame.
            # The point cloud is transformed to the lidar frame for visualization purposes.
            chan = sd_record["channel"]
            ref_chan = "LIDAR_TOP"
            pc, times = RadarPointCloud.from_file_multisweep(
                self.lyftd, sample_rec, chan, ref_chan, num_sweeps=num_sweeps
            )

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point
            # cloud.
            radar_cs_record = self.lyftd.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )
            lidar_sd_record = self.lyftd.get("sample_data", lidar_token)
            lidar_cs_record = self.lyftd.get(
                "calibrated_sensor", lidar_sd_record["calibrated_sensor_token"]
            )
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(
                Quaternion(radar_cs_record["rotation"]).rotation_matrix, velocities
            )
            velocities = np.dot(
                Quaternion(lidar_cs_record["rotation"]).rotation_matrix.T, velocities
            )
            velocities[2, :] = np.zeros(pc.points.shape[1])

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Show point cloud.
            points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            sc = ax.scatter(points[0, :], points[1, :], c=colors, s=3)

            # Show velocities.
            points_vel = view_points(
                pc.points[:3, :] + velocities, np.eye(4), normalize=False
            )
            max_delta = 10
            deltas_vel = points_vel - points
            deltas_vel = 3 * deltas_vel  # Arbitrary scaling
            deltas_vel = np.clip(
                deltas_vel, -max_delta, max_delta
            )  # Arbitrary clipping
            colors_rgba = sc.to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(
                    points[0, i],
                    points[1, i],
                    deltas_vel[0, i],
                    deltas_vel[1, i],
                    color=colors_rgba[i],
                )

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="black")

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == "camera":
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level
            )

            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(
                        ax, view=camera_intrinsic, normalize=True, colors=(c, c, c)
                    )

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis("off")
        ax.set_title(sd_record["channel"])
        ax.set_aspect("equal")

        if out_path is not None:
            num = len([name for name in os.listdir(out_path)])
            out_path = out_path + str(num).zfill(5) + "_" + sample_data_token + ".png"
            plt.savefig(out_path)
            plt.close("all")
            return out_path

    def render_ego_centric_map_custom(
        self,
        sample_data_token: str,
        limit_left: float = 10,
        limit_right: float = 70,
        limit_top: float = 40,
        limit_bottom: float = 40,
        ax: Axes = None,
        out_path: str = None,
    ) -> None:
        """Render map centered around the associated ego pose.

        Args:
            sample_data_token: Sample_data token.
            axes_limit: Axes limit measured in meters.
            ax: Axes onto which to render.

        """

        def crop_image(
            image: np.array,
            x_px: int,
            y_px: int,
            limit_left: int,
            limit_right: int,
            limit_top: int,
            limit_bottom: int,
        ) -> np.array:
            x_min = int(x_px - limit_left)
            x_max = int(x_px + limit_right)
            y_min = int(y_px - limit_top)
            y_max = int(y_px + limit_bottom)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        sd_record = self.lyftd.get("sample_data", sample_data_token)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        sample = self.lyftd.get("sample", sd_record["sample_token"])
        scene = self.lyftd.get("scene", sample["scene_token"])
        log = self.lyftd.get("log", scene["log_token"])
        map_mask = self.lyftd.get("map", log["map_token"])["mask"]

        pose = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
        pixel_coords = map_mask.to_pixel_coords(
            pose["translation"][0], pose["translation"][1]
        )

        scaled_limit_left = int(limit_left * (1.0 / map_mask.resolution))
        scaled_limit_right = int(limit_right * (1.0 / map_mask.resolution))
        scaled_limit_top = int(limit_top * (1.0 / map_mask.resolution))
        scaled_limit_bottom = int(limit_bottom * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()

        axes_limit = max(
            [
                scaled_limit_left,
                scaled_limit_right,
                scaled_limit_top,
                scaled_limit_bottom,
            ]
        )
        cropped = crop_image(
            mask_raster,
            pixel_coords[0],
            pixel_coords[1],
            axes_limit * 3,
            axes_limit * 3,
            axes_limit * 3,
            axes_limit * 3,
        )

        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        ego_centric_map = crop_image(
            rotated_cropped,
            rotated_cropped.shape[1] / 2,
            rotated_cropped.shape[0] / 2,
            scaled_limit_left,
            scaled_limit_right,
            scaled_limit_top,
            scaled_limit_bottom,
        )
        ax.imshow(
            ego_centric_map,
            extent=[-limit_left, limit_right, -limit_top, limit_bottom],
            cmap="gray",
            vmin=0,
            vmax=150,
        )

        if out_path is not None:
            plt.savefig(out_path)

        return [-limit_left, -limit_top, limit_right, limit_bottom]

    def render_sample_camera_data_custom(
        self,
        sample_data_token: str,
        with_annotations: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = None,
    ):
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            with_annotations: Whether to draw annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            out_path: Optional path to save the rendered figure to disk.
            underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.

        """

        # Get sensor modality.
        sd_record = self.lyftd.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]
        assert sensor_modality == "camera", "The sensor modality needs to be the camera"

        # Load boxes and image.
        data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
            sample_data_token, box_vis_level=box_vis_level
        )

        # data = Image.open(data_path).resize((1600, 900), resample=Image.NEAREST)
        data = Image.open(data_path)
        data = np.array(data)

        # Show boxes.
        if with_annotations:
            for box in boxes:
                if box:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render_cv2(
                        data, view=camera_intrinsic, normalize=True, colors=(c, c, c)
                    )

        if out_path is not None:
            data = Image.fromarray(data)
            data.save(out_path)

        image_objects_bbox = []
        for box in boxes:
            if box:
                image_objects_bbox.append(
                    view_points(box.corners(), view=camera_intrinsic, normalize=True)[
                        :2, :
                    ]
                    .transpose()
                    .tolist()
                )
            else:
                image_objects_bbox.append(None)

        camera_intrinsic = camera_intrinsic.tolist()

        cs_record = self.lyftd.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        camera_translation = cs_record["translation"]
        camera_rotation = Quaternion(cs_record["rotation"]).rotation_matrix.tolist()

        return (
            image_objects_bbox,
            camera_intrinsic,
            camera_translation,
            camera_rotation,
        )

    def render_sample_lidar_data_custom(
        self,
        sample_data_token: str,
        with_annotations: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        limit_left: float = 10,
        limit_right: float = 70,
        limit_top: float = 40,
        limit_bottom: float = 40,
        out_path: str = None,
    ):
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            with_annotations: Whether to draw annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            axes_limit: Axes limit for lidar and radar (measured in meters).
            out_path: Optional path to save the rendered figure to disk.

        """

        # Get sensor modality.
        sd_record = self.lyftd.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]
        assert sensor_modality == "lidar", "The sensor modality needs to be the lidar"

        # Get boxes in lidar frame.
        _, boxes, _ = self.lyftd.get_sample_data(
            sample_data_token,
            box_vis_level=box_vis_level,
            flat_vehicle_coordinates=True,
        )

        # Compute transformation matrices for lidar point cloud
        cs_record = self.lyftd.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        pose_record = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
        vehicle_from_sensor = np.eye(4)
        vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
        vehicle_from_sensor[:3, 3] = cs_record["translation"]

        ego_bbox = Box(
            center=pose_record["translation"],
            size=[1.730, 4.084, 1.562],
            orientation=Quaternion(pose_record["rotation"]),
        )
        ego_translation = pose_record["translation"]
        ego_rotation = Quaternion(pose_record["rotation"]).rotation_matrix.tolist()
        ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]

        ego_bbox.translate(-np.array(pose_record["translation"]))
        ego_bbox.rotate_around_origin(
            Quaternion(
                scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]
            ).inverse
        )

        rot_vehicle_flat_from_vehicle = np.dot(
            Quaternion(
                scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]
            ).rotation_matrix,
            Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
        )
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

        # Init axes.
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        map_patch = self.render_ego_centric_map_custom(
            sample_data_token=sample_data_token,
            limit_left=limit_left,
            limit_right=limit_right,
            limit_top=limit_top,
            limit_bottom=limit_bottom,
            ax=ax,
        )

        # Show ego vehicle.
        if with_annotations:
            ego_bbox.render(ax, view=np.eye(4), colors=("red", "red", "red"))

        # Show boxes.
        if with_annotations:
            for box in boxes:
                c = np.array(self.get_color(box.name)) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(-limit_left, limit_right)
        ax.set_ylim(-limit_top, limit_bottom)

        ax.axis("off")
        if out_path is not None:
            plt.savefig(
                out_path,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close("all")

        ego_bbox = ego_bbox.bottom_corners()[:2, :].transpose().tolist()
        ego_translation = [0.0, 0.0, 0.0]  # because it's centered
        ego_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]  # because it's already rotated
        map_objects_center = []
        map_objects_elevation = []
        map_objects_bbox = []
        objects_type = []
        objects_token = []

        for box in boxes:
            map_objects_center.append(box.center[:2].tolist())
            objects_type.append(box.name)
            objects_token.append(box.token)
            map_objects_bbox.append(box.bottom_corners()[:2, :].transpose().tolist())
            map_objects_elevation.append(box.bottom_corners()[2, 0])

        return (
            ego_bbox,
            ego_translation,
            ego_rotation,
            map_objects_center,
            map_objects_elevation,
            map_objects_bbox,
            objects_type,
            objects_token,
            map_patch,
        )
