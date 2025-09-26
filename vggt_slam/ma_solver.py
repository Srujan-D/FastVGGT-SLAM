import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored
import os

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

from collections import defaultdict
from typing import List, Dict

import sys
sys.path.append("/home/srujan/work/map-anything")
from mapanything.utils.image import load_images

# numpy pretty print options
np.set_printoptions(precision=4, suppress=True)

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"

    # Normalize confidence to [0, 1]
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    
    # Map to colors using matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]  # Drop alpha channel

    # Assign to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # Global toggle for all frames and frustums
        self.gui_show_frames = self.server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Store frames and frustums by submap
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float=0.5) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Downsample for visualization with `image_scale`
            img_resized = cv2.resize(
                img,
                (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale)),
                interpolation=cv2.INTER_AREA
            )

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible



class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        vis_stride: int = 1,         # represents how much the visualized point clouds are sparsified
        vis_point_size: float = 0.001):
        
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3
        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None

        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        print("Starting viser server...")

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_"+name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        # NOTE(hlim): `stride` is used only to reduce the visualization cost in viser,
        # and does not affect the underlying point cloud data.
        points_in_world_frame = submap.get_points_in_world_frame(stride = self.vis_stride)
        points_colors = submap.get_points_colors(stride = self.vis_stride)
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, self.vis_point_size)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)

        # estimate focal length from points
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)

            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)

            # TODO conf should be using the threshold in its own submap
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())

            # Keep original world points to prevent scale accumulation
            world_points_original = world_points.copy()

            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case,
                # and SIM3 and SE3 are also subsets of the SL4 group

                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)

                # Transform prior points to current frame
                transformed_prior = (T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3]
                prior_norms = np.linalg.norm(transformed_prior, axis=1)
                current_norms = np.linalg.norm(current_pts[good_mask], axis=1)

                # Check for problematic ratios with better outlier handling
                # Filter out points that are too close to origin (unreliable)
                min_distance = 0.1  # Minimum distance threshold
                valid_mask = (prior_norms > min_distance) & (current_norms > min_distance)

                if valid_mask.sum() < 100:  # Need minimum points for reliable estimate
                    print(colored("Warning: Too few reliable points for scale estimation, using fallback", 'red'))
                    scale_factor = 1.0
                else:
                    valid_prior = prior_norms[valid_mask]
                    valid_current = current_norms[valid_mask]
                    ratios = valid_prior / valid_current

                    # Use robust statistics to handle outliers
                    median_ratio = np.median(ratios)
                    mad = np.median(np.abs(ratios - median_ratio))  # Median absolute deviation

                    # Filter outliers using MAD (more robust than std)
                    outlier_threshold = mad
                    inlier_mask = np.abs(ratios - median_ratio) < outlier_threshold

                    if inlier_mask.sum() < 50:
                        print(colored("Warning: Too many outliers, using median", 'yellow'))
                        scale_factor = median_ratio
                    else:
                        clean_ratios = ratios[inlier_mask]
                        scale_factor = np.median(clean_ratios)  # Use median instead of mean

                    print(f"Scale ratios (before filtering): min={ratios.min():.4f}, max={ratios.max():.4f}, median={np.median(ratios):.4f}")
                    print(f"Valid points: {valid_mask.sum()}/{len(prior_norms)}, Inliers: {inlier_mask.sum() if 'inlier_mask' in locals() else 'N/A'}")
                    print(f"MAD: {mad:.4f}, Outlier threshold: {outlier_threshold:.4f}")

                print(colored(f"COMPUTED SCALE FACTOR: {scale_factor}", 'green'))
                print(colored("=========================", 'cyan'))

                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                # apply scale factor to points and poses
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            # Visualize the point clouds
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(self.prior_pcd)
            # pcd1 = color_point_cloud_by_confidence(pcd1, self.prior_conf)
            # pcd2 = o3d.geometry.PointCloud()
            # current_pts = world_points[0,...].reshape(-1, 3)
            # points = apply_homography(H_relative, current_pts)
            # pcd2.points = o3d.utility.Vector3dVector(points)
            # # pcd2 = color_point_cloud_by_confidence(pcd2, conf_flat, cmap='jet')
            # o3d.visualization.draw_geometries([pcd1, pcd2])

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)

            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            print("added between factor", prior_pcd_num, new_pcd_num, H_relative)

        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf) # TODO should make this work for point cloud conf as well

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

            # print("relative_pose factor added", relative_pose)

            # Visualize query and detected frames
            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(loop.detected_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(loop.query_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()

            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()


        self.map.add_submap(self.current_working_submap)


    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, args=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names, rotate=270).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Check for loop closures
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        # new_submap.add_all_frames(images)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))

        print(colored(">>>> Frame ids in new submap:", "blue"), new_submap.get_frame_ids())
        # TODO implement this
        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)
        retrieval_frame_ids = self.map.get_frame_ids_from_loops(detected_loops)
        print(colored("retrieval_frame_ids", "green"), retrieval_frame_ids)

        base_name = image_names[0].split("/")[:-1]
        base_name = "/".join(base_name) + "/"
        retrieved_frame_names = []
        for frame_ids in retrieval_frame_ids:
            retrieved_frame_names.extend([base_name + f"{int(frame_id):06d}.png" for frame_id in frame_ids])

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            new_submap.add_all_frames(images)
            image_names = retrieved_frame_names + image_names
        print(">>>>>>>> len(images)", len(images), " ----------- len(image_names)", len(image_names))
        self.current_working_submap = new_submap

        if args is None:
            # plot all images in a grid
            num_images = images.shape[0]
            num_cols = 5
            num_rows = (num_images + num_cols - 1) // num_cols + 1
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < num_images:
                        axes[i, j].imshow(images[idx].cpu().numpy().transpose(1, 2, 0))
                        axes[i, j].set_title(f"Image {idx}")
                        axes[i, j].axis("off")
                    else:
                        axes[i, j].axis("off")
            plt.tight_layout()
            plt.show()
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            predictions["detected_loops"] = detected_loops

            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

            # DEBUG: Print VGGT predictions to understand what it returns
            print(colored("=== VGGT PREDICTIONS DEBUG ===", 'yellow'))
            print(f"VGGT prediction keys: {list(predictions.keys())}")
            for key, value in predictions.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(colored("==============================", 'yellow'))

            return predictions # keys: dict_keys(['frame_id', 'submap_id', 'local_pose', 'image', 'points_3d', 'intrinsics', 'conf_mask'])

        else: # USING MapAnything instead of VGGT
            
            views = load_images(image_names, rotate=270)

            # # Plot all images in views as subplots,
            # images = [img.cpu().numpy() for img in images] # list of (3, H, W) arrays
            
            # # Combine views and images for plotting
            # all_images_to_plot = []
            
            # # Process views
            # for i, view in enumerate(views):
            #     img = view['img']
            #     if len(img.shape) == 4:  # 1, 3, H, W
            #         img = img[0]
            #     if isinstance(img, torch.Tensor):
            #         img = img.permute(1, 2, 0)  # H, W, 3
            #     else:
            #         img = img.transpose(1, 2, 0)  # H, W, 3
                
            #     # Normalize to [0, 255]
            #     img = (img - img.min()) / (img.max() - img.min()) * 255.0
            #     if isinstance(img, torch.Tensor):
            #         img = img.cpu().numpy()
            #     img = img.astype(np.uint8)
                
            #     all_images_to_plot.append(img)
            
            # # Process images
            # for i, img in enumerate(images):
            #     if isinstance(img, torch.Tensor):
            #         img = img.cpu().numpy()
                
            #     # Convert from (3, H, W) to (H, W, 3)
            #     if img.shape[0] == 3:
            #         img = img.transpose(1, 2, 0)
                
            #     # Normalize to [0, 255]
            #     img = (img - img.min()) / (img.max() - img.min()) * 255.0
            #     img = img.astype(np.uint8)
                
            #     # # Rotate 90 degrees anticlockwise
            #     # img = np.rot90(img, k=1)
                
            #     all_images_to_plot.append(img)
            
            # # Plot with num_views columns
            # num_views = len(views)
            # num_total = len(all_images_to_plot)
            # num_rows = (num_total + num_views - 1) // num_views  # Ceiling division
            
            # fig, axes = plt.subplots(num_rows, num_views, figsize=(num_views * 4, num_rows * 6))
            
            # # Handle case where we have only one row
            # if num_rows == 1:
            #     axes = axes.reshape(1, -1)
            # elif num_views == 1:
            #     axes = axes.reshape(-1, 1)
            
            # for i, img in enumerate(all_images_to_plot):
            #     row = i // num_views
            #     col = i % num_views
            #     axes[row, col].imshow(img)
            #     axes[row, col].axis('off')
            #     if i < len(views):
            #         axes[row, col].set_title(f'View {i+1}')
            #     else:
            #         axes[row, col].set_title(f'Image {i-len(views)+1}')
            
            # # Hide empty subplots
            # for i in range(num_total, num_rows * num_views):
            #     row = i // num_views
            #     col = i % num_views
            #     axes[row, col].axis('off')
            
            # plt.tight_layout()
            # plt.show()


            with torch.no_grad():
                outputs = model.infer(
                    views, memory_efficient_inference=args.memory_efficient_inference
                    )


                predictions = {}

                # Initialize lists to collect data from all outputs
                all_images = []
                all_world_points = []
                all_world_points_conf = []
                all_depth = []
                all_depth_conf = []
                all_extrinsics = []
                all_intrinsics = []

                for output in outputs:
                    # Extract batch data and squeeze batch dimension if B=1
                    images_batch = output['img_no_norm'].permute(0, 3, 1, 2).cpu().numpy()  # (B, 3, H, W)
                    world_points_batch = output['pts3d'].cpu().numpy()  # (B, H, W, 3)
                    conf_batch = output['conf'].cpu().numpy()  # (B, H, W)
                    depth_batch = output['depth_z'].cpu().numpy()  # (B, H, W, 1)

                    # Camera poses from MapAnything are world-to-camera, need camera-to-world
                    camera_poses_batch = output['camera_poses'].cpu().numpy()  # (B, 4, 4)
                    intrinsics_batch = output['intrinsics'].cpu().numpy()  # (B, 3, 3)

                    # Process each item in the batch
                    for i in range(images_batch.shape[0]):
                        all_images.append(images_batch[i])  # (3, H, W)
                        all_depth.append(depth_batch[i])  # (H, W, 1)
                        all_depth_conf.append(conf_batch[i])  # (H, W)

                        # CRITICAL FIX: Convert camera poses properly
                        # MapAnything gives world-to-camera, but VGGT-SLAM expects camera-to-world
                        cam_pose_4x4 = camera_poses_batch[i]  # (4, 4) world-to-camera
                        cam_to_world = np.linalg.inv(cam_pose_4x4)  # Invert to get camera-to-world
                        all_extrinsics.append(cam_to_world[:3, :4])  # (3, 4)
                        all_intrinsics.append(intrinsics_batch[i])  # (3, 3)

                        # MAJOR FIX: Don't use MapAnything's pts3d directly!
                        # Instead, compute world points from depth + camera pose like VGGT does
                        # This ensures consistent coordinate systems
                        depth_map = depth_batch[i]  # (H, W, 1)
                        K = intrinsics_batch[i]  # (3, 3)
                        cam_to_world_3x4 = cam_to_world[:3, :4]  # (3, 4)

                        # Compute world points using the same method as VGGT
                        world_pts = unproject_depth_map_to_point_map(
                            depth_map[np.newaxis, :, :, :],  # Add batch dim (1, H, W, 1)
                            cam_to_world_3x4[np.newaxis, :, :],  # Add batch dim (1, 3, 4)
                            K[np.newaxis, :, :]  # Add batch dim (1, 3, 3)
                        )[0]  # Remove batch dim -> (H, W, 3)

                        all_world_points.append(world_pts)
                        all_world_points_conf.append(conf_batch[i])  # (H, W)

                # Convert lists to numpy arrays with correct shapes for VGGT-SLAM
                predictions["images"] = np.stack(all_images, axis=0)  # (S, 3, H, W)
                predictions["world_points"] = np.stack(all_world_points, axis=0)  # (S, H, W, 3)
                predictions["world_points_conf"] = np.stack(all_world_points_conf, axis=0)  # (S, H, W)
                predictions["depth"] = np.stack(all_depth, axis=0)  # (S, H, W, 1)
                predictions["depth_conf"] = np.stack(all_depth_conf, axis=0)  # (S, H, W)
                predictions["extrinsic"] = np.stack(all_extrinsics, axis=0)  # (S, 3, 4)
                predictions["intrinsic"] = np.stack(all_intrinsics, axis=0)  # (S, 3, 3)
                predictions["detected_loops"] = detected_loops
                predictions["outputs"] = outputs

                # # DEBUG: Print MapAnything data characteristics
                # print(colored("=== MAPANYTHING DATA DEBUG ===", 'magenta'))
                # print(f"Input image names: {image_names}")
                # print(f"Number of MapAnything outputs: {len(outputs)}")
                # print(f"Images shape: {predictions['images'].shape}")
                # print(f"World points shape: {predictions['world_points'].shape}")
                # print(f"Extrinsics shape: {predictions['extrinsic'].shape}")

                # Check world points statistics
                wp = predictions["world_points"]
                # print(f"World points stats: min={wp.min():.4f}, max={wp.max():.4f}, mean={wp.mean():.4f}")
                # print(f"World points norms: min={np.linalg.norm(wp.reshape(-1, 3), axis=1).min():.4f}, max={np.linalg.norm(wp.reshape(-1, 3), axis=1).max():.4f}, mean={np.linalg.norm(wp.reshape(-1, 3), axis=1).mean():.4f}")

                # Check camera poses
                ext = predictions["extrinsic"]
                # print(f"Camera positions (first 3): {ext[:3, :3, 3]}")
                # print(f"Camera translation norms: {np.linalg.norm(ext[:, :3, 3], axis=1)}")

                # Check depth statistics
                depth = predictions["depth"]
                # print(f"Depth stats: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")
                # print(f"Depth range per image: {[f'{depth[i].min():.2f}-{depth[i].max():.2f}' for i in range(min(3, depth.shape[0]))]}")

                print(colored("==============================", 'magenta'))

                return predictions
