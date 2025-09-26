import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored
from typing import Dict, List, Optional

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.eval_utils import (
    get_vgg_input_imgs,
    load_images_rgb,
    infer_vggt_and_reconstruct,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

# Import visualization components from regular solver
from vggt_slam.solver import Viewer, color_point_cloud_by_confidence

# numpy pretty print options
np.set_printoptions(precision=4, suppress=True)


class FastVGGTSolver:
    """
    FastVGGT-based SLAM solver that uses FastVGGT's optimized inference pipeline
    instead of the regular VGGT model. Handles both SLAM processing and reconstruction.
    """

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

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            # plot depth maps for debugging
            # import matplotlib.pyplot as plt
            # # plot all depth maps in a grid
            # num_depths = depth_map.shape[0]
            # cols = 4
            # rows = (num_depths + cols - 1) // cols
            # plt.figure(figsize=(15, 5 * rows))
            # for i in range(num_depths):
            #     plt.subplot(rows, cols, i + 1)
            #     plt.imshow(depth_map[i, :, :, 0], cmap='plasma')
            #     plt.title(f"Depth Map {i}")
            # plt.tight_layout()
            # plt.show()
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

            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case,
                # and SIM3 and SE3 are also subsets of the SL4 group
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                # apply scale factor to points and poses
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])

            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            print("added between factor", prior_pcd_num, new_pcd_num, H_relative)

        # Create and add submap.
        print(f"Final H_w_submap:\n{H_w_submap}")
        print(f"Final cam_to_world translations: {cam_to_world[:3, :3, 3] if cam_to_world.shape[0] >= 3 else cam_to_world[:, :3, 3]}")
        print(f"Final world_points range: min={world_points.min():.4f}, max={world_points.max():.4f}")

        # Create and add submap.
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf)

        print(f"Submap {self.current_working_submap.get_id()} reference homography:\n{self.current_working_submap.get_reference_homography()}")
        print(colored("=== ADD_POINTS DEBUG END ===", 'red'))

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
            self.graph.increment_loop_closure()

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

        self.map.add_submap(self.current_working_submap)

    def run_predictions(self, image_names, model, max_loops):
        """
        FastVGGT-specific prediction pipeline that uses FastVGGT's optimized inference.
        This replaces the regular VGGT model call with FastVGGT's preprocessing and inference.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use FastVGGT's image loading instead of VGGT-SLAM's
        print(f"Loading images with FastVGGT pipeline: {len(image_names)} images")
        images = load_images_rgb([image_names] if isinstance(image_names, str) else image_names, rotate=cv2.ROTATE_90_CLOCKWISE)

        if not images or len(images) < 1:
            raise ValueError("No valid images loaded")

        images_array = np.stack(images)
        print(f"Loaded images shape: {images_array.shape}")

        # Use FastVGGT's preprocessing
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        print(f"VGG input shape: {vgg_input.shape}, patch dimensions: {patch_width}x{patch_height}")

        # Update model patch dimensions (required for FastVGGT)
        model.update_patch_dimensions(patch_width, patch_height)

        # Check for loop closures (same as regular VGGT-SLAM)
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)

        # Convert vgg_input back to the format expected by submap
        # vgg_input is [S, 3, H, W] in range [0,1], need to convert back to torch tensor
        submap_images = vgg_input  # Keep as is, since submap expects this format
        new_submap.add_all_frames(submap_images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        print(colored(">>>> Frame ids in new submap:", "blue"), new_submap.get_frame_ids())

        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)
        retrieval_frame_ids = self.map.get_frame_ids_from_loops(detected_loops)
        print(colored("retrieval_frame_ids", "green"), retrieval_frame_ids)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(vgg_input.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            vgg_input = torch.cat([vgg_input, image_tensor], dim=0) # Shape (s+n, 3, w, h)
            new_submap.add_all_frames(vgg_input)
            image_names = retrieval_frame_ids + image_names

        self.current_working_submap = new_submap
        print(f">>>>>>>> Processing {len(vgg_input)} images with FastVGGT")

        # Use FastVGGT's inference and reconstruction pipeline
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        depth_conf_thresh = 0.0  # Default depth confidence threshold for FastVGGT

        try:
            (
                depth_np,
                depth_conf_np,
                extrinsic_np,
                intrinsic_np,
                all_world_points,
                all_point_colors,
                all_cam_to_world_mat,
                inference_time_ms,
            ) = infer_vggt_and_reconstruct(
                model, vgg_input, dtype, depth_conf_thresh, image_names
            )
            print(f"FastVGGT inference time: {inference_time_ms:.2f}ms")

        except Exception as e:
            print(f"FastVGGT inference failed: {e}")
            # Fallback to basic model call if FastVGGT pipeline fails
            print("Falling back to basic model call...")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    vgg_input_cuda = vgg_input.to(device).to(dtype)
                    predictions = model(vgg_input_cuda)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], vgg_input.shape[-2:])

            # Convert to the format expected by add_points
            predictions_dict = {
                "images": vgg_input.cpu().numpy(),
                "extrinsic": extrinsic.cpu().numpy().squeeze(0),
                "intrinsic": intrinsic.cpu().numpy().squeeze(0),
                "detected_loops": detected_loops,
            }

            # Add depth or world points based on use_point_map
            if self.use_point_map and "world_points" in predictions:
                predictions_dict["world_points"] = predictions["world_points"].cpu().numpy().squeeze(0)
                predictions_dict["world_points_conf"] = predictions["world_points_conf"].cpu().numpy().squeeze(0)
            else:
                predictions_dict["depth"] = predictions["depth"].cpu().numpy().squeeze(0)
                predictions_dict["depth_conf"] = predictions["depth_conf"].cpu().numpy().squeeze(0)

            return predictions_dict

        # Build the prediction dictionary in VGGT-SLAM format
        predictions = {
            "images": vgg_input.cpu().numpy() if torch.is_tensor(vgg_input) else vgg_input,
            "extrinsic": extrinsic_np,
            "intrinsic": intrinsic_np,
            "detected_loops": detected_loops,
            "depth": depth_np,
            "depth_conf": depth_conf_np,
        }

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    
        return predictions

    def run_reconstruction(
        self,
        image_names: List[str],
        model,
        depth_conf_thresh: float = 3.0,
    ):
        """
        Run FastVGGT reconstruction directly (batch mode).
        This is for reconstruction-only mode, bypassing SLAM processing.
        """
        print(f"Running FastVGGT reconstruction on {len(image_names)} images...")

        # Load images with FastVGGT preprocessing
        images = load_images_rgb(image_names, rotate=270)
        if not images or len(images) < 3:
            raise ValueError("Need at least 3 valid images for reconstruction")

        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)

        # Update model patch dimensions
        model.update_patch_dimensions(patch_width, patch_height)

        # Run FastVGGT inference and reconstruction
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        (
            depth_np,
            depth_conf_np,
            extrinsic_np,
            intrinsic_np,
            all_world_points,
            all_point_colors,
            all_cam_to_world_mat,
            inference_time_ms,
        ) = infer_vggt_and_reconstruct(
            model, vgg_input, dtype, depth_conf_thresh, image_names
        )

        return {
            'cam_to_world_matrices': all_cam_to_world_mat,
            'world_points': all_world_points,
            'point_colors': all_point_colors,
            'intrinsics': intrinsic_np,
            'extrinsics': extrinsic_np,
            'inference_time_ms': inference_time_ms,
        }