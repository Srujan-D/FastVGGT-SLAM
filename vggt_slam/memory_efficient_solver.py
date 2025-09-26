"""
Memory-efficient solver that extends the original Solver with disk caching and GPU memory management.
"""
import os
import gc
import torch
import numpy as np
import gtsam
from typing import Dict, Any, Optional
from termcolor import colored

from .solver import Solver
from .memory_config import MemoryConfig, get_default_memory_config
from .memory_utils import MemoryManager
from .memory_efficient_submap import MemoryEfficientSubmap
from .submap import Submap

from vggt_slam.h_solve import ransac_projective
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3


class MemoryEfficientSolver(Solver):
    """
    Memory-efficient version of Solver with disk caching and GPU cleanup.
    Maintains API compatibility while adding memory optimizations.
    """

    def __init__(self,
                 init_conf_threshold: float,
                 use_point_map: bool = False,
                 visualize_global_map: bool = False,
                 use_sim3: bool = False,
                 gradio_mode: bool = False,
                 vis_stride: int = 1,
                 vis_point_size: float = 0.001,
                 memory_config: Optional[MemoryConfig] = None):

        # Initialize parent class
        super().__init__(
            init_conf_threshold=init_conf_threshold,
            use_point_map=use_point_map,
            visualize_global_map=visualize_global_map,
            use_sim3=use_sim3,
            gradio_mode=gradio_mode,
            vis_stride=vis_stride,
            vis_point_size=vis_point_size
        )

        # Initialize memory management
        self.memory_config = memory_config or get_default_memory_config()
        self.memory_manager = MemoryManager(self.memory_config)
        self.use_memory_efficient_submaps = True

        # Track submaps for memory management
        self._recent_submaps = []

        print(f"MemoryEfficientSolver initialized with config: {self.memory_config}")
        self.memory_manager.log_memory_usage("Solver initialization")

    def run_predictions_with_memory_management(self, image_names, model, max_loops):
        """
        Memory-efficient version of run_predictions with GPU cleanup and caching.
        """
        self.memory_manager.log_memory_usage("Before predictions")

        # Run standard prediction logic but with memory management
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = self._load_images_with_cleanup(image_names, device)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Create memory-efficient submap
        new_pcd_num = self.map.get_largest_key() + 1
        if self.use_memory_efficient_submaps:
            new_submap = MemoryEfficientSubmap(new_pcd_num, self.memory_manager)
        else:
            new_submap = Submap(new_pcd_num)

        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))

        # Loop closure detection
        print(colored(">>>> Frame ids in new submap:", "blue"), new_submap.get_frame_ids())
        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)

        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)

        retrieved_frames = self.map.get_frames_from_loops(detected_loops)
        retrieval_frame_ids = self.map.get_frame_ids_from_loops(detected_loops)
        print(colored("retrieval_frame_ids", "green"), retrieval_frame_ids)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            # Ensure retrieved frames are on same device as images
            device = images.device
            retrieved_frames_on_device = [frame.to(device) if hasattr(frame, 'to') else torch.tensor(frame).to(device)
                                        for frame in retrieved_frames]
            image_tensor = torch.stack(retrieved_frames_on_device)
            images = torch.cat([images, image_tensor], dim=0)
            new_submap.add_all_frames(images)
            image_names = retrieval_frame_ids + image_names

        self.current_working_submap = new_submap

        # VGGT inference with memory management
        self.memory_manager.log_memory_usage("Before VGGT inference")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # Immediate GPU cleanup after inference
        if self.memory_config.cleanup_after_inference:
            self.memory_manager.cleanup_gpu_memory()

        self.memory_manager.log_memory_usage("After VGGT inference")

        # Convert pose encoding
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert predictions to CPU and clean up GPU
        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

        # Additional GPU cleanup
        self.memory_manager.cleanup_gpu_memory()

        predictions["detected_loops"] = detected_loops
        predictions["image_names"] = image_names

        self.memory_manager.log_memory_usage("After processing predictions")

        return predictions

    def add_points_with_caching(self, pred_dict):
        """
        Memory-efficient version of add_points with disk caching.
        """
        self.memory_manager.log_memory_usage("Before add_points")

        # Same logic as original add_points but with memory management
        extrinsics_cam = pred_dict["extrinsic"][0, ...] # shape (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"][0, ...] # shape (S, 3, 3)
        detected_loops = pred_dict["detected_loops"]
        images = pred_dict["images"] if "images" in pred_dict else self.current_working_submap.get_all_frames().cpu().numpy()
        images = images[0, ...]
        if self.use_point_map:
            world_points_map = pred_dict["world_points"]
            conf = pred_dict["world_points_conf"]
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (1, S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (1, S, H, W)
            depth_map = depth_map[0, ...]  # (S, H, W, 1)
            conf = conf[0, ...]  # (S, H, W)

            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Convert and process data
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)

        # Continue with original logic but with memory management
        self._process_submap_alignment_with_memory_management(world_points, conf, cam_to_world)

        # Add data to submap with memory management
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf)

        # Process loop closures (same as original)
        self._process_loop_closures(detected_loops)

        # Add submap to map
        self.map.add_submap(self.current_working_submap)

        # Update recent submaps list for memory management
        self._update_recent_submaps_cache()

        self.memory_manager.log_memory_usage("After add_points")

    def _load_images_with_cleanup(self, image_names, device):
        """Load images with memory management."""
        images = load_and_preprocess_images(image_names, rotate=270).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        # Cleanup after loading
        if self.memory_config.enable_gpu_cleanup:
            torch.cuda.empty_cache()

        return images

    def _process_submap_alignment_with_memory_management(self, world_points, conf, cam_to_world):
        """Process submap alignment with memory optimizations."""
        # Same logic as parent class but with memory management
        new_pcd_num = self.current_working_submap.get_id()

        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)

            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            # Load previous submap data if needed
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())

            if self.use_sim3:
                # SIM(3) alignment with memory management
                H_relative = self._process_sim3_alignment(prior_submap, current_pts, good_mask, world_points, cam_to_world)
            else:
                # Projective alignment
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])

            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            # Update prior data
            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add to graph
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

        # Set reference homography
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)

    def _process_sim3_alignment(self, prior_submap, current_pts, good_mask, world_points, cam_to_world):
        """Process SIM(3) alignment with memory considerations."""
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

        # Apply scale factor
        world_points *= scale_factor
        cam_to_world[:, 0:3, 3] *= scale_factor

        return H_relative

    def _process_loop_closures(self, detected_loops):
        """Process loop closures with memory management."""
        # Same logic as original but with memory-aware loading
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
                # Load point clouds on-demand if using memory-efficient submaps
                detected_submap = self.map.get_submap(loop.detected_submap_id)
                if hasattr(detected_submap, 'get_frame_pointcloud'):
                    points_world_detected = detected_submap.get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                else:
                    # Fallback for regular submaps
                    points_world_detected = detected_submap.get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)

                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)

            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure()

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)

    def _update_recent_submaps_cache(self):
        """Update cache of recent submaps for memory management."""
        if not self.memory_config.keep_recent_submaps_in_memory:
            return

        self._recent_submaps.append(self.current_working_submap.get_id())

        # Keep only recent submaps in memory
        if len(self._recent_submaps) > self.memory_config.recent_submaps_count:
            old_submap_id = self._recent_submaps.pop(0)
            old_submap = self.map.get_submap(old_submap_id)

            # Cache old submap data to disk if it's memory-efficient
            if isinstance(old_submap, MemoryEfficientSubmap):
                self.memory_manager.log_memory_usage(f"Caching old submap {old_submap_id}")
                # The submap will automatically cache its data when accessed

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'config': self.memory_config,
            'num_submaps': self.map.get_num_submaps(),
            'recent_submaps': self._recent_submaps,
            'submaps_info': []
        }

        # Get info about each submap
        for submap in self.map.get_submaps():
            if isinstance(submap, MemoryEfficientSubmap):
                stats['submaps_info'].append(submap.get_memory_info())
            else:
                stats['submaps_info'].append({
                    'submap_id': submap.get_id(),
                    'type': 'regular',
                    'cached_data': {}
                })

        return stats

    def cleanup_memory(self):
        """Clean up all cached data and temporary files."""
        self.memory_manager.cleanup_all_cache()
        self.memory_manager.cleanup_gpu_memory()
        print("Memory cleanup completed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_memory()
        except:
            pass  # Ignore errors during cleanup