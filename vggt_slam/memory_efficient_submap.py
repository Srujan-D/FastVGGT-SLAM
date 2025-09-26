"""
Memory-efficient submap implementation with disk caching support.
"""
import re
import os
import cv2
import torch
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Any
from .memory_utils import MemoryManager, CachedArray
from .memory_config import MemoryConfig


class MemoryEfficientSubmap:
    """Memory-efficient version of Submap with disk caching capabilities."""

    def __init__(self, submap_id: int, memory_manager: MemoryManager):
        self.submap_id = submap_id
        self.memory_manager = memory_manager

        # Core submap data - same as original
        self.H_world_map = None
        self.R_world_map = None
        self.poses = None
        self.vggt_intrinscs = None
        self.retrieval_vectors = None
        self.conf_threshold = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None

        # Cached large arrays
        self._cached_frames = None
        self._cached_pointclouds = None
        self._cached_colors = None
        self._cached_conf = None
        self._cached_conf_masks = None

        # Cache metadata
        self._cache_paths = {}
        self._shapes = {}  # Store shapes for efficiency

        # Voxelization cache - same as original submap
        self.voxelized_points = None

    def add_all_poses(self, poses):
        """Add poses - keep in memory as they're small."""
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics):
        """Add points with memory-efficient caching."""
        self.memory_manager.log_memory_usage(f"Before adding points to submap {self.submap_id}")

        # Calculate confidence threshold
        self.conf_threshold = np.percentile(conf, conf_threshold_percentile)
        self.vggt_intrinscs = intrinsics

        # Store shapes for later reference
        self._shapes['pointclouds'] = points.shape
        self._shapes['colors'] = colors.shape
        self._shapes['conf'] = conf.shape

        # Cache large arrays if enabled
        if self.memory_manager.config.cache_pointclouds:
            self._cached_pointclouds = CachedArray(
                points, self.memory_manager, f"submap_{self.submap_id}_pointclouds"
            )
            self._cached_pointclouds.cache_to_disk()
        else:
            self._cached_pointclouds = CachedArray(
                points, self.memory_manager, f"submap_{self.submap_id}_pointclouds"
            )

        if self.memory_manager.config.cache_colors:
            self._cached_colors = CachedArray(
                colors, self.memory_manager, f"submap_{self.submap_id}_colors"
            )
            self._cached_colors.cache_to_disk()
        else:
            self._cached_colors = CachedArray(
                colors, self.memory_manager, f"submap_{self.submap_id}_colors"
            )

        if self.memory_manager.config.cache_confidence:
            self._cached_conf = CachedArray(
                conf, self.memory_manager, f"submap_{self.submap_id}_conf"
            )
            self._cached_conf.cache_to_disk()
        else:
            self._cached_conf = CachedArray(
                conf, self.memory_manager, f"submap_{self.submap_id}_conf"
            )

        self.memory_manager.log_memory_usage(f"After adding points to submap {self.submap_id}")
        self.memory_manager.cleanup_gpu_memory()

    def add_all_frames(self, frames):
        """Add frames with memory-efficient caching."""
        self.memory_manager.log_memory_usage(f"Before adding frames to submap {self.submap_id}")

        # Store shape
        self._shapes['frames'] = frames.shape if hasattr(frames, 'shape') else frames.shape

        # Cache frames if enabled (they're usually large)
        if self.memory_manager.config.cache_frames:
            # Convert tensor to numpy if needed
            frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
            self._cached_frames = CachedArray(
                frames_np, self.memory_manager, f"submap_{self.submap_id}_frames"
            )
            self._cached_frames.cache_to_disk()
        else:
            # Keep frames in memory (for loop closure)
            frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
            self._cached_frames = CachedArray(
                frames_np, self.memory_manager, f"submap_{self.submap_id}_frames"
            )

        self.memory_manager.log_memory_usage(f"After adding frames to submap {self.submap_id}")

    def add_all_retrieval_vectors(self, retrieval_vectors):
        """Add retrieval vectors - keep in memory as they're small."""
        self.retrieval_vectors = retrieval_vectors

    def set_conf_masks(self, conf_masks):
        """Set confidence masks with caching."""
        if self.memory_manager.config.cache_confidence:
            self._cached_conf_masks = CachedArray(
                conf_masks, self.memory_manager, f"submap_{self.submap_id}_conf_masks"
            )
            self._cached_conf_masks.cache_to_disk()
        else:
            self._cached_conf_masks = CachedArray(
                conf_masks, self.memory_manager, f"submap_{self.submap_id}_conf_masks"
            )

    # ============ Getter methods with lazy loading ============

    def get_pointclouds(self, load_if_needed: bool = True) -> Optional[np.ndarray]:
        """Get pointclouds, loading from cache if needed."""
        if self._cached_pointclouds and load_if_needed:
            return self._cached_pointclouds.get_array()
        return None

    def get_colors(self, load_if_needed: bool = True) -> Optional[np.ndarray]:
        """Get colors, loading from cache if needed."""
        if self._cached_colors and load_if_needed:
            return self._cached_colors.get_array()
        return None

    def get_conf(self, load_if_needed: bool = True) -> Optional[np.ndarray]:
        """Get confidence values, loading from cache if needed."""
        if self._cached_conf and load_if_needed:
            return self._cached_conf.get_array()
        return None

    def get_all_frames(self, load_if_needed: bool = True) -> Optional[np.ndarray]:
        """Get all frames, loading from cache if needed."""
        if self._cached_frames and load_if_needed:
            frames = self._cached_frames.get_array()
            # Convert back to tensor if needed
            if frames is not None and isinstance(frames, np.ndarray):
                return torch.from_numpy(frames)
            return frames
        return None

    def get_conf_masks(self, load_if_needed: bool = True) -> Optional[np.ndarray]:
        """Get confidence masks, loading from cache if needed."""
        if self._cached_conf_masks and load_if_needed:
            return self._cached_conf_masks.get_array()
        return None

    # ============ Original API compatibility methods ============

    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold

    def get_frame_at_index(self, index):
        frames = self.get_all_frames()
        if frames is not None:
            return frames[index, ...]
        return None

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_all_poses_world(self, ignore_loop_closure_frames=False):
        """Get all poses in world frame - same logic as original."""
        projection_mat_list = self.vggt_intrinscs @ np.linalg.inv(self.poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
        poses = []
        for index, projection_mat in enumerate(projection_mat_list):
            cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
            trans = trans/trans[3,0]
            pose = np.eye(4)
            pose[0:3, 0:3] = np.linalg.inv(rot)
            pose[0:3,3] = trans[0:3,0]
            poses.append(pose)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break
        return np.stack(poses, axis=0)

    def get_frame_pointcloud(self, pose_index):
        pointclouds = self.get_pointclouds()
        if pointclouds is not None:
            return pointclouds[pose_index]
        return None

    def set_frame_ids(self, file_paths):
        """Extract frame IDs from file paths - same as original."""
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map

    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def get_reference_homography(self):
        return self.H_world_map

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    def get_frame_ids(self):
        return self.frame_ids

    def get_frame_ids_at_indices(self, indices):
        if isinstance(indices, (np.int64, int)):
            return [self.frame_ids[indices]]
        return [self.frame_ids[i] for i in indices]

    def filter_data_by_confidence(self, data, stride=1):
        """Filter data by confidence - loads conf if needed."""
        conf = self.get_conf()
        if conf is None:
            return data

        if stride == 1:
            init_conf_mask = conf >= self.conf_threshold
            return data[init_conf_mask]
        else:
            conf_sub = conf[:, ::stride, ::stride]
            data_sub = data[:, ::stride, ::stride, :]
            init_conf_mask = conf_sub >= self.conf_threshold
            return data_sub[init_conf_mask]

    def get_points_list_in_world_frame(self, ignore_loop_closure_frames=False):
        """Get points in world frame - loads pointclouds if needed."""
        pointclouds = self.get_pointclouds()
        conf_masks = self.get_conf_masks()

        if pointclouds is None or conf_masks is None:
            return [], [], []

        point_list = []
        frame_id_list = []
        frame_conf_mask = []

        for index, points in enumerate(pointclouds):
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (self.H_world_map @ points_homogeneous.T).T
            point_list.append((points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape))
            frame_id_list.append(self.frame_ids[index])
            conf_mask = conf_masks[index] >= self.conf_threshold
            frame_conf_mask.append(conf_mask)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break

        return point_list, frame_id_list, frame_conf_mask

    def get_points_in_world_frame(self, stride=1):
        """Get filtered points in world frame."""
        pointclouds = self.get_pointclouds()
        if pointclouds is None:
            return None

        points = self.filter_data_by_confidence(pointclouds, stride)
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        return points_transformed[:, :3] / points_transformed[:, 3:]

    def get_points_colors(self, stride=1):
        """Get filtered point colors - required for visualization."""
        colors = self.get_colors()
        if colors is None:
            return None

        filtered_colors = self.filter_data_by_confidence(colors, stride)
        return filtered_colors.reshape(-1, 3)

    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        """Get voxelized points in world frame - required for visualization."""
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                pointclouds = self.get_pointclouds()
                colors = self.get_colors()

                if pointclouds is None or colors is None:
                    return None

                points = self.filter_data_by_confidence(pointclouds)
                points_flat = points.reshape(-1, 3)
                filtered_colors = self.filter_data_by_confidence(colors)
                colors_flat = filtered_colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if nb_points > 0:
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(
                        nb_points=nb_points,
                        radius=voxel_size * factor_for_outlier_rejection
                    )
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(
            points_transformed[:, :3] / points_transformed[:, 3:]
        )
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame

    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about what's cached vs in memory."""
        info = {
            'submap_id': self.submap_id,
            'cached_data': {},
            'shapes': self._shapes
        }

        if self._cached_pointclouds:
            info['cached_data']['pointclouds'] = self._cached_pointclouds.is_cached
        if self._cached_colors:
            info['cached_data']['colors'] = self._cached_colors.is_cached
        if self._cached_conf:
            info['cached_data']['conf'] = self._cached_conf.is_cached
        if self._cached_frames:
            info['cached_data']['frames'] = self._cached_frames.is_cached
        if self._cached_conf_masks:
            info['cached_data']['conf_masks'] = self._cached_conf_masks.is_cached

        return info
    