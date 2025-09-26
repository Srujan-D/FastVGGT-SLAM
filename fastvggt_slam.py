#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

# Import FastVGGT components
from vggt.models.vggt import VGGT as FastVGGT

# Import VGGT-SLAM components
import vggt_slam.slam_utils as utils
from vggt_slam.fastvggt_solver import Solver
from vggt_slam.memory_config import MemoryConfig, get_aggressive_memory_config, get_default_memory_config


class FastVGGTSLAM:
    """
    Unified FastVGGT-SLAM system that combines FastVGGT's optimized model
    with VGGT-SLAM's incremental SLAM capabilities.

    Supports two modes:
    1. SLAM mode: Incremental processing with submaps and loop closure
    2. Reconstruction mode: Batch processing for offline reconstruction
    """

    def __init__(
        self,
        # Model parameters
        model_source: str = "url",  # "huggingface", "checkpoint", "url"
        model_path: str = None,
        merging: int = 0,
        vis_attn_map: bool = False,

        # SLAM parameters
        conf_threshold: float = 25.0,
        use_point_map: bool = False,
        use_sim3: bool = True,
        submap_size: int = 32,
        overlapping_window_size: int = 1,
        max_loops: int = 1,
        min_disparity: float = 50.0,

        # Memory management
        use_memory_efficient: bool = False,
        memory_config: Optional[MemoryConfig] = None,

        # Processing parameters
        downsample_factor: int = 1,
        device: str = "auto",
    ):
        """
        Initialize FastVGGT-SLAM system.

        Args:
            model_source: "huggingface", "checkpoint", or "url"
            model_path: Path to checkpoint file or HuggingFace model name
            merging: FastVGGT merging parameter
            vis_attn_map: Enable attention map visualization
            conf_threshold: Confidence threshold for point filtering
            use_point_map: Use point map instead of depth-based points
            use_sim3: Use Sim3 instead of SL(4)
            submap_size: Number of frames per submap
            overlapping_window_size: Number of overlapping frames
            max_loops: Maximum loop closures per submap
            min_disparity: Minimum disparity for keyframe selection
            use_memory_efficient: Use memory-efficient solver
            memory_config: Memory configuration (if None, uses default)
            downsample_factor: Image downsampling factor
            device: Computing device ("auto", "cuda", "cpu")
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store configuration
        self.model_source = model_source
        self.model_path = model_path
        self.merging = merging
        self.vis_attn_map = vis_attn_map
        self.conf_threshold = conf_threshold
        self.use_point_map = use_point_map
        self.use_sim3 = use_sim3
        self.submap_size = submap_size
        self.overlapping_window_size = overlapping_window_size
        self.max_loops = max_loops
        self.min_disparity = min_disparity
        self.downsample_factor = downsample_factor

        # Initialize model
        self.model = self._load_model()

        # Initialize solver for SLAM mode
        self.use_memory_efficient = use_memory_efficient
        if memory_config is None:
            memory_config = get_default_memory_config()
        self.memory_config = memory_config
        self.solver = None
        self.fastvggt_solver = None

    def _load_model(self) -> FastVGGT:
        """Load FastVGGT model from various sources."""
        print("Initializing FastVGGT model...")

        model = FastVGGT(
            merging=self.merging,
            vis_attn_map=self.vis_attn_map,
            enable_camera=True,
            enable_depth=True,
            enable_point=True,  # Enable both depth and point heads
            enable_track=False,
        )

        if self.model_source == "checkpoint":
            if not self.model_path or not os.path.exists(self.model_path):
                raise ValueError(f"Checkpoint path not found: {self.model_path}")
            print(f"Loading from checkpoint: {self.model_path}")
            ckpt = torch.load(self.model_path, map_location="cpu")
            incompatible = model.load_state_dict(ckpt, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"⚠️  Partially incompatible keys: {incompatible}")

        elif self.model_source == "huggingface":
            model_name = self.model_path or "facebook/VGGT-1B"
            print(f"Loading from HuggingFace: {model_name}")
            # Use the same loading method as original VGGT-SLAM
            try:
                model = FastVGGT.from_pretrained(model_name)
            except:
                print("HuggingFace loading failed, using URL method...")
                self.model_source = "url"

        if self.model_source == "url":
            url = self.model_path or "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            print(f"Loading from URL: {url}")
            state_dict = torch.hub.load_state_dict_from_url(url)

            # Filter out weights for disabled heads (point_head is disabled)
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('point_head.') and not key.startswith('track_head.'):
                    filtered_state_dict[key] = value

            model.load_state_dict(filtered_state_dict, strict=False)

        model.eval()
        model = model.to(self.device)

        # Convert to bfloat16 for efficiency (like FastVGGT)
        if self.device == "cuda":
            model = model.to(torch.bfloat16)

        print("✅ FastVGGT model loaded successfully")
        return model

    def _initialize_solver(self):
        """Initialize FastVGGT SLAM solver."""
        if self.fastvggt_solver is not None:
            return

        # Always use FastVGGT solver for both memory-efficient and standard modes
        self.fastvggt_solver = Solver(
            init_conf_threshold=self.conf_threshold,
            use_point_map=self.use_point_map,
            use_sim3=self.use_sim3,
            gradio_mode=False
        )

        # Keep reference for backward compatibility
        self.solver = self.fastvggt_solver

    def run_slam(
        self,
        image_folder: str,
        vis_map: bool = False,
        vis_flow: bool = False,
        log_results: bool = False,
        log_path: str = "fastvggt_results/poses.txt",
        skip_dense_log: bool = False,
        use_optical_flow_downsample: bool = True,
    ) -> Dict:
        """
        Run FastVGGT-SLAM in incremental SLAM mode.

        Args:
            image_folder: Path to folder containing images
            vis_map: Visualize point cloud as it's being built
            vis_flow: Visualize optical flow for keyframe selection
            log_results: Save results to file
            log_path: Path to save log file
            skip_dense_log: Skip dense point cloud logging
            use_optical_flow_downsample: Use optical flow for keyframe selection

        Returns:
            Dictionary containing SLAM results and extracted frame data
        """
        self._initialize_solver()

        print(f"Loading images from {image_folder}...")
        image_names = [f for f in glob.glob(os.path.join(image_folder, "*"))
                      if "depth" not in os.path.basename(f).lower()
                      and "txt" not in os.path.basename(f).lower()
                      and "db" not in os.path.basename(f).lower()]

        image_names = utils.sort_images_by_number(image_names)
        image_names = utils.downsample_images(image_names, self.downsample_factor)
        print(f"Found {len(image_names)} images")

        image_names_subset = []
        focal_lengths = []

        for image_name in tqdm(image_names, desc="Processing images"):
            if use_optical_flow_downsample:
                img = cv2.imread(image_name)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                enough_disparity = self.fastvggt_solver.flow_tracker.compute_disparity(
                    img, self.min_disparity, vis_flow
                )
                if enough_disparity:
                    image_names_subset.append(image_name)
            else:
                image_names_subset.append(image_name)

            # Process submap when enough images collected
            if (len(image_names_subset) == self.submap_size + self.overlapping_window_size
                or image_name == image_names[-1]):
                if len(image_names_subset) < 3:
                    continue
                print(f"Processing submap with {len(image_names_subset)} images")

                # Run FastVGGT predictions using Solver
                predictions = self.fastvggt_solver.run_predictions(
                    image_names_subset, self.model, self.max_loops
                )
                focal_lengths.append(predictions["intrinsic"][:,0,0])
                self.fastvggt_solver.add_points(predictions)

                print("=== BEFORE OPTIMIZATION ===")
                print(f"Number of submaps: {self.fastvggt_solver.map.get_num_submaps()}")
                for submap in self.fastvggt_solver.map.get_submaps():
                    print(f"Submap {submap.get_id()} reference translation: "
                          f"{submap.get_reference_homography()[:3, 3]}")

                # Graph optimization
                self.fastvggt_solver.graph.optimize()
                self.fastvggt_solver.map.update_submap_homographies(self.fastvggt_solver.graph)

                print("=== AFTER OPTIMIZATION ===")
                for submap in self.fastvggt_solver.map.get_submaps():
                    print(f"Submap {submap.get_id()} reference translation: "
                          f"{submap.get_reference_homography()[:3, 3]}")

                # Visualization
                loop_closure_detected = len(predictions["detected_loops"]) > 0
                if vis_map:
                    if loop_closure_detected:
                        self.fastvggt_solver.update_all_submap_vis()
                    else:
                        self.fastvggt_solver.update_latest_submap_vis()

                # Reset for next submap
                image_names_subset = image_names_subset[-self.overlapping_window_size:]

        print(f"Total submaps: {self.fastvggt_solver.map.get_num_submaps()}")
        print(f"Total loop closures: {self.fastvggt_solver.graph.get_num_loops()}")

        # Extract frame data
        frame_data = self._extract_vggt_frame_data(self.fastvggt_solver)

        # Show final map if not shown during processing
        if not vis_map:
            self.fastvggt_solver.update_all_submap_vis()

        # Save results
        if log_results:
            self._save_slam_results(log_path, skip_dense_log)

        # Cleanup memory if using memory-efficient solver
        if (self.use_memory_efficient
            and hasattr(self.fastvggt_solver, 'cleanup_memory')):
            print("Cleaning up cached data...")
            self.fastvggt_solver.cleanup_memory()

        return {
            'frame_data': frame_data,
            'solver': self.fastvggt_solver,
            'focal_lengths': focal_lengths,
            'num_submaps': self.fastvggt_solver.map.get_num_submaps(),
            'num_loops': self.fastvggt_solver.graph.get_num_loops(),
        }

    def run_reconstruction(
        self,
        data_path: Union[str, Path],
        output_path: Union[str, Path] = "./fastvggt_reconstruction",
        max_frames: int = 200,
        depth_conf_thresh: float = 3.0,
        enable_evaluation: bool = False,
        chamfer_max_dist: float = 0.5,
        plot: bool = False,
    ) -> Dict:
        """
        Run FastVGGT in batch reconstruction mode (like FastVGGT eval).

        Args:
            data_path: Path to dataset (should contain images/ subdirectory)
            output_path: Output directory for results
            max_frames: Maximum number of frames to process
            depth_conf_thresh: Depth confidence threshold
            enable_evaluation: Enable evaluation (requires pose and gt_ply)
            chamfer_max_dist: Maximum distance for Chamfer distance
            plot: Generate plots

        Returns:
            Dictionary containing reconstruction results and metrics
        """
        data_path = Path(data_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check required directories
        images_dir = data_path / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        print(f"📁 Dataset path: {data_path}")
        print(f"🏃 Reconstruction mode, evaluation: {'enabled' if enable_evaluation else 'disabled'}")

        # Load images
        image_paths = get_sorted_image_paths(images_dir)
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

        print(f"🖼️  Found {len(image_paths)} images")

        # Handle evaluation data if enabled
        poses_gt = None
        first_gt_pose = None
        c2ws = None

        if enable_evaluation:
            from vggt.utils.eval_utils import load_poses
            pose_dir = data_path / "pose"
            if not pose_dir.exists():
                raise ValueError(f"Pose directory required for evaluation: {pose_dir}")

            poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_dir)
            if poses_gt is None:
                raise ValueError("Failed to load pose data")
            print(f"📐 Loaded {len(poses_gt)} poses")

            # Frame selection based on poses
            selected_frame_ids, selected_image_paths, selected_pose_indices = (
                build_frame_selection(image_paths, available_pose_frame_ids, max_frames)
            )
            c2ws = poses_gt[selected_pose_indices]
            image_paths = selected_image_paths
        else:
            # Simple frame selection
            num_frames = min(len(image_paths), max_frames)
            selected_frame_ids = list(range(num_frames))
            image_paths = image_paths[:num_frames]

        print(f"📋 Selected {len(image_paths)} frames for processing")

        # Load and process images
        print("🔄 Loading images...")
        images = load_images_rgb(image_paths)
        if not images or len(images) < 3:
            raise ValueError("Not enough valid images (need at least 3)")

        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

        # Update model patch dimensions
        self.model.update_patch_dimensions(patch_width, patch_height)

        # Run inference and reconstruction
        print("🚀 Starting inference and reconstruction...")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        (
            extrinsic_np,
            intrinsic_np,
            all_world_points,
            all_point_colors,
            all_cam_to_world_mat,
            inference_time_ms,
        ) = infer_vggt_and_reconstruct(
            self.model, vgg_input, dtype, depth_conf_thresh, image_paths
        )

        print(f"⏱️  Inference time: {inference_time_ms:.2f}ms")

        if not all_cam_to_world_mat or not all_world_points:
            raise ValueError("Failed to obtain valid camera poses or point clouds")

        # Save results and run evaluation if enabled
        results = {
            'cam_to_world_matrices': all_cam_to_world_mat,
            'world_points': all_world_points,
            'point_colors': all_point_colors,
            'intrinsics': intrinsic_np,
            'extrinsics': extrinsic_np,
            'inference_time_ms': inference_time_ms,
            'frame_ids': selected_frame_ids if enable_evaluation else list(range(len(image_paths))),
        }

        if enable_evaluation:
            from vggt.utils.eval_utils import evaluate_scene_and_save

            print("📊 Starting evaluation...")
            gt_ply_dir = data_path / "gt_ply"
            if not gt_ply_dir.exists():
                raise ValueError(f"GT PLY directory required for evaluation: {gt_ply_dir}")

            metrics = evaluate_scene_and_save(
                "fastvggt_reconstruction",
                c2ws,
                first_gt_pose,
                selected_frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_path,
                gt_ply_dir,
                chamfer_max_dist,
                inference_time_ms,
                plot,
            )

            if metrics:
                print("📈 Evaluation results:")
                for key, value in metrics.items():
                    if key in ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"]:
                        print(f"  {key}: {float(value):.4f}")
                results['metrics'] = metrics
        else:
            # Save reconstruction results
            self._save_reconstruction_results(
                output_path, all_cam_to_world_mat, all_world_points,
                all_point_colors, selected_frame_ids if enable_evaluation else None
            )

        if plot and not enable_evaluation:
            self._visualize_trajectory(all_cam_to_world_mat, output_path)

        return results

    def _extract_vggt_frame_data(self, solver) -> List[Dict]:
        """Extract frame data from SLAM results (same as original main.py)."""
        frame_data = []

        for submap in solver.map.get_submaps():
            submap_id = submap.get_id()
            local_poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
            frame_ids = submap.get_frame_ids()
            images = submap.get_all_frames()

            point_clouds, _, conf_masks = submap.get_points_list_in_world_frame(
                ignore_loop_closure_frames=True
            )

            intrinsics = submap.vggt_intrinscs

            for i, (local_pose, frame_id, image, points, conf_mask) in enumerate(
                zip(local_poses, frame_ids, images, point_clouds, conf_masks)
            ):
                points_flat = points.reshape(-1, 3)
                conf_flat = conf_mask.reshape(-1)
                confident_points = points_flat[conf_flat]

                frame_data.append({
                    'frame_id': frame_id,
                    'submap_id': submap_id,
                    'local_pose': local_pose,
                    'image': image.contiguous().cpu().numpy(),
                    'points_3d': confident_points,
                    'intrinsics': intrinsics,
                    'conf_mask': conf_mask,
                })

        return frame_data

    def _save_slam_results(self, log_path: str, skip_dense_log: bool = False):
        """Save SLAM results to files."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Save poses
        self.solver.map.write_poses_to_file(log_path)

        # Save point cloud
        self.solver.map.write_points_to_file(log_path.replace(".txt", "_points.pcd"))

        if not skip_dense_log:
            # Save dense point clouds
            self.solver.map.save_framewise_pointclouds(log_path.replace(".txt", "_logs"))

        print(f"💾 SLAM results saved to: {log_path}")

    def _save_reconstruction_results(
        self,
        output_dir: Path,
        cam_to_world_matrices: List,
        world_points: List,
        point_colors: List,
        frame_ids: Optional[List] = None,
    ):
        """Save reconstruction results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save poses
        poses_path = output_dir / "estimated_poses.txt"
        with open(poses_path, "w") as f:
            for i, pose in enumerate(cam_to_world_matrices):
                frame_id = frame_ids[i] if frame_ids else i
                f.write(f"# Frame {frame_id}\n")
                for row in pose:
                    f.write(" ".join(map(str, row)) + "\n")
                f.write("\n")

        # Save point cloud
        if world_points:
            self._save_point_cloud(
                output_dir / "reconstructed_points.ply",
                world_points,
                point_colors
            )

        print(f"💾 Reconstruction results saved to: {output_dir}")

    def _save_point_cloud(self, output_path: Path, world_points: List, point_colors: List):
        """Save point cloud as PLY file."""
        try:
            # Merge all point clouds
            merged_points = np.vstack(world_points)
            merged_colors = (
                np.vstack(point_colors).astype(np.uint8)
                if point_colors and len(point_colors) > 0 else None
            )

            # Subsample if too many points
            max_points = 100000
            if len(merged_points) > max_points:
                print(f"🔽 Subsampling {len(merged_points)} -> {max_points} points")
                indices = np.random.choice(len(merged_points), size=max_points, replace=False)
                merged_points = merged_points[indices]
                if merged_colors is not None:
                    merged_colors = merged_colors[indices]

            # Write PLY file
            with open(output_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(merged_points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                if merged_colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                f.write("end_header\n")

                if merged_colors is None:
                    for point in merged_points:
                        if not (np.isnan(point).any() or np.isinf(point).any()):
                            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                else:
                    for point, color in zip(merged_points, merged_colors):
                        if not (np.isnan(point).any() or np.isinf(point).any()):
                            r, g, b = np.clip(color, 0, 255).astype(int)
                            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")

            print(f"💾 Point cloud saved: {output_path}")

        except Exception as e:
            print(f"⚠️  Error saving point cloud: {e}")

    def _visualize_trajectory(self, cam_to_world_matrices: List, output_dir: Path):
        """Visualize camera trajectory."""
        if len(cam_to_world_matrices) < 2:
            return

        poses = np.array(cam_to_world_matrices)
        positions = poses[:, :3, 3]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(positions[0, 0], positions[0, 2], color='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 2], color='red', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('Camera Trajectory (XZ projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        trajectory_path = output_dir / "trajectory.png"
        plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Trajectory saved: {trajectory_path}")


def main():
    """Main function with unified argument parsing for both modes."""
    parser = argparse.ArgumentParser(
        description="FastVGGT-SLAM: Unified SLAM and reconstruction with FastVGGT"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["slam", "reconstruction"],
        default="slam",
        help="Processing mode: slam (incremental) or reconstruction (batch)"
    )

    # Input/Output
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input data (image folder for SLAM, dataset folder for reconstruction)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./fastvggt_results/gr_hq_living_room",
        help="Output directory for results"
    )

    # Model parameters
    parser.add_argument(
        "--model_source",
        choices=["huggingface", "checkpoint", "url"],
        default="url",
        help="Model source"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint or HuggingFace model name"
    )
    parser.add_argument("--merging", type=int, default=0, help="FastVGGT merging parameter")
    parser.add_argument("--vis_attn_map", action="store_true", help="Visualize attention maps")

    # SLAM-specific parameters
    parser.add_argument("--submap_size", type=int, default=8, help="Frames per submap")
    parser.add_argument("--overlapping_window_size", type=int, default=1, help="Overlapping frames")
    parser.add_argument("--max_loops", type=int, default=1, help="Max loop closures per submap")
    parser.add_argument("--min_disparity", type=float, default=50.0, help="Min disparity for keyframes")
    parser.add_argument("--vis_map", action="store_true", help="Visualize map during processing")
    parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow")
    parser.add_argument("--log_results", action="store_true", help="Save SLAM results")
    parser.add_argument("--skip_dense_log", action="store_true", help="Skip dense point cloud logging")

    # Reconstruction-specific parameters
    parser.add_argument("--max_frames", type=int, default=200, help="Max frames for reconstruction")
    parser.add_argument("--enable_evaluation", action="store_true", help="Enable evaluation")
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5, help="Chamfer distance threshold")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    # Common parameters
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Confidence threshold")
    parser.add_argument("--depth_conf_thresh", type=float, default=3.0, help="Depth confidence threshold")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map")
    parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
    parser.add_argument("--downsample_factor", type=int, default=1, help="Image downsample factor")

    # Memory management
    parser.add_argument("--use_memory_efficient", action="store_true", help="Use memory-efficient solver")
    parser.add_argument("--cache_dir", type=str, default="./.fastvggt_cache", help="Cache directory")
    parser.add_argument("--disable_caching", action="store_true", help="Disable disk caching")
    parser.add_argument("--aggressive_memory", action="store_true", help="Aggressive memory settings")
    parser.add_argument("--keep_recent_submaps", type=int, default=2, help="Recent submaps in memory")
    parser.add_argument("--log_memory_usage", action="store_true", help="Log memory usage")

    args = parser.parse_args()

    # Setup memory configuration
    memory_config = None
    if args.use_memory_efficient:
        if args.aggressive_memory:
            memory_config = get_aggressive_memory_config()
        else:
            memory_config = get_default_memory_config()

        if args.cache_dir:
            memory_config.cache_directory = args.cache_dir
        if args.disable_caching:
            memory_config.enable_disk_caching = False
        if args.log_memory_usage:
            memory_config.enable_memory_monitoring = True
            memory_config.log_memory_usage = True
        memory_config.recent_submaps_count = args.keep_recent_submaps

    # Initialize FastVGGT-SLAM
    fast_vggt_slam = FastVGGTSLAM(
        model_source=args.model_source,
        model_path=args.model_path,
        merging=args.merging,
        vis_attn_map=args.vis_attn_map,
        conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        submap_size=args.submap_size,
        overlapping_window_size=args.overlapping_window_size,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        use_memory_efficient=args.use_memory_efficient,
        memory_config=memory_config,
        downsample_factor=args.downsample_factor,
    )

    try:
        if args.mode == "slam":
            print("🚀 Running FastVGGT-SLAM in incremental SLAM mode...")
            results = fast_vggt_slam.run_slam(
                image_folder=args.input_path,
                vis_map=args.vis_map,
                vis_flow=args.vis_flow,
                log_results=args.log_results,
                log_path=os.path.join(args.output_path, "poses.txt"),
                skip_dense_log=args.skip_dense_log,
            )

            # Save frame data
            os.makedirs(args.output_path, exist_ok=True)
            frame_data_path = os.path.join(args.output_path, "fastvggt_frame_data.pkl")
            with open(frame_data_path, "wb") as f:
                pickle.dump(results['frame_data'], f)
            print(f"💾 Frame data saved: {frame_data_path}")

            print(f"✅ SLAM completed: {results['num_submaps']} submaps, {results['num_loops']} loops")

        elif args.mode == "reconstruction":
            print("🚀 Running FastVGGT-SLAM in batch reconstruction mode...")
            results = fast_vggt_slam.run_reconstruction(
                data_path=args.input_path,
                output_path=args.output_path,
                max_frames=args.max_frames,
                depth_conf_thresh=args.depth_conf_thresh,
                enable_evaluation=args.enable_evaluation,
                chamfer_max_dist=args.chamfer_max_dist,
                plot=args.plot,
            )

            print(f"✅ Reconstruction completed: {len(results['cam_to_world_matrices'])} poses")
            if 'metrics' in results:
                print("📊 Evaluation metrics available in results")

        print("🎉 FastVGGT-SLAM finished successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())