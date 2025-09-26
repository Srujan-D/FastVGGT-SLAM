"""
Memory management utilities for VGGT-SLAM.
Provides disk caching, GPU memory cleanup, and memory monitoring.
"""
import os
import gc
import psutil
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple
from .memory_config import MemoryConfig


class MemoryManager:
    """Manages memory optimization for VGGT-SLAM."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cached_files = []  # Track files for cleanup

        if config.enable_memory_monitoring:
            self.process = psutil.Process()
        else:
            self.process = None

    def log_memory_usage(self, context: str = ""):
        """Log current memory usage if monitoring is enabled."""
        if not self.config.log_memory_usage or not self.process:
            return

        mem_info = self.process.memory_info()
        cpu_mem_mb = mem_info.rss / 1024 / 1024

        gpu_mem_mb = 0
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024

        # print(f"[MEMORY] {context}: CPU={cpu_mem_mb:.1f}MB, GPU={gpu_mem_mb:.1f}MB")

    def cleanup_gpu_memory(self):
        """Aggressive GPU memory cleanup."""
        if self.config.enable_gpu_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            if self.config.log_memory_usage:
                self.log_memory_usage("After GPU cleanup")

    def save_array_to_cache(self, array: np.ndarray, filename: str) -> str:
        """Save numpy array to disk cache and return file path."""
        if not self.config.enable_disk_caching:
            return ""

        cache_path = self.config.get_cache_path(filename)
        np.save(cache_path, array)
        self.cached_files.append(cache_path + ".npy")  # np.save adds .npy extension

        if self.config.log_memory_usage:
            size_mb = array.nbytes / 1024 / 1024
            self.log_memory_usage(f"Cached {filename} ({size_mb:.1f}MB)")

        return cache_path + ".npy"

    def load_array_from_cache(self, cache_path: str) -> Optional[np.ndarray]:
        """Load numpy array from disk cache."""
        if not cache_path or not os.path.exists(cache_path):
            return None

        try:
            array = np.load(cache_path, allow_pickle=True)
            if self.config.log_memory_usage:
                size_mb = array.nbytes / 1024 / 1024
                self.log_memory_usage(f"Loaded cache ({size_mb:.1f}MB)")
            return array
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None

    def save_submap_data_to_cache(self, submap_id: int, data_dict: Dict[str, Any]) -> Dict[str, str]:
        """Save submap data arrays to cache and return cache paths."""
        cache_paths = {}

        if self.config.cache_pointclouds and "pointclouds" in data_dict:
            cache_paths["pointclouds"] = self.save_array_to_cache(
                data_dict["pointclouds"], f"submap_{submap_id}_pointclouds"
            )

        if self.config.cache_colors and "colors" in data_dict:
            cache_paths["colors"] = self.save_array_to_cache(
                data_dict["colors"], f"submap_{submap_id}_colors"
            )

        if self.config.cache_confidence and "conf" in data_dict:
            cache_paths["conf"] = self.save_array_to_cache(
                data_dict["conf"], f"submap_{submap_id}_conf"
            )

        if self.config.cache_frames and "frames" in data_dict:
            # Convert tensor to numpy if needed
            frames = data_dict["frames"]
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            cache_paths["frames"] = self.save_array_to_cache(
                frames, f"submap_{submap_id}_frames"
            )

        return cache_paths

    def load_submap_data_from_cache(self, cache_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load submap data from cache paths."""
        loaded_data = {}

        for key, cache_path in cache_paths.items():
            if cache_path:
                data = self.load_array_from_cache(cache_path)
                if data is not None:
                    loaded_data[key] = data

        return loaded_data

    def cleanup_all_cache(self):
        """Remove all cached files."""
        for cache_file in self.cached_files:
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            except Exception as e:
                print(f"Warning: Failed to remove cache file {cache_file}: {e}")

        self.cached_files.clear()

        if self.config.delete_cache_on_exit:
            self.config.cleanup_cache()


class CachedArray:
    """Wrapper for arrays that can be cached to disk."""

    def __init__(self, array: np.ndarray, memory_manager: MemoryManager, cache_key: str):
        self.memory_manager = memory_manager
        self.cache_key = cache_key
        self._array = array
        self._cache_path = ""
        self._is_cached = False

    def cache_to_disk(self):
        """Move array data to disk cache."""
        if not self._is_cached and self._array is not None:
            self._cache_path = self.memory_manager.save_array_to_cache(self._array, self.cache_key)
            if self._cache_path:
                self._array = None  # Free memory
                self._is_cached = True

    def load_from_disk(self) -> Optional[np.ndarray]:
        """Load array data from disk cache."""
        if self._is_cached and self._cache_path:
            return self.memory_manager.load_array_from_cache(self._cache_path)
        return self._array

    def get_array(self) -> Optional[np.ndarray]:
        """Get array data, loading from cache if needed."""
        if self._is_cached:
            return self.load_from_disk()
        return self._array

    @property
    def is_cached(self) -> bool:
        return self._is_cached

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get array shape without loading full array."""
        if self._array is not None:
            return self._array.shape
        elif self._is_cached and self._cache_path:
            # Try to get shape without loading full array
            try:
                # For .npy files, we can peek at the header
                with open(self._cache_path, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                    return shape
            except:
                pass
        return None


def create_memory_efficient_prediction_cache(predictions: Dict[str, Any],
                                           memory_manager: MemoryManager,
                                           submap_id: int) -> Dict[str, Any]:
    """Create a memory-efficient version of predictions with caching."""
    cached_predictions = {}

    # Cache large arrays
    cache_paths = memory_manager.save_submap_data_to_cache(submap_id, {
        "pointclouds": predictions.get("world_points"),
        "colors": predictions.get("images"),
        "conf": predictions.get("world_points_conf") or predictions.get("depth_conf"),
        "frames": predictions.get("images")
    })

    # Keep small data in memory, replace large arrays with cache info
    for key, value in predictions.items():
        if key in ["world_points", "world_points_conf", "depth_conf"] and memory_manager.config.cache_pointclouds:
            cached_predictions[key] = {"cache_path": cache_paths.get("pointclouds" if "world_points" in key else "conf")}
        elif key == "images" and memory_manager.config.cache_frames:
            cached_predictions[key] = {"cache_path": cache_paths.get("frames")}
        else:
            # Keep small arrays and metadata in memory
            cached_predictions[key] = value

    return cached_predictions