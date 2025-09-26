"""
Memory management configuration for VGGT-SLAM.
Contains settings to optimize GPU and CPU memory usage.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration class for memory optimization settings."""

    # Disk caching settings
    enable_disk_caching: bool = True
    cache_directory: str = "./.vggt_cache"
    delete_cache_on_exit: bool = True

    # GPU memory management
    enable_gpu_cleanup: bool = True
    cleanup_after_inference: bool = True

    # Submap data caching strategy
    cache_pointclouds: bool = True
    cache_colors: bool = True
    cache_confidence: bool = True
    cache_frames: bool = False  # Keep frames in memory for loop closure

    # Graph optimization memory handling
    keep_recent_submaps_in_memory: bool = True  # vs on-demand loading
    recent_submaps_count: int = 2  # Number of recent submaps to keep in memory

    # Memory monitoring (for debugging)
    enable_memory_monitoring: bool = False
    log_memory_usage: bool = False

    def __post_init__(self):
        """Validate configuration and create cache directory if needed."""
        if self.enable_disk_caching:
            os.makedirs(self.cache_directory, exist_ok=True)

    def get_cache_path(self, filename: str) -> str:
        """Get full path for a cache file."""
        return os.path.join(self.cache_directory, filename)

    def cleanup_cache(self):
        """Remove all cache files if delete_cache_on_exit is True."""
        if self.delete_cache_on_exit and self.enable_disk_caching:
            import shutil
            if os.path.exists(self.cache_directory):
                shutil.rmtree(self.cache_directory)
                print(f"Cleaned up cache directory: {self.cache_directory}")


def get_default_memory_config() -> MemoryConfig:
    """Get default memory configuration with conservative settings."""
    return MemoryConfig()


def get_aggressive_memory_config() -> MemoryConfig:
    """Get aggressive memory configuration for memory-constrained systems."""
    return MemoryConfig(
        enable_disk_caching=True,
        cache_pointclouds=True,
        cache_colors=True,
        cache_confidence=True,
        cache_frames=True,  # Cache frames too for maximum memory savings
        enable_gpu_cleanup=True,
        cleanup_after_inference=True,
        keep_recent_submaps_in_memory=False,  # Load everything on-demand
        recent_submaps_count=1,
        enable_memory_monitoring=True,
        log_memory_usage=True,
    )


def get_memory_config_from_args(args) -> MemoryConfig:
    """Create memory config from command line arguments."""
    config = MemoryConfig()

    # Override defaults with any command line arguments
    if hasattr(args, 'cache_dir') and args.cache_dir:
        config.cache_directory = args.cache_dir
    if hasattr(args, 'disable_caching') and args.disable_caching:
        config.enable_disk_caching = False
    if hasattr(args, 'aggressive_memory') and args.aggressive_memory:
        config = get_aggressive_memory_config()

    return config