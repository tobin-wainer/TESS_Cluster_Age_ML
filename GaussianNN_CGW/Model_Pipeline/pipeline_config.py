"""
Pipeline Configuration - Multi-Machine Portability

Auto-detects machine environment and configures paths, device, and DataLoader settings.
Supports:
- Mac laptop (Darwin/macOS): CPU, paths for Mac
- Windows desktop via WSL (Linux): CUDA GPU, paths for WSL

Usage:
    from pipeline_config import config, project_dir_path, temp_files_path, device
"""

import platform
import os
import getpass
from pathlib import Path

import torch


# ==============================================================================
# Machine Detection
# ==============================================================================

def detect_machine():
    """
    Detect which machine we're running on based on platform and username.

    Returns:
        str: Machine identifier ('mac_laptop', 'windows_wsl', or 'unknown')
    """
    system = platform.system()
    username = getpass.getuser()

    if system == 'Darwin':
        return 'mac_laptop'
    elif system == 'Linux' and 'cgw35' in username:
        # WSL shows as Linux, but we detect via username
        return 'windows_wsl'
    else:
        return 'unknown'


# ==============================================================================
# Path Configuration
# ==============================================================================

# Machine-specific paths
MACHINE_PATHS = {
    'mac_laptop': '/Users/howard_willard/Desktop/TESS_Cluster_Age_ML/',
    'windows_wsl': '/home/cgw35/projects/TESS_Cluster_Age_ML/',
}


def get_project_paths(machine_id):
    """
    Get project paths for the detected machine.
    Falls back to auto-detection from this file's location if machine unknown.

    Args:
        machine_id: Machine identifier from detect_machine()

    Returns:
        tuple: (project_dir_path, pipeline_dir_path, temp_files_path)
    """
    if machine_id in MACHINE_PATHS:
        project_root = MACHINE_PATHS[machine_id]
    else:
        # Fallback: derive from this file's location
        # This file is at: <project_root>/GaussianNN_CGW/Model_Pipeline/pipeline_config.py
        this_file = Path(__file__).resolve()
        project_root = str(this_file.parent.parent.parent) + '/'

    pipeline_dir = project_root + 'GaussianNN_CGW/Model_Pipeline/'
    temp_files = pipeline_dir + 'TempFiles/'

    return project_root, pipeline_dir, temp_files


def validate_paths(project_root, pipeline_dir, temp_files):
    """
    Validate that paths exist and create TempFiles if missing.

    Args:
        project_root, pipeline_dir, temp_files: Path strings

    Returns:
        bool: True if all paths valid
    """
    valid = True

    if not Path(project_root).exists():
        print(f"  WARNING: Project root does not exist: {project_root}")
        valid = False

    if not Path(pipeline_dir).exists():
        print(f"  WARNING: Pipeline directory does not exist: {pipeline_dir}")
        valid = False

    # Create TempFiles if it doesn't exist
    temp_path = Path(temp_files)
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {temp_files}")

    # Also create ModelGridLogs subdirectory
    model_logs = temp_path / 'ModelGridLogs'
    if not model_logs.exists():
        model_logs.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {model_logs}")

    return valid


# ==============================================================================
# Device Configuration
# ==============================================================================

def get_device():
    """
    Auto-detect the best available compute device.
    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info(device):
    """
    Get descriptive info about the device.

    Args:
        device: torch.device

    Returns:
        tuple: (device_type_str, device_description)
    """
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 'cuda', f'{name}'
    elif device.type == 'mps':
        return 'mps', 'Apple Silicon GPU'
    else:
        return 'cpu', 'CPU'


# ==============================================================================
# DataLoader Configuration
# ==============================================================================

def get_dataloader_settings(device):
    """
    Get optimal DataLoader settings for the device.

    Args:
        device: torch.device

    Returns:
        dict: {num_workers, pin_memory}
    """
    if device.type == 'cuda':
        # GPU: use pin_memory for faster transfer,
        # num_workers=0 due to WSL multiprocessing issues
        return {
            'num_workers': 0,  # WSL has issues with multiprocessing workers
            'pin_memory': True
        }
    else:
        # CPU/MPS: no pin_memory needed
        return {
            'num_workers': 0,
            'pin_memory': False
        }


# ==============================================================================
# Configuration Class
# ==============================================================================

class PipelineConfig:
    """
    Central configuration object containing all environment settings.
    """

    def __init__(self):
        # Detect machine
        self.machine_id = detect_machine()
        self.platform = platform.system()

        # Get paths
        self.project_root, self.pipeline_dir, self.temp_files = get_project_paths(self.machine_id)

        # Validate paths
        self._paths_valid = validate_paths(self.project_root, self.pipeline_dir, self.temp_files)

        # Get device
        self.device = get_device()
        self.device_type, self.device_description = get_device_info(self.device)

        # Get DataLoader settings
        loader_settings = get_dataloader_settings(self.device)
        self.num_workers = loader_settings['num_workers']
        self.pin_memory = loader_settings['pin_memory']

        # Print configuration summary
        self._print_summary()

    def _print_summary(self):
        """Print configuration summary on import."""
        print("\nPIPELINE CONFIGURATION")
        print(f"  Machine:     {self.machine_id} ({self.platform})")
        print(f"  Device:      {self.device_type} ({self.device_description})")

        if self.device.type == 'cuda':
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Memory:  {memory_gb:.1f} GB")

        print(f"  num_workers: {self.num_workers}")
        print(f"  pin_memory:  {self.pin_memory}")
        print()


# ==============================================================================
# Module-Level Exports (created on import)
# ==============================================================================

# Create config instance
config = PipelineConfig()

# Export commonly used values at module level for easy imports
project_dir_path = config.project_root
pipeline_dir_path = config.pipeline_dir
temp_files_path = config.temp_files
device = config.device


# ==============================================================================
# Convenience function for scripts
# ==============================================================================

def get_config():
    """Return the global configuration object."""
    return config


if __name__ == '__main__':
    # When run directly, show detailed config
    print("=" * 60)
    print("Pipeline Configuration Details")
    print("=" * 60)
    print(f"Machine ID:       {config.machine_id}")
    print(f"Platform:         {config.platform}")
    print(f"Project Root:     {config.project_root}")
    print(f"Pipeline Dir:     {config.pipeline_dir}")
    print(f"TempFiles Dir:    {config.temp_files}")
    print(f"Device:           {config.device}")
    print(f"Device Type:      {config.device_type}")
    print(f"Device Desc:      {config.device_description}")
    print(f"num_workers:      {config.num_workers}")
    print(f"pin_memory:       {config.pin_memory}")
    print("=" * 60)
