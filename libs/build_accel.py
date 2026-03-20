"""
build_accel.py - Consolidated builder for overhauled ASR Acceleration C Modules
@author: taylosh
Created: Dec 8 2025
Last edited: Mar 7 2026

Compiles all ASR acceleration C modules into the ./bin/ directory.
FIXED: Removed deprecated --clean option that causes build failures.
FIXED: Expanded configuration to include all 6 refactored source files as individual modules.
"""

import os
import sys
import platform
import subprocess
import shutil
import tempfile
from pathlib import Path

# Path resolution
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
BIN_DIR = PROJECT_ROOT / "bin"

SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
MODULE_EXT = '.pyd' if IS_WINDOWS else ('.dylib' if SYSTEM == "darwin" else '.so')

def get_compilation_flags():
    """Detect hardware and generate flags."""
    flags = ['/O2', '/D_CRT_SECURE_NO_WARNINGS'] if IS_WINDOWS else ['-O3', '-fPIC', '-std=c11', '-Wno-int-conversion']
    if shutil.which("nvcc"):
        flags.append("-DCUDA_AVAILABLE=1")
    # Detect OpenCL for AMD hardware
    if shutil.which("clinfo") or os.path.exists("C:/Windows/System32/OpenCL.dll") or os.path.exists("/usr/include/CL"):
        flags.append("-DOPENCL_AVAILABLE=1")
    return flags

def check_source_files(sources):
    """Check if all source files exist."""
    missing_files = []
    for source in sources:
        if not (SRC_DIR / source).exists():
            missing_files.append(source)
    return missing_files

def clean_build_artifacts(name):
    """Manually clean build artifacts before compilation."""
    for ext in ['*.so', '*.pyd', '*.dylib']:
        for f in BIN_DIR.glob(f"{name}{ext}"):
            f.unlink(missing_ok=True)
    
    build_dir = PROJECT_ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)

# UPDATED Consolidated Module Configuration (6 Modules for 6 Source Files)
MODULES_CONFIG = {
    # Refactored Segment Engine
    '_audio_segment_engine': {
        'sources': ['audio_segment_engine.c'], 
        'libraries': ['m'], 
        'needs_numpy': True, 
        'needs_gpu': True
    },
    # Refactored Signal Core
    '_audio_signal_core': {
        'sources': ['audio_signal_core.c'], 
        'libraries': ['m'], 
        'needs_numpy': True, 
        'needs_gpu': True
    },
    # Refactored Spectral Analysis
    '_audio_spectral_analysis': {
        'sources': ['audio_spectral_analysis.c'], 
        'libraries': ['m'], 
        'needs_numpy': True, 
        'needs_gpu': True
    },
    # Utility
    '_corpus_utils': {'sources': ['corpus_utils.c'], 'libraries': [], 'needs_numpy': False, 'needs_gpu': False},
    # GPU Backend
    '_gpu_backend': {'sources': ['gpu_backend.c'], 'libraries': [], 'needs_numpy': False, 'needs_gpu': True},
    # Textgrid
    '_textgrid': {'sources': ['textgrid.c'], 'libraries': ['m'], 'needs_numpy': False, 'needs_gpu': False}
}

def build_module(name, config, flags):
    print(f"Building: {name}...")
    
    # Check if source files exist
    missing_files = check_source_files(config['sources'])
    if missing_files:
        print(f" ⚠ Skipping {name} - missing source files: {', '.join(missing_files)}")
        print(f"   Expected in: {SRC_DIR}")
        return False
    
    # Clean before build
    clean_build_artifacts(name)
    
    # Ensure all paths are absolute strings
    sources = [os.path.abspath(str(SRC_DIR / s)) for s in config['sources']]
    include_dirs = [os.path.abspath(str(SRC_DIR))]
    
    if config['needs_numpy']:
        try:
            import numpy as np
            include_dirs.append(np.get_include())
        except ImportError:
            print(f" ⚠ Skipping {name} - numpy not available but required")
            return False

    # Create a minimal temporary setup script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("from setuptools import setup, Extension\n")
        f.write(f"ext = Extension('{name}', sources={sources}, include_dirs={include_dirs}, ")
        f.write(f"libraries={config['libraries']}, extra_compile_args={flags}, ")
        f.write("define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])\n")
        f.write(f"setup(name='{name}', ext_modules=[ext])\n")
        temp_file = f.name

    try:
        # FIXED: Removed --clean option
        cmd = [
            sys.executable, temp_file, 
            'build_ext', 
            '--build-lib', os.path.abspath(str(BIN_DIR)),
            '--build-temp', os.path.abspath(str(PROJECT_ROOT / "build" / "temp"))
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f" ✓ Successfully built {name}")
        return True
    except subprocess.CalledProcessError as e:
        # Extract meaningful error message
        error_output = e.stderr if e.stderr else e.stdout
        print(f" ✗ Build failed for {name}: {error_output[:500]}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def main():
    print("=== ASR ACCELERATION BUILDER ===")
    
    # Create necessary directories
    BIN_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / "build" / "temp").mkdir(parents=True, exist_ok=True)
    
    # Check if src directory exists
    if not SRC_DIR.exists():
        print(f"! Source directory not found: {SRC_DIR}")
        SRC_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  Place your .c files in {SRC_DIR}")
        return
    
    flags = get_compilation_flags()
    
    # Clean overall build directory
    build_dir = PROJECT_ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    
    successful_builds = 0
    total_modules = len(MODULES_CONFIG)
    
    for name, config in MODULES_CONFIG.items():
        if build_module(name, config, flags):
            successful_builds += 1
    
    built_count = len(list(BIN_DIR.glob(f'*{MODULE_EXT}')))
    print(f"\nBuild Summary:")
    print(f"  Modules attempted: {total_modules}")
    print(f"  Successfully built: {successful_builds}")
    print(f"  Compiled files in bin/: {built_count}")
    
    if successful_builds == 0:
        print("\n! No modules were built successfully.")
        print("  The pipeline will still work with CPU-only processing.")

if __name__ == "__main__":
    main()
