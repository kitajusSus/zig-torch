# python/setup.py
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import platform
import shutil
import sys
from pathlib import Path

class ZigBuild(build_ext):
    def run(self):
        # Build the Zig library first
        self.announce("Building ZigTorch Zig library...", level=3)
        
        # Determine optimization level
        optimize = "ReleaseFast" if not self.debug else "Debug"
        
        # Run Zig build
        cmd = ["zig", "build", f"-Doptimize={optimize}"]
        subprocess.check_call(cmd, cwd=str(Path(__file__).parent.parent))
        
        # Make sure output directory exists
        os.makedirs(os.path.join(self.build_lib, "zigtorch"), exist_ok=True)
        
        # Copy the shared library to the Python package
        if platform.system() == "Windows":
            lib_name = "zigtorch.dll"
        elif platform.system() == "Darwin":
            lib_name = "libzigtorch.dylib"
        else:
            lib_name = "libzigtorch.so"
            
        src_path = os.path.join(
            Path(__file__).parent.parent,
            "zig-out", "lib", lib_name
        )
        
        dst_path = os.path.join(
            self.build_lib, "zigtorch", lib_name
        )
        
        self.announce(f"Copying {src_path} to {dst_path}", level=3)
        shutil.copy2(src_path, dst_path)

setup(
    name="zigtorch",
    version="0.1.0",
    description="Fast tensor operations with Zig",
    author="kitajusSus",
    packages=find_packages(),
    cmdclass={
        "build_ext": ZigBuild,
    },
    setup_requires=["numpy>=1.19.0"],
    install_requires=["numpy>=1.19.0"],
    python_requires=">=3.7",
)