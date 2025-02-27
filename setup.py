from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import platform
import shutil
from pathlib import Path

class ZigBuildExt(build_ext):
    def build_extension(self, ext):
        # Determine output extension based on platform
        if platform.system() == "Windows":
            output_ext = ".dll"
        elif platform.system() == "Darwin":  # macOS
            output_ext = ".dylib"
        else:  # Linux and others
            output_ext = ".so"
        
        # Output file path
        output_file = self.get_ext_fullpath(ext.name) + output_ext
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Determine optimization level
        optimize_flag = "-OReleaseFast"
        
        # Build the Zig code
        build_cmd = [
            "zig", "build-lib",
            "-dynamic",
            optimize_flag,
            "-fPIC",
            "-lc",
            f"-femit-bin={output_file}",
            "mm.zig",
            "src/native.zig"
        ]
        
        self.announce(f"Building with command: {' '.join(build_cmd)}", level=2)
        subprocess.check_call(build_cmd)
        
        
        package_dir = Path("zigtorch")
        package_dir.mkdir(exist_ok=True)
        
        lib_name = "libnative" + output_ext
        lib_path = package_dir / lib_name
        
       
        shutil.copy2(output_file, lib_path)
        self.announce(f"Copied library to {lib_path}", level=2)

setup(
    name="zigtorch",
    version="0.1.0",
    description="Accelerated PyTorch made with zig",
    author="kitajusSus",
    packages=["zigtorch"],
    ext_modules=[Extension("native", sources=[])],
    cmdclass={"build_ext": ZigBuildExt},
    package_data={"zigtorch": ["*.so", "*.dll", "*.dylib"]},
    install_requires=["torch>=1.7.0", "numpy>=1.19.0"],
    python_requires=">=3.7",
)
