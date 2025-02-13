from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import platform

class ZigBuildExt(build_ext):
    def build_extension(self, ext):
        cmd = [
            "zig", "build-lib",
            "-dynamic",
            "-OReleaseFast",
            f"-femit-bin={self.get_ext_fullpath(ext.name)}",
            "src/native.zig"
        ]
        subprocess.check_call(cmd)

setup(
    name="zigtorch",
    version="0.1.0",
    packages=["zigtorch"],
    ext_modules=[Extension("libnative", [])],
    cmdclass={"build_ext": ZigBuildExt},
    install_requires=["numpy"],
)
