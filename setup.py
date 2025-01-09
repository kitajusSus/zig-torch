from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch
import os
import subprocess

# Paths to include and library directories for PyTorch
torch_include = torch.utils.cpp_extension.include_paths()[0]
torch_lib = torch.utils.cpp_extension.library_paths()[0]
"""
# Compile Zig code to object file before building the extension
def compile_zig():
    zig_cmd = [
        'zig',
        'cc',
        'zig_mm.zig',
        '-c',
        '-O3',
        '-march=native',
        '-fPIC',
        '-o',
        'mm.o'
    ]
    subprocess.check_call(zig_cmd)

# Compile Zig code
compile_zig()
"""
setup(
    name='zigtorch',
    ext_modules=[
        CppExtension(
            name='zigtorch',
            sources=['zig_ops.cpp'],  # C++ wrapper
            include_dirs=[
                torch_include,
                os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')
            ],
            library_dirs=[torch_lib],
            libraries=['torch', 'torch_cpu', 'c10'],
            extra_compile_args=[
                '-O3',                # High optimization level
                '-march=native',     # Optimize for local machine architecture
                '-fPIC', 
                '-fno-omit-frame-pointer'
            ],
            extra_link_args=[
                '-Wl,-rpath,' + torch_lib,
                '-lpthread'
            ],
            extra_objects=['mm.o']  # Link the Zig object file
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


