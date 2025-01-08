from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch
import os

torch_include = os.path.join(torch.__path__[0], 'include')
torch_lib = os.path.join(torch.__path__[0], 'lib')

setup(
    name='zigtorch',
    ext_modules=[
        CppExtension(
            name='zigtorch',
            sources=['zig_ops.cpp'],  # Only source files here
            include_dirs=[
                torch_include,
                os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')
            ],
            library_dirs=[torch_lib],
            libraries=['torch', 'torch_cpu', 'c10'],
            extra_compile_args=['-fPIC'],
            extra_link_args=['-Wl,-rpath,' + torch_lib],
            extra_objects=['mm.o']  # Include your object file here
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

