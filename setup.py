from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='zigtorch',
    ext_modules=[
        CppExtension(
            name='zigtorch',
            sources=['zig_ops.cpp'],     #  C++
            extra_objects=['mm.o'],       # tu plik .o z Zig
            extra_compile_args=['-std=c++17'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
