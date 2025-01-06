
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='zigtorch',
    ext_modules=[
        cpp_extension.CppExtension(
            'zigtorch',
            sources=['zig_ops.cpp'],
            extra_objects=['mm.o'],
            extra_compile_args=['-O3'],
            extra_link_args=['-pthread']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)


