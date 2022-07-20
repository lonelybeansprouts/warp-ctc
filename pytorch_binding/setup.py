import os
import platform
import sys
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

extra_compile_args = ['-std=c++14', '-fPIC']

build_extension = CppExtension

ext_modules = [
    build_extension(
        name='warpctc_pytorch._bf_ctc',
        language='c++',
        sources=['src/binding.cpp'],
        extra_compile_args=extra_compile_args
    )
]

setup(
    name="bf_ctc",
    version="0.1",
    description="blankfree ctc",
    license="Apache",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
