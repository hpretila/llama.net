#!/usr/bin/env python3

import os
import platform
import shutil

# Get the current working directory
cwd = os.getcwd()


def build_llama_bin():
    """
    Builds the llama.cpp binary
    """
    # Prepare the llama.cpp/build directory; remove if it exists! Use path builder instead of strings to keep it cross-platform
    print ("[BUILD] Clearing for build.")
    build_dir = os.path.join('llama.cpp', 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # Create the build directory
    os.mkdir(build_dir)

    # Build the llama.cpp binary
    print ("[BUILD] Invoking CMAKE.")
    os.system(f'cd {build_dir} && cmake -DLLAMA_OPENBLAS=ON -DBUILD_SHARED_LIBS=ON .. && cmake --build . && cd {cwd}')
