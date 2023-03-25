#!/usr/bin/env python3

import os
import platform
import shutil

# Get the current working directory
cwd = os.getcwd()


def llama_dll():
    # Temporary placeholder
    filename = "llama"
    os_name = "generic"
    arch = "generic"
    ext = ""
    build_artifact = ""

    # Get OS strings
    if platform.system() == "Windows":
        os_name = "win"
        ext = ".dll"
        build_artifact = os.path.join("bin", "Debug", "llama.dll")
    elif platform.system() == "Linux":
        os_name = "linux"
        ext = ".so"
        build_artifact = f"lib{filename}{ext}"
    elif platform.system() == "Darwin":
        os_name = "osx"
        ext = ".dylib"
        build_artifact = f"lib{filename}{ext}"

    # Get architecture string
    if platform.machine() == "x86_64" or platform.machine() == "AMD64":
        arch = "x64"
    elif platform.machine() == "x86" or platform.machine() == "i386" or platform.machine() == "i686":
        arch = "x86"
    elif platform.machine().startswith("aarch64"):
        arch = "arm64"
    elif platform.machine().startswith("arm64"):
        arch = "arm64"
    elif platform.machine().startswith("arm"):
        arch = "arm"

    if os_name == "generic" or arch == "generic":
        raise OSError("The current platform is not supported.")

    # if runtimes doesnt exist, make the folder
    runtimes_dir = os.path.join(cwd, "LLaMA.NET", "runtimes")
    if not os.path.exists(runtimes_dir):
        os.mkdir(runtimes_dir)

    # if the os folder doesnt exist, make the folder
    os_dir = os.path.join(runtimes_dir, f"{os_name}-{arch}")
    if not os.path.exists(os_dir):
        os.mkdir(os_dir)

    # Destination path
    dest = os.path.join(os_dir, f"{filename}{ext}")

    # Return the paths
    return os.path.join(cwd, "llama.cpp","build", build_artifact), \
        dest

def build_llama_bin():
    """
    Builds the llama.cpp binary
    """
    # Prepare the llama.cpp/build directory; remove if it exists! Use path builder instead of strings to keep it cross-platform
    build_dir = os.path.join('llama.cpp', 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # Create the build directory
    os.mkdir(build_dir)

    # Build the llama.cpp binary
    os.system(f'cd {build_dir} && cmake -DBUILD_SHARED_LIBS=ON .. && cmake --build . && cd {cwd}')

def prepare_llama_bin():
    """
    Copies the llama.cpp binary to the LLaMA.NET/lib/llama directory
    """

    # Find out what OS we've got
    build_dir, dest_dir = llama_dll()

    shutil.copy(build_dir, dest_dir)


def build_llama_net():
    """
    Builds and runs the LLaMA.NET.Testing application

    """
    # Run the LLaMA.NET application
    os.system('dotnet build LLaMA.NET/LLaMA.NET.csproj')


def run_test():
    """
    Builds and runs the LLaMA.NET.Testing application
    """

    # Run the LLaMA.NET application
    os.system('dotnet run --project LLaMA.NET.Testing/LLaMA.NET.Testing.csproj')