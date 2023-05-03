#!/usr/bin/env python3

import os
import platform
import shutil

# Get the current working directory
cwd = os.getcwd()


def llama_dll():
    """
    Grabs all of the directories for everything to sit in.

    Returns:
        str: Source directory for the library binary.
        str: Destination directory for the library binary.
        str: Binary directory.
        str: Destination directory for this OS.
    """

    # Temporary placeholder
    filename = "llama"
    os_name = "generic"
    arch = "generic"
    lib_ext = ""
    build_artifact_lib = ""

    # Get OS strings
    if platform.system() == "Windows":
        os_name = "win"
        lib_ext = ".dll"
        build_artifact_lib = os.path.join("bin", "Debug", "llama.dll")
    elif platform.system() == "Linux":
        os_name = "linux"
        lib_ext = ".so"
        build_artifact_lib = f"lib{filename}{lib_ext}"
    elif platform.system() == "Darwin":
        os_name = "osx"
        lib_ext = ".dylib"
        build_artifact_lib = f"lib{filename}{lib_ext}"

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
    llama_source_dir = os.path.join(cwd, "llama.cpp","build", build_artifact_lib)
    llama_dest_dir = os.path.join(os_dir, f"{filename}{lib_ext}")
    llama_build_bin_dir = os.path.join(cwd, "llama.cpp","build", "bin")

    # Return the paths
    return llama_source_dir, llama_dest_dir, llama_build_bin_dir, os_dir

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

def prepare_llama_bin():
    """
    Copies the following to the LLaMA.NET/lib/llama directory:
    - libllama - the model library
    - convert-pth-to-ggml.py - the script to convert state dicts to ggml.
    - quantize - the quantize executable for quantizing models further from ggml format.
    - quantize.py - the script to invoke quantize.
    """

    # Find out what OS we've got
    lib_build_dir, lib_dest_dir, build_dir, os_dest_dir = llama_dll()

    # Copy our library
    print (f"[PREPARE] Copying libllama to {lib_dest_dir}.")
    shutil.copy(lib_build_dir, lib_dest_dir)

    # Copy our pth to ggml script
    convert_py_build_dir = os.path.join(cwd, "llama.cpp","convert-pth-to-ggml.py")
    convert_py_dest_dir = os.path.join(
        os_dest_dir,"convert-pth-to-ggml.py")
    print (f"[PREPARE] Copying convert-pth-to-ggml.py script to {convert_py_dest_dir}.")
    shutil.copy(convert_py_build_dir, convert_py_dest_dir)

    # Copy our quantize binary
    executable_ext = "" if not "win" in lib_dest_dir else ".exe"
    quantize_build_dir = os.path.join(
        build_dir,f"quantize{executable_ext}")
    quantize_dest_dir = os.path.join(
        os_dest_dir,f"quantize{executable_ext}")
    print (f"[PREPARE] Copying quantize to {quantize_dest_dir}.")
    shutil.copy(quantize_build_dir, quantize_dest_dir)

    # Copy our quantize script
    quantize_py_build_dir = os.path.join(cwd, "llama.cpp","quantize.py")
    quantize_py_dest_dir = os.path.join(
        os_dest_dir,f"quantize.py")
    print (f"[PREPARE] Copying quantize.py script to {quantize_dest_dir}.")
    shutil.copy(quantize_py_build_dir, quantize_py_dest_dir)


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
