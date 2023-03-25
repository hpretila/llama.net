using System;
using System.IO;
using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader
{
    public struct LibraryEnvironment {
        public string LibraryPath { get; set; }
        public string OperatingSystem { get; set; }
    }

    public static class NativeHelper{
        public const string LLAMA_DLL = "libllama";
        
        public static LibraryEnvironment GetNativeLibraryEnvironment {
            get {
                // Temporary placeholder
                var filename = "llama";
                var os = "generic";
                var arch = "generic";
                var ext = "";

                // Get OS strings
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    os = "win";
                    ext = ".dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    os = "linux";
                    ext = ".so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    os = "osx";
                    ext = ".dylib";
                }

                // Get architecture string
                if (RuntimeInformation.ProcessArchitecture == Architecture.X64)
                {
                    arch = "x64";
                }
                else if (RuntimeInformation.ProcessArchitecture == Architecture.X86)
                {
                    arch = "x86";
                }
                else if (RuntimeInformation.ProcessArchitecture == Architecture.Arm)
                {
                    arch = "arm";
                }
                else if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64)
                {
                    arch = "arm64";
                }

                if (os == "generic" || arch == "generic")
                {
                    throw new PlatformNotSupportedException("The current platform is not supported.");
                }

                // Return the path
                return new LibraryEnvironment{
                    LibraryPath = Path.Combine(
                        AppContext.BaseDirectory, 
                        "runtimes",
                        $"{os}-{arch}", 
                        $"{filename}{ext}"),
                    OperatingSystem = os
                };
            }
        }   
    }
}