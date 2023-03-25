using System;
using System.Runtime.InteropServices;

using LLaMA.NET.LibLoader;
using LLaMA.NET.LibLoader.Loaders;

namespace LLaMA.NET.LibLoader
{
    public static class LibLoader {
        public static void LibraryLoad() {
            LibraryEnvironment libEnv = NativeHelper.GetNativeLibraryEnvironment;

            switch (libEnv.OperatingSystem) {
                case "win":
                    WindowsLoader.LibraryLoad(libEnv.LibraryPath);
                    break;
                case "linux":
                    LinuxLoader.LibraryLoad(libEnv.LibraryPath);
                    break;
                case "mac":
                    MacLoader.LibraryLoad(libEnv.LibraryPath);
                    break;
            }
        }
    }
}