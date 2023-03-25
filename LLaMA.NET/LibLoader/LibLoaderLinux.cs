using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader.Loaders
{
    public static class LinuxLoader
    {
        [DllImport("libdl.so")]
        private static extern IntPtr dlopen(string filename, int flags);

        [DllImport("libdl.so.2", EntryPoint = "dlopen")]
        private static extern IntPtr dlopen2(string filename, int flags);

        public static void LibraryLoad(string libraryPath)
        {
            IntPtr libHandle;
            try {
                libHandle = dlopen(libraryPath, RTLD_LAZY);
            } catch (DllNotFoundException) {
                libHandle = dlopen2(libraryPath, RTLD_LAZY);
            }
            
            if (libHandle == IntPtr.Zero)
            {
                throw new Exception($"Failed to load {libraryPath}");
            }

        }

        private const int RTLD_LAZY = 0x00001;
    }
}