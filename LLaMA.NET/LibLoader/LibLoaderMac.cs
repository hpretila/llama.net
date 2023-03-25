using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader.Loaders
{
    public static class MacLoader
    {
        [DllImport("libSystem.dylib")]
        private static extern IntPtr dlopen(string filename, int flags);

        [DllImport("libSystem.dylib")]
        private static extern IntPtr dlsym(IntPtr handle, string symbol);

        public static void LibraryLoad(string libraryPath)
        {
            IntPtr libHandle = dlopen(libraryPath, RTLD_LAZY);
            if (libHandle == IntPtr.Zero)
            {
                throw new Exception($"Failed to load {libraryPath}");
            }
        }

        private const int RTLD_LAZY = 0x00001;
    }
}