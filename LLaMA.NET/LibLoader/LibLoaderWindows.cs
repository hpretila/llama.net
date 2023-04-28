using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader.Loaders
{
    public static partial class WindowsLoader
    {
        [LibraryImport("kernel32.dll", StringMarshalling = StringMarshalling.Utf8)]
        private static partial IntPtr LoadLibrary(string lpFileName);

        [LibraryImport("kernel32.dll", StringMarshalling = StringMarshalling.Utf8)]
        private static partial IntPtr GetProcAddress(IntPtr hModule, string lpProcName);

        public static void LibraryLoad(string libraryPath)
        {
            IntPtr libHandle = LoadLibrary(libraryPath);
            if (libHandle == IntPtr.Zero)
            {
                throw new Exception($"Failed to load {libraryPath}");
            }
        }
    }
}
