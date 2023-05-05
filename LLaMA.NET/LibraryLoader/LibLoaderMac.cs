using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader;

public static partial class MacLoader
{
    [LibraryImport("libSystem.dylib", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlopen(string filename, int flags);

    [LibraryImport("libSystem.dylib", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlsym(IntPtr handle, string symbol);

    public static void LibraryLoad(string libraryPath)
    {
        IntPtr libHandle = dlopen(libraryPath, RTLD_LAZY);
        if (libHandle == IntPtr.Zero)
            throw new Exception($"Failed to load {libraryPath}");
    }

    private const int RTLD_LAZY = 0x00001;
}