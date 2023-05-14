using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader;

public static partial class LinuxLoader
{
    [LibraryImport("libdl.so", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlopen(string filename, int flags);

    [LibraryImport("libdl.so.2", EntryPoint = "dlopen", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlopen2(string filename, int flags);

    public static void LibraryLoad(string libraryPath)
    {
        IntPtr libHandle;
        try
        {
            libHandle = dlopen(libraryPath, RTLD_LAZY);
        }
        catch (DllNotFoundException)
        {
            libHandle = dlopen2(libraryPath, RTLD_LAZY);
        }

        if (libHandle == IntPtr.Zero)
        {
            throw new Exception($"Failed to load {libraryPath}");
        }

    }

    private const int RTLD_LAZY = 0x00001;
}
