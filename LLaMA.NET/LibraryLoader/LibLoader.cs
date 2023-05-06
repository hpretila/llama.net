using System.Runtime.InteropServices;

namespace LLaMA.NET.LibLoader;

public static partial class LibraryLoader 
{
    [LibraryImport("libdl.so", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlopen(string filename, int flags);

    [LibraryImport("libdl.so.2", EntryPoint = "dlopen", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr dlopen2(string filename, int flags);

    public static void LoadNativeLib()
    {
        nint libHandle;
        try
        {
            libHandle = dlopen(AppDomain.CurrentDomain.BaseDirectory + "llama.so", 0x00001);
        }
        catch (Exception e)
        {
            libHandle = dlopen2(AppDomain.CurrentDomain.BaseDirectory + "llama.so", 0x00001);
            Console.WriteLine(e);
        }

        if (libHandle == nint.Zero)
            throw new Exception($"Failed to load {AppDomain.CurrentDomain.BaseDirectory + "llama.so"}");
    }
}