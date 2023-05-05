namespace LLaMA.NET.LibLoader;

public static class LibraryLoader {
    public static void LoadNativeLib() {
        LibraryEnvironment libEnv = NativeHelper.GetNativeLibraryEnvironment;

        switch (libEnv.OperatingSystem) {
            case "win":
                WindowsLoader.LibraryLoad(libEnv.LibraryPath);
                break;
            case "linux":
                LinuxLoader.LibraryLoad(libEnv.LibraryPath);
                break;
            case "osx":
                MacLoader.LibraryLoad(libEnv.LibraryPath);
                break;
        }
    }
}