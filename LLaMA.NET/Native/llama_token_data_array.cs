using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaTokenDataArray
    {
        public IntPtr ptr;
        public UIntPtr length;

        public LlamaTokenDataArray(LlamaTokenData[] data)
        {
            length = (UIntPtr)data.Length;
            ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenData)) * data.Length);
            for (int i = 0; i < data.Length; i++)
            {
                Marshal.StructureToPtr(data[i], IntPtr.Add(ptr, i * Marshal.SizeOf(typeof(LlamaTokenData))), false);
            }
        }

        public void Free()
        {
            Marshal.FreeHGlobal(ptr);
            length = UIntPtr.Zero;
        }
    }
}