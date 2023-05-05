using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{
    [StructLayout(LayoutKind.Sequential)]
    public struct llama_token_data_array
    {
        public IntPtr ptr;
        public UIntPtr length;

        public llama_token_data_array(llama_token_data[] data)
        {
            length = (UIntPtr)data.Length;
            ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(llama_token_data)) * data.Length);
            for (int i = 0; i < data.Length; i++)
            {
                Marshal.StructureToPtr(data[i], IntPtr.Add(ptr, i * Marshal.SizeOf(typeof(llama_token_data))), false);
            }
        }

        public void Free()
        {
            Marshal.FreeHGlobal(ptr);
            length = UIntPtr.Zero;
        }
    }
}