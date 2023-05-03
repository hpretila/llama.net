using System;
using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{
    /// <summary>
    /// The LLaMA context parameters.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLaMAContextParams
    {
        public int n_ctx;   // text context
        public int n_parts; // -1 for default
        public int seed;    // RNG seed, 0 for random
        public bool f16_kv;     // use fp16 for KV cache
        public bool logits_all; // the llama_eval() call computes all logits, not just the last one
        public bool vocab_only; // only load the vocabulary, no weights
        public bool use_mlock;  // force system to keep model in RAM
        public bool embedding;  // embedding mode only
    }

    /// <summary>
    /// The native methods for LLaMA.
    /// </summary>
    internal static partial class LLaMANativeMethods
    {

        [DllImport("llama", EntryPoint = "llama_context_default_params", CallingConvention = CallingConvention.Cdecl)]
        internal static extern LLaMAContextParams llama_context_default_params();

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern IntPtr llama_init_from_file(string path_model, LLaMAContextParams parameters);

        [LibraryImport("llama", EntryPoint = "llama_free")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial void llama_free(IntPtr context);

        [LibraryImport("llama", EntryPoint= "llama_model_quantize", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_model_quantize(string fname_inp, string fname_out, int itype, int qk);

        [LibraryImport("llama", EntryPoint = "llama_eval")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_eval(IntPtr context, int[] tokens, int n_tokens, int n_past, int n_threads);

        [LibraryImport("llama", EntryPoint = "llama_tokenize", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_tokenize(IntPtr context, string text, int[] tokens, int n_max_tokens, [MarshalAs(UnmanagedType.Bool)] bool add_bos);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_n_vocab(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_n_ctx(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial IntPtr llama_get_logits(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial IntPtr llama_get_embeddings(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial IntPtr llama_token_to_str(IntPtr context, int token);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_token_bos();

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_token_eos();

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial int llama_sample_top_p_top_k(IntPtr context, int[] last_n_tokens_data, int last_n_tokens_size, int top_k, double top_p, double temp, double repeat_penalty);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial void llama_print_timings(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial void llama_reset_timings(IntPtr context);

        [LibraryImport("llama")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        internal static partial IntPtr llama_print_system_info();
    }
}