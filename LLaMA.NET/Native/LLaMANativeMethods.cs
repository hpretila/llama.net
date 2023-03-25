using System;
using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{
    
    using llama_token = Int32;

    /// <summary>
    /// The LLaMA token data.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLaMATokenData
    {
        public llama_token id;  // token id

        public float p;     // probability of the token
        public float plog;  // log probability of the token
    }

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
    public static class LLaMANativeMethods
    {

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern LLaMAContextParams llama_context_default_params();

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_init_from_file(string path_model, LLaMAContextParams parameters);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_quantize(string fname_inp, string fname_out, int itype, int qk);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_eval(IntPtr context, [In] llama_token[] tokens, int n_tokens, int n_past, int n_threads);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_tokenize(IntPtr context, string text, [Out] llama_token[] tokens, int n_max_tokens, bool add_bos);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_vocab(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_ctx(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_logits(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_embeddings(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_token_to_str(IntPtr context, llama_token token);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_token_bos();

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_token_eos();

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_sample_top_p_top_k(IntPtr context, [In] llama_token[] last_n_tokens_data, int last_n_tokens_size, int top_k, double top_p, double temp, double repeat_penalty);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_print_timings(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_reset_timings(IntPtr context);

        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_print_system_info();
    }
}