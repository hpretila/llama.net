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
        public bool use_mmap;
        public LLama.ProgressCallback progress_callback;
        public IntPtr progress_callback_user_data;
    }
}