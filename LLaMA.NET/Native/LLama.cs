using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{
    public static unsafe partial class LLama
    {
        [DllImport("llama", CharSet = CharSet.Ansi)]
        public static extern nint llama_init_from_file(string path_model, LLaMAContextParams parameters);

        [LibraryImport("llama", StringMarshalling = StringMarshalling.Custom, StringMarshallingCustomType = typeof(System.Runtime.InteropServices.Marshalling.AnsiStringMarshaller))]
        public static partial int llama_apply_lora_from_file(nint ctx, string path_lora, string path_base_model, int n_threads);

        [LibraryImport("llama", EntryPoint = "llama_free")]
        internal static partial void llama_free(nint context);

        [LibraryImport("llama", EntryPoint = "llama_eval")]
        internal static partial int llama_eval(nint context, int[] tokens, int n_tokens, int n_past, int n_threads);

        [LibraryImport("llama", EntryPoint = "llama_tokenize", StringMarshalling = StringMarshalling.Utf8)]
        internal static partial int llama_tokenize(nint context, string text, int[] tokens, int n_max_tokens, [MarshalAs(UnmanagedType.Bool)] bool add_bos);

        [LibraryImport("llama")]
        internal static partial int llama_n_vocab(nint context);

        [LibraryImport("llama")]
        internal static partial int llama_n_ctx(nint context);

        [LibraryImport("llama")]
        internal static partial float* llama_get_logits(nint context);

        [LibraryImport("llama")]
        internal static partial nint llama_get_embeddings(nint context);

        [LibraryImport("llama")]
        internal static partial nint llama_token_to_str(nint context, int token);
        [LibraryImport("llama")]
        public static partial void llama_sample_repetition_penalty(nint ctx, ref LlamaTokenDataArray candidates, int[] last_tokens, int last_tokens_size, float penalty);

        [LibraryImport("llama")]
        public static partial void llama_sample_frequency_and_presence_penalties(nint ctx, ref LlamaTokenDataArray candidates, int[] last_tokens, int last_tokens_size, float alpha_frequency, float alpha_presence);

        [LibraryImport("llama")]
        public static partial void llama_sample_softmax(nint ctx, ref LlamaTokenDataArray candidates);

        [LibraryImport("llama")]
        public static partial void llama_sample_top_k(nint ctx, ref LlamaTokenDataArray candidates, int k, nint min_keep = default);

        [LibraryImport("llama")]
        public static partial void llama_sample_top_p(nint ctx, ref LlamaTokenDataArray candidates, float p, nint min_keep = default);

        [LibraryImport("llama")]
        public static partial void llama_sample_tail_free(nint ctx, ref LlamaTokenDataArray candidates, float z, nint min_keep = default);

        [LibraryImport("llama")]
        public static partial void llama_sample_typical(nint ctx, ref LlamaTokenDataArray candidates, float p, nint min_keep = default);

        [LibraryImport("llama")]
        public static partial void llama_sample_temperature(nint ctx, ref LlamaTokenDataArray candidates, float temp);

        [LibraryImport("llama")]
        public static partial nint llama_sample_token_mirostat(nint ctx, ref LlamaTokenDataArray candidates, float tau, float eta, int m, float* mu);

        [LibraryImport("llama")]
        public static partial nint llama_sample_token_mirostat_v2(nint ctx, ref LlamaTokenDataArray candidates, float tau, float eta, float* mu);

        [LibraryImport("llama")]
        public static partial nint llama_sample_token_greedy(nint ctx, ref LlamaTokenDataArray candidates);

        [LibraryImport("llama")]
        public static partial nint llama_sample_token(nint ctx, ref LlamaTokenDataArray candidates);

        [LibraryImport("llama")]
        internal static partial nint llama_print_system_info();

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void ProgressCallback(float progress, nint ctx);

        [LibraryImport("llama")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static partial bool llama_mmap_supported();

        [LibraryImport("llama")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static partial bool llama_mlock_supported();

        [LibraryImport("llama")]
        public static partial void llama_set_rng_seed(nint ctx, int seed);
        [LibraryImport("llama")]
        public static partial int llama_get_kv_cache_token_count(nint ctx);

        [LibraryImport("llama")]
        public static partial nint llama_get_state_size(nint ctx);

        [LibraryImport("llama")]
        public static partial nint llama_copy_state_data(nint ctx, nint dest, nint dest_size);

        
        [LibraryImport("llama")]
        internal static partial int llama_token_nl();

        [LibraryImport("llama")]
        internal static partial int llama_token_bos();

        [LibraryImport("llama")]
        internal static partial int llama_token_eos();
    }
}