using System.Runtime.InteropServices;
using LLaMA.NET.LibLoader;
using LLaMA.NET.Native;

namespace LLaMA.NET
{
    public unsafe class LLM : IDisposable
    {
        private readonly int _threads = 8;

        public string ModelName;
        public nint Context;
        public LLaMAContextParams ContextParams;
        private int[] _embeds = Array.Empty<int>();
        private int[] firstPrompt = Array.Empty<int>();
        private int[] lastPrompt = Array.Empty<int>();
        private bool isDisposed = false;
        public bool Busy;

        static LLM() => LibraryLoader.LoadNativeLib();
        public LLM(string modelPath, int threads = 2)
        {
            _threads = threads;
            ContextParams.n_ctx = 2048;
            ContextParams.seed = 1;
            ContextParams.use_mmap = true;
            ModelName = Path.GetFileName(modelPath);
            Context = LLama.llama_init_from_file(modelPath, ContextParams);
            LLama.llama_set_rng_seed(Context, 1);
            Console.WriteLine(Marshal.PtrToStringAnsi(LLama.llama_print_system_info()));
        }

        public IEnumerable<string> IngestPrompt(string prompt)
        {
            var inputTokens = new int[prompt.Length + 1];
            var tokens = LLama.llama_tokenize(Context, prompt, inputTokens, inputTokens.Length, true);

            if (tokens > ContextParams.n_ctx)
            {
                yield return $"Prompt is too long. Max length is {ContextParams.n_ctx} tokens.";
                yield break;
            }

            Array.Resize(ref inputTokens, tokens);

            Lobotomize();

            LLama.llama_eval(Context, inputTokens, inputTokens.Length, _embeds.Length, _threads);

            _embeds = _embeds.Concat(inputTokens).ToArray();
            if (firstPrompt.Length == 0)
                firstPrompt = inputTokens.ToArray();
            lastPrompt = inputTokens.ToArray();
        }

        public IEnumerable<string> InferenceStream
        (
            int nTokensToPredict,
            string[] reversePrompts,
            bool ignoreEos = false,
            int top_k = 40,
            float top_p = 0.8f,
            float temperature = 0.85f,
            float repetition_penalty = 1f,
            int mirostat = 2,
            float surprise = 0.2f,
            float learningRate = 0.1f,
            float tfs_z = 1.5f,
            float typical_p = 0.5f,
            CancellationToken ctx = default
        )
        {
            var msg = string.Empty;
            for (int i = 0; i < nTokensToPredict; i++)
            {
                if (ctx.IsCancellationRequested)
                {
                    _embeds = _embeds[..^(i + lastPrompt.Length)];
                    break;
                }

                Lobotomize();

                var candidates = GetCandidates();
                LLama.llama_sample_repetition_penalty(Context, ref candidates, _embeds, _embeds.Length, repetition_penalty);
                LLama.llama_sample_frequency_and_presence_penalties(Context, ref candidates, _embeds, _embeds.Length, 1, 1);

                int tokenId=0;
                if (temperature <= 0)
                {
                    tokenId = LLama.llama_sample_token_greedy(Context, ref candidates).ToInt32();
                }
                else
                {
                    if (mirostat == 1)
                        tokenId = MirostatV1(surprise, learningRate, ref candidates).ToInt32();
                    if (mirostat == 2)
                        tokenId = MirostatV2(temperature, surprise, learningRate, ref candidates).ToInt32();
                    else
                    {
                        LLama.llama_sample_top_k(Context, ref candidates, top_k, 12);
                        LLama.llama_sample_tail_free(Context, ref candidates, tfs_z, 9);
                        LLama.llama_sample_typical(Context, ref candidates, typical_p, 6);
                        LLama.llama_sample_top_p(Context, ref candidates, top_p, 3);
                        LLama.llama_sample_temperature(Context, ref candidates, temperature);
                        tokenId = LLama.llama_sample_token(Context, ref candidates).ToInt32();
                    }
                }
                candidates.Free();

                var res = Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, tokenId)) ?? string.Empty;
                msg += res;
                int[] newEmbds = { tokenId };
                _embeds = _embeds.Concat(newEmbds).ToArray();
                LLama.llama_eval(Context, newEmbds, 1, _embeds.Length, _threads);

                // for (int y = 0; y < reversePrompts.Length; y++)
                // {
                //     var last = _embeds.TakeLast(3).ToArray();
                //     var str = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(Context, last[0]));
                //     str += Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(Context, last[1]));
                //     str += Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(Context, last[2]));
                //     // str = str.Trim();
                //     if(str.Contains('\n'))
                //         yield return "\n";
                //     if (string.Equals(str, reversePrompts[y], StringComparison.InvariantCultureIgnoreCase) || str.Contains('\n'))
                //         yield break;
                // }

                if (tokenId == LLama.llama_token_eos() && !ignoreEos)
                    break;

                yield return res;
            }
            Busy = false;
        }

        private nint MirostatV2(float temperature, float tau, float eta, ref llama_token_data_array candidates)
        {
            float mu = 2.0f * tau;
            int m = 100;
            LLama.llama_sample_temperature(Context, ref candidates, temperature);
            return LLama.llama_sample_token_mirostat(Context, ref candidates, tau, eta, m, &mu);
        }

        private nint MirostatV1(float tau, float eta, ref llama_token_data_array candidates)
        {
            float mu = 2.0f * tau;
            int m = 100;
            return LLama.llama_sample_token_mirostat(Context, ref candidates, tau, eta, m, &mu);
        }

        private void Lobotomize()
        {
            if (_embeds.Length < ContextParams.n_ctx - 1)
                return;

            var newEmbeds = new int[ContextParams.n_ctx];
            Array.Copy(firstPrompt, 0, newEmbeds, 0, firstPrompt.Length);
            Array.Copy(_embeds, _embeds.Length - ContextParams.n_ctx + firstPrompt.Length, newEmbeds, firstPrompt.Length, ContextParams.n_ctx - firstPrompt.Length);

            LLama.llama_eval(Context, _embeds, _embeds.Length, 1, _threads);
        }

        private llama_token_data_array GetCandidates()
        {
            var logits = LLama.llama_get_logits(Context);
            var vocab = LLama.llama_n_vocab(Context);

            var candidates = new List<llama_token_data>();
            for (int token_id = 0; token_id < vocab; token_id++)
                candidates.Add(new llama_token_data(token_id, logits[token_id], 0.0f));

            var candidates_p = new llama_token_data_array(candidates.ToArray());

            // LLama.llama_sample_typical(Context, ref candidates_p, 1.0f);
            return candidates_p;
        }

        /// <summary>
        ///     Makes the model forget everything it has seen so far.
        /// </summary>
        public void ClearContext() => _embeds = Array.Empty<int>();

        public void Dispose()
        {
            if (isDisposed)
                return;

            GC.SuppressFinalize(this);

            if (Context != nint.Zero)
                LLama.llama_free(Context);

            isDisposed = true;
        }
    }
}