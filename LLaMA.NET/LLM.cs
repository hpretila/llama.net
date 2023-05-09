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
            ContextParams.n_ctx = 2049;
            ContextParams.seed = 1;
            ContextParams.f16_kv = true;
            ContextParams.embedding = false;
            ContextParams.logits_all = false;
            ContextParams.progress_callback += (float progress, nint ctx) => Console.WriteLine($"Progress_Callback: Never Fires {progress}");
            ContextParams.use_mmap = true;
            ModelName = Path.GetFileName(modelPath);
            Context = LLama.llama_init_from_file(modelPath, ContextParams);
            LLama.llama_set_rng_seed(Context, 1); // not needed?
            Console.WriteLine(Marshal.PtrToStringAnsi(LLama.llama_print_system_info()));
        }

        public IEnumerable<string> IngestPrompt(string prompt)
        {
            var inputTokens = new int[prompt.Length + 1];
            var tokens = LLama.llama_tokenize(Context, prompt, inputTokens, inputTokens.Length, true);

            if (tokens >= ContextParams.n_ctx)
            {
                yield return $"Prompt is too long. Max length is {ContextParams.n_ctx} tokens.";
                yield break;
            }
            Array.Resize(ref inputTokens, tokens);

            int batchSize = 1; // Set the batch size
            for (int i = 0; i < inputTokens.Length; i += batchSize)
            {
                LLama.llama_eval(Context, new[] { inputTokens[i] }, 1, Math.Min(ContextParams.n_ctx-1, _embeds.Length+i), _threads);
                yield return Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, inputTokens[i])) ?? string.Empty;
            }

            // int batchSize = 256; // Set the batch size
            // for (int i = 0; i < inputTokens.Length; i += batchSize)
            // {
            //     int count = Math.Min(batchSize, inputTokens.Length - i); // Calculate the number of tokens in the current batch
            //     var batch = new int[count];
            //     for (int j = 0; j < count; j++)
            //         batch[j] = inputTokens[i + j]; // Copy the token data from inputTokens to batch

            //     LLama.llama_eval(Context, batch, count, Math.Min(ContextParams.n_ctx-1, _embeds.Length+i), _threads);

            //     for (int j = 0; j < batch.Length; j++)
            //         yield return Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, batch[j])) ?? string.Empty;
            // }

            _embeds = _embeds.Concat(inputTokens).ToArray();
            // while (_embeds.Length > ContextParams.n_ctx - 1)
            //     Lobotomize();

            if (firstPrompt.Length == 0)
                firstPrompt = inputTokens.ToArray();
            lastPrompt = inputTokens.ToArray();
        }

        public IEnumerable<string> InferenceStream
        (
            int nTokensToPredict,
            string[]? reversePrompts=null,
            bool ignoreEos = false,
            int top_k = 40,
            float top_p = 0.8f,
            float temp = 0.25f, // 0 = greedy
            float repetition_penalty = 1f, // 1=disabled
            int mirostat = 1,
            float tauSurpriseOrEntropy = 3f,
            float etaLearningRate = 0.01f,
            float tailFreeSampling = 1f, // broken 1=disabled   
            float typical_p = 1f, // broken 1=disabled
            bool penalize_nl = false,
            bool penalize_space = false,
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

                // while (_embeds.Length > ContextParams.n_ctx - 1)
                //     Lobotomize();

                int tokenId = SampleToken(top_k, top_p, temp, repetition_penalty, mirostat, tauSurpriseOrEntropy, etaLearningRate, tailFreeSampling, typical_p, penalize_nl);

                var res = Marshal.PtrToStringUTF8(LLama.llama_token_to_str(Context, tokenId)) ?? string.Empty;
                msg += res;
                int[] newEmbds = { tokenId };
                _embeds = _embeds.Concat(newEmbds).ToArray();
                LLama.llama_eval(Context, newEmbds, 1, Math.Min(ContextParams.n_ctx-1, _embeds.Length), _threads);

                // abort if AI keeps repeating itself
                if(_embeds.Length > 3)
                {
                    var sum = 0;
                    for(int y = _embeds.Length; y > _embeds.Length -3; y--)
                        sum += _embeds[y];

                    if(sum == _embeds[_embeds.Length] * 3)
                        yield break;
                }


                // for (int y = 0; y < reversePrompts.Length; y++)
                // {
                //     var last = _embeds.TakeLast(3).ToArray();
                //     var str = Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, last[0]));
                //     str += Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, last[1]));
                //     str += Marshal.PtrToStringAnsi(LLama.llama_token_to_str(Context, last[2]));
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

        private int SampleToken(int top_k, float top_p, float temp, float repetition_penalty, int mirostat, float tauSurpriseOrEntropy, float etaLearningRate, float tailFreeSampling, float typical_p, bool penalize_nl)
        {
            var candidates_p = GetCandidates(repetition_penalty, penalize_nl, penalizeSpaces: false);
            LLama.llama_sample_repetition_penalty(Context, ref candidates_p, _embeds, _embeds.Length, repetition_penalty);
            LLama.llama_sample_frequency_and_presence_penalties(Context, ref candidates_p, _embeds, _embeds.Length, 0, 0);

            int tokenId;
            if (temp > 0)
            {
                if (mirostat == 1)
                {
                    float mu = 2.0f * tauSurpriseOrEntropy;
                    int m = 100;
                    tokenId= LLama.llama_sample_token_mirostat(Context, ref candidates_p, tauSurpriseOrEntropy, etaLearningRate, m, &mu).ToInt32();
                }
                else if (mirostat == 2)
                {
                    float mu = 2.0f * tauSurpriseOrEntropy;
                    int m = 100;
                    LLama.llama_sample_temperature(Context, ref candidates_p, temp);
                    tokenId= LLama.llama_sample_token_mirostat(Context, ref candidates_p, tauSurpriseOrEntropy, etaLearningRate, m, &mu).ToInt32();
                }
                else
                {
                    LLama.llama_sample_top_k(Context, ref candidates_p, top_k, 10);
                    LLama.llama_sample_tail_free(Context, ref candidates_p, tailFreeSampling, 10);
                    LLama.llama_sample_typical(Context, ref candidates_p, typical_p, 10);
                    LLama.llama_sample_top_p(Context, ref candidates_p, top_p, 10);
                    LLama.llama_sample_temperature(Context, ref candidates_p, temp);
                    tokenId = LLama.llama_sample_token(Context, ref candidates_p).ToInt32();
                }
            }
            else
                tokenId = LLama.llama_sample_token_greedy(Context, ref candidates_p).ToInt32();

            candidates_p.Free();
            return tokenId;
        }

        private void Lobotomize()
        {
            // var new_embeds = _embeds[(_embeds.Length / 2)..];
            // LLama.llama_eval(Context, new_embeds, new_embeds.Length, 1, _threads);
            // _embeds = new_embeds;
        }

        private LlamaTokenDataArray GetCandidates(float repetition_penalty, bool penalizeNewLines = false, bool penalizeSpaces = false)
        {
            var logits = LLama.llama_get_logits(Context);
            var vocab = LLama.llama_n_vocab(Context);
            float newLineLogit = logits[LLama.llama_token_nl()];
            // float spaceLogit = logits[LLama.llama_token_bos()]; oops this isnt a space

            var candidates = new List<LlamaTokenData>();
            for (int token_id = 0; token_id < vocab; token_id++)
                candidates.Add(new LlamaTokenData(token_id, logits[token_id], 0.0f));

            var candidates_p = new LlamaTokenDataArray(candidates.ToArray());

            if (!penalizeNewLines)
                logits[LLama.llama_token_nl()] = newLineLogit;
            // if (!penalizeSpaces)
            //     logits[LLama.llama_token_bos()] = spaceLogit;

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