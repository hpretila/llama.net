using System.Runtime.InteropServices;
using LLaMA.NET.Native;

namespace LLaMA.NET
{
    public class LLaMARunner : IDisposable
    {
        private readonly LLaMAModel _model;
        private readonly int _threads = 8;
        private int[] _embeds = Array.Empty<int>();
        private int[] firstPrompt = Array.Empty<int>();
        private int[] lastPrompt = Array.Empty<int>();
        public bool Busy = false;

        public LLaMARunner(LLaMAModel model, int threads = 6)
        {
            _model = model;
            _threads = threads;
            Console.WriteLine(Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_print_system_info()));
        }

        public IEnumerable<string> IngestPrompt(string prompt, CancellationToken ctx = default)
        {
            var inputEmbeds = new int[prompt.Length];
            var inputTokenLength = LLaMANativeMethods.llama_tokenize(_model.ctx, prompt, inputEmbeds, inputEmbeds.Length, false);
            Array.Resize(ref inputEmbeds, inputTokenLength);

            // Evaluate the prompt batchsize 1
            for (int i = 0; i < inputEmbeds.Length; i++)
            {   
                var token = LLaMANativeMethods.llama_token_to_str(_model.ctx, inputEmbeds[i]);
                var tokenStr = Marshal.PtrToStringAnsi(token) ?? string.Empty;
                yield return tokenStr;
            }
            
            LLaMANativeMethods.llama_eval(_model.ctx, inputEmbeds, inputEmbeds.Length, _embeds.Length, _threads);

            _embeds = _embeds.Concat(inputEmbeds).ToArray();
            if (firstPrompt.Length == 0)
                firstPrompt = inputEmbeds.ToArray();
            lastPrompt = inputEmbeds.ToArray();
        }

        public IEnumerable<string> InferenceStream(int nTokensToPredict, string[] reversePrompts, bool ignoreEos = false, int top_k = 40, float top_p = 0.8f, float temperature = 0.85f, float repetition_penalty = 1f, CancellationToken ctx = default)
        {
            var msg = string.Empty;
            for (int i = 0; i < nTokensToPredict; i++)
            {
                if (ctx.IsCancellationRequested)
                {
                    _embeds = _embeds[..^(i + lastPrompt.Length)];
                    break;
                }

                if (_embeds.Length == _model.ContextParams.n_ctx - 1)
                {
                    _embeds = _embeds[(_embeds.Length / 2)..];
                    LLaMANativeMethods.llama_eval(_model.ctx, _embeds, _embeds.Length, 1, _threads);
                }

                var tokenId = LLaMANativeMethods.llama_sample_top_p_top_k(_model.ctx, _embeds, _embeds.Length, top_k, top_p, temperature, repetition_penalty);
                var res = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx, tokenId)) ?? string.Empty;
                msg += res;
                int[] newEmbds = { tokenId };
                _embeds = _embeds.Concat(newEmbds).ToArray();
                LLaMANativeMethods.llama_eval(_model.ctx, newEmbds, 1, _embeds.Length, _threads);

                // for (int y = 0; y < reversePrompts.Length; y++)
                // {
                //     var last = _embeds.TakeLast(3).ToArray();
                //     var str = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx, last[0]));
                //     str += Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx, last[1]));
                //     str += Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx, last[2]));
                //     // str = str.Trim();
                //     if(str.Contains('\n'))
                //         yield return "\n";
                //     if (string.Equals(str, reversePrompts[y], StringComparison.InvariantCultureIgnoreCase) || str.Contains('\n'))
                //         yield break;
                // }

                if (tokenId == LLaMANativeMethods.llama_token_eos() && !ignoreEos)
                    break;

                yield return res;
            }
        }


        public void Clear() => _embeds = Array.Empty<int>();

        public void Dispose()
        {
            Clear();
            _model.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}