using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using LLaMA.NET.Native;

namespace LLaMA.NET
{
    public class LLaMARunner : IDisposable
    {
        private readonly LLaMAModel _model;
        private readonly int _threads = 8;
        private int[] _embeds = Array.Empty<int>();

        private int activeJobs = 0;
        public bool Busy = false;

        public LLaMARunner(LLaMAModel model, int threads = 4)
        {
            _model = model;
            _threads = threads;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerable<string> Instruction(string instruction, string input = "", string response = "", CancellationToken ctx = default)
        {
            // Interlocked.Increment(ref activeJobs);
            // var betterPrompt = $"Below is an instruction that describes a task";
            // if (string.IsNullOrEmpty(input))
            //     betterPrompt += ".";
            // else
            //     betterPrompt += ", paired with an input that provides further context.";

            // betterPrompt += $"Write a response that appropriately completes the request.\n\n";
            var betterPrompt = "### Instruction:\n";
            betterPrompt += $"{instruction}\n\n"; 
            if(!string.IsNullOrEmpty(input))
            {
                betterPrompt += "### Input:\n";
                betterPrompt += $"{input}\n\n";
            }
            betterPrompt += "### Response:\n";
            betterPrompt += $"{response}";

            return IngestPrompt(betterPrompt, ctx);
        }

        public IEnumerable<string> IngestPrompt(string prompt, CancellationToken ctx = default)
        {
            var inputEmbeds = new int[prompt.Length + 1];
            var inputTokenLength = LLaMANativeMethods.llama_tokenize(_model.ctx.Value, prompt, inputEmbeds, inputEmbeds.Length, true);
            Array.Resize(ref inputEmbeds, inputTokenLength);

            // Evaluate the prompt batchsize 1
            for (int i = 0; i < inputEmbeds.Length; i++)
            {
                if(ctx.IsCancellationRequested)
                    break;
                
                LLaMANativeMethods.llama_eval(_model.ctx.Value, new int[] { inputEmbeds[i] }, 1, _embeds.Length + i, _threads);
                var token = LLaMANativeMethods.llama_token_to_str(_model.ctx.Value, inputEmbeds[i]);
                var tokenStr = Marshal.PtrToStringAnsi(token) ?? string.Empty;
                yield return tokenStr;
            }
            _embeds = _embeds.Concat(inputEmbeds).ToArray();
        }

        public (int tokens, string text, bool isEos) Inference(int nTokensToPredict = 50, int top_k = 40, float top_p = 0.8f, float temperature = 0.85f, float repetition_penalty = 1f, CancellationToken ctx = default)
        {
            var isEos = false;
            var prediction = "";
            int tokens;
            for (tokens = 0; tokens < nTokensToPredict; tokens++)
            {
                // Grab the next token
                var tokenId = LLaMANativeMethods.llama_sample_top_p_top_k(_model.ctx.Value, Array.Empty<int>(), 0, top_k, top_p, temperature, repetition_penalty);
                var res = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx.Value, tokenId));

                if(res == "#" || res == "##")
                {
                    isEos =true;
                    break;
                }
                
                if (tokenId == LLaMANativeMethods.llama_token_eos())
                {
                    isEos = true;
                    break;
                }

                // Add it to the context (all tokens, prompt + predict)
                var newEmbds = new int[] { tokenId };
                _embeds = _embeds.Concat(newEmbds).ToArray();
                // Add to string
                prediction += res;

                // eval next token
                _ = LLaMANativeMethods.llama_eval(_model.ctx.Value, newEmbds, 1, _embeds.Length, _threads);
            }

            Interlocked.Decrement(ref activeJobs);
            return (tokens, prediction, isEos);
        }

        public IEnumerable<string> InferenceStream(int nTokensToPredict = 50, int top_k = 40, float top_p = 0.8f, float temperature = 0.85f, float repetition_penalty = 1f, CancellationToken ctx = default)
        {
            var newLineCounter = 0;
            var tokens = 0;
            (int count, string text, bool eos) = (default, string.Empty, default);
            for (int i = 0; i < nTokensToPredict; i++)
            {
                if(ctx.IsCancellationRequested)
                    break;
                (count, text, eos) = Inference(1,  top_k, top_p, temperature, repetition_penalty, ctx);;
                tokens++;

                if (eos)
                    break;

                if (text == Environment.NewLine)
                {
                    newLineCounter++;
                    if (newLineCounter > 3)
                        break;
                }
                else
                    newLineCounter = 0;

                yield return text;
            }
            Interlocked.Decrement(ref activeJobs);
            yield return $"{Environment.NewLine}Tokens: {tokens}, Termination reason {(eos ? "EOS" : (tokens == nTokensToPredict ? "Token Limit" : (newLineCounter > 3 ? "New Line Limit" : "Unknown")))}";
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