using System.Runtime.InteropServices;
using LLaMA.NET.Native;

namespace LLaMA.NET
{
    public class LLaMARunner : IDisposable
    {
        private readonly LLaMAModel _model;
        private readonly int _threads = 8;
        private int[] _embeds = Array.Empty<int>();

        public LLaMARunner(LLaMAModel model, int threads = 4)
        {
            _model = model;
            _threads = threads;
        }

        public void Instruction(string instruction, string input = "", string response = "")
        {
            var betterPrompt = $"Below is an instruction that describes a task";
            if (string.IsNullOrEmpty(input))
                betterPrompt += ".";
            else
                betterPrompt += ", paired with an input that provides further context.";

            betterPrompt += $"Write a response that appropriately completes the request.\n\n";
            betterPrompt += "### Instruction:\n";
            betterPrompt += $"{instruction}\n\n"; 
            if(!string.IsNullOrEmpty(input))
            {
                betterPrompt += "### Input:\n";
                betterPrompt += $"{input}\n\n";
            }
            betterPrompt += "### Response:\n\n";
            betterPrompt += $"{response}";

            IngestPrompt(betterPrompt);
        }

        public void Continuation(string beginning) => Instruction("Continue the following text", beginning, beginning);

        public void IngestPrompt(string prompt)
        {
            Console.WriteLine(prompt);
            var inputEmbeds = new int[prompt.Length + 1];
            var inputTokenLength = LLaMANativeMethods.llama_tokenize(_model.ctx.Value, prompt, inputEmbeds, inputEmbeds.Length, true);
            Array.Resize(ref inputEmbeds, inputTokenLength);

            _ = LLaMANativeMethods.llama_eval(_model.ctx.Value, inputEmbeds, inputEmbeds.Length, _embeds.Length, _threads);
            _embeds = _embeds.Concat(inputEmbeds).ToArray();
        }

        public (int tokens, string text, bool isEos) Inference(int nTokensToPredict = 50)
        {
            var isEos = false;
            var prediction = "";
            int tokens;
            for (tokens = 0; tokens < nTokensToPredict; tokens++)
            {
                // Grab the next token
                var tokenId = LLaMANativeMethods.llama_sample_top_p_top_k(_model.ctx.Value, Array.Empty<int>(), 0, 40, 0.8f, 0.2f, 1f / 0.85f);

                if (tokenId == LLaMANativeMethods.llama_token_eos())
                {
                    isEos = true;
                    break;
                }

                // Add it to the context (all tokens, prompt + predict)
                var newEmbds = new int[] { tokenId };
                _embeds = _embeds.Concat(newEmbds).ToArray();

                // Get res!
                var res = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(_model.ctx.Value, tokenId));

                // Add to string
                prediction += res;

                // eval next token
                _ = LLaMANativeMethods.llama_eval(_model.ctx.Value, newEmbds, 1, _embeds.Length, _threads);
            }

            return (tokens, prediction, isEos);
        }

        public IEnumerable<string> InferenceStream(int nTokensToPredict = 50)
        {
            var newLineCounter = 0;
            var tokens = 0;
            (int count, string text, bool eos) = (0, "", false);
            for (int i = 0; i < nTokensToPredict; i++)
            {
                (count, text, eos) = Inference(1);
                tokens += count;

                if (eos)
                    break;

                if (text == Environment.NewLine)
                {
                    newLineCounter++;
                    if (newLineCounter > 2)
                        break;
                }
                else
                    newLineCounter = 0;

                yield return text;
            }
            yield return $"{Environment.NewLine}Tokens: {tokens} - Termination reason {(eos ? "EOS" : (tokens == nTokensToPredict ? "Token Limit" : (newLineCounter > 2 ? "New Line Limit" : "Unknown")))}";
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