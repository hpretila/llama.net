using System;
using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{

    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaTokenData
    {
        public int token_id;
        public float logit;
        public float score;

        public LlamaTokenData(int token_id, float logit, float score)
        {
            this.token_id = token_id;
            this.logit = logit;
            this.score = score;
        }
    }
}