using System;
using System.Runtime.InteropServices;

namespace LLaMA.NET.Native
{

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_token_data
    {
        public Int32 token_id;
        public float logit;
        public float score;

        public llama_token_data(Int32 token_id, float logit, float score)
        {
            this.token_id = token_id;
            this.logit = logit;
            this.score = score;
        }
    }
}