/**
 * Testing script!
 * Using https://chat.openai.com/chat/75848fef-e306-4cac-aa3f-cc55843ffbb0 for reference!
 */

using System;
using System.Runtime.InteropServices;

using LLaMA.NET;
using LLaMA.NET.Native;
using llama_token = System.Int32;

static void Test_Native_Inference()
{
    Console.WriteLine("\n ======= Testing Native LLaMA.NET... =======");
        
    const int N_THREADS = 8;
    const int nTokensToPredict = 50;
    const string prompt = " This is the story of a man named ";
    IntPtr ctx;

    var lparams = LLaMANativeMethods.llama_context_default_params();

    // load model
    ctx = LLaMANativeMethods.llama_init_from_file(
                "/mnt/d/LLaMA/7B-LoRA/ggml-model-q4_0.bin", 
                lparams
        );

    // convert prompt to embedings
    var embd_inp = new llama_token[prompt.Length + 1];
    var n_of_tok = LLaMANativeMethods.llama_tokenize(ctx, prompt, embd_inp, embd_inp.Length, true);
    Array.Resize(ref embd_inp, n_of_tok);

    // evaluate the prompt
    for (int i = 0; i < embd_inp.Length; i++) {
        // batch size 1
        LLaMANativeMethods.llama_eval(ctx, new llama_token[] { embd_inp[i] }, 1, i, N_THREADS);
    }

    var prediction = "";
    var embd = embd_inp;

    // for (number of tokens to predict) {
    for (int i = 0; i < nTokensToPredict; i++) {
        var id = LLaMANativeMethods.llama_sample_top_p_top_k(ctx, null, 0, 40, 0.8f, 0.2f, 1f/0.85f);

        // TODO: break here if EOS

        // add it to the context (all tokens, prompt + predict)
        var newEmbds = new llama_token[] { id };
        embd = embd.Concat(newEmbds).ToArray();

        // Get res!
        var res = Marshal.PtrToStringAnsi(LLaMANativeMethods.llama_token_to_str(ctx, id));
        
        // Write it
        System.Console.Write(res);

        // add to string
        prediction += res;

        // eval next token
        LLaMANativeMethods.llama_eval(ctx, newEmbds, 1, embd.Length, N_THREADS);
    }

    LLaMANativeMethods.llama_free(ctx); // cleanup
}

static void Test_API_Inference () {
    Console.WriteLine("\n ======= Testing API LLaMA.NET... ==========");
    
    const int nTokensToPredict = 50;

    LLaMAModel model = LLaMAModel.FromPath("/mnt/d/LLaMA/7B-LoRA/ggml-model-q4_0.bin");
    LLaMARunner runner = model.CreateRunner()
        .WithPrompt(" This is the story of a man named ");

    int i = 0;
    bool stop = false || i >= nTokensToPredict;
    while (!stop) {
        bool isEOS;
        var res = runner.Infer(out isEOS, 1);
        Console.Write(res);
        i++;

        stop = isEOS || i >= nTokensToPredict;
    }

    model.Dispose();
}

Test_Native_Inference();
Test_API_Inference();