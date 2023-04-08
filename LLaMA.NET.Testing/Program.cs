/**
 * Testing script!
 * Using https://chat.openai.com/chat/75848fef-e306-4cac-aa3f-cc55843ffbb0 for reference!
 */

using System;
using System.Runtime.InteropServices;

using LLaMA.NET;
using LLaMA.NET.Native;
using LLaMA.NET.LibLoader;
using llama_token = System.Int32;

static void Test_Model_Quantisation()
{

}

static void Test_Native_Inference()
{
    Console.WriteLine("\n ======= Testing Native LLaMA.NET... =======");
        
    const int N_THREADS = 8;
    const int nTokensToPredict = 50;
    const string prompt = "### Instruction ###\nTell me about Emanuel Macron, King of France.\n### Response ###\n";
    IntPtr ctx;

    LibLoader.LibraryLoad();

    var lparams = LLaMANativeMethods.llama_context_default_params();

    // load model
    ctx = LLaMANativeMethods.llama_init_from_file(
        System.IO.Path.Join("/path/to/llama-7b-lora","ggml-model-q4_0.bin"),
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

    LLaMAModel model = LLaMAModel.FromPath(System.IO.Path.Join("/Volumes/AI DRIVE/weights/llama-7b-lora","ggml-model-q4_0.bin"));
    LLaMARunner runner = model.CreateRunner()
        .WithPrompt("### Instruction ###\nTell me about Emanuel Macron, King of France.\n### Response ###\n");

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


static void Test_API_Inference_Interactive () {
    Console.WriteLine("\n ======= Running Interactive API Test with LLaMA.NET... ==========");
    
    const int nTokensToPredict = 150;

    LLaMAModel model = LLaMAModel.FromPath(System.IO.Path.Join("/Volumes/AI DRIVE/weights/llama-7b-lora","ggml-model-q4_0.bin"));
    LLaMARunner runner = model.CreateRunner()
        .WithPrompt("### Instruction ###\nTell me about Emanuel Macron, King of France.\n### Response ###\n");

    int i = 0;
    bool stop = false || i >= nTokensToPredict;
    while (true) {
        Console.Write ("\n> ");
        string input = Console.ReadLine();
        input = "### Instruction ###\n" + input + "\n### Response ###\n";
        runner = runner.WithThreads(4).WithPrompt(input);
        string response = "";
        while (!stop) {
            bool isEOS;
            var res = runner.Infer(out isEOS, 5);
            response += res;
            Console.Write(".");
            i++;

            stop = isEOS || i >= nTokensToPredict || response.Contains("### Instruction");
        }
        i = 0;
        stop = false;
        if (response.Contains("### Instruction"))
            response = response.Substring(0, response.IndexOf("### Instruction")).Trim();
        Console.WriteLine("Done.\n" + response);
    }

    model.Dispose();
}

// Test_API_Inference();
Test_API_Inference_Interactive();