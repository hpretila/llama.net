using LLaMA.NET.LibLoader;
using LLaMA.NET.Native;

namespace LLaMA.NET
{
    /// <summary>
    /// A factory for LLaMA models. Processed with <seealso cref="WhisperProcessor"/>
    /// </summary>
    public unsafe class LLaMAModel : IDisposable
    {
        public string ModelName;
        public IntPtr ctx;
        private bool isDisposed = false;
        public readonly LLaMAContextParams ContextParams;

        static LLaMAModel() => LibLoader.LibLoader.LibraryLoad();
        public LLaMAModel(string modelPath)
        {   
            ContextParams = LLaMANativeMethods.llama_context_default_params();
            ContextParams.n_ctx = 64;
            ctx = (IntPtr)LLaMANativeMethods.llama_init_from_file(modelPath, ContextParams).ToPointer();
            ModelName = Path.GetFileName(modelPath);
        }

        /// <summary>
        /// Creates a LLaMARunner for this model.
        /// </summary>
        /// <returns>A new LLaMARunner.</returns>
        public LLaMARunner CreateRunner(int threads = 4) => new(this, threads);

        public void Dispose()
        {
            if (isDisposed)
                return;

            GC.SuppressFinalize(this);
         
            if (ctx != IntPtr.Zero)
                LLaMANativeMethods.llama_free(ctx);

            isDisposed = true;
        }
    }
}