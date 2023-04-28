using LLaMA.NET.Native;
using LLaMA.NET.LibLoader;

namespace LLaMA.NET
{
    /// <summary>
    /// A factory for LLaMA models. Processed with <seealso cref="WhisperProcessor"/>
    /// </summary>
    public class LLaMAModel : IDisposable
    {
        public Lazy<IntPtr> ctx;
        private bool isDisposed = false;

        private LLaMAModel(IntPtr context) => ctx = new Lazy<IntPtr>(() => context);

        /// <summary>
        /// Creates a new LLaMAModelFactory from a model path.
        /// </summary>
        /// <param name="modelPath">The path to the model.</param>
        public static LLaMAModel FromPath(string modelPath)
        {
            LibLoader.LibLoader.LibraryLoad();

            return new LLaMAModel(
                LLaMANativeMethods.llama_init_from_file(
                    modelPath, 
                    new LLaMAContextParams{
                        seed = 1,
                        n_ctx = 1024
                    }
                )
            );
        }

        /// <summary>
        /// Creates a LLaMARunner for this model.
        /// </summary>
        /// <returns>A new LLaMARunner.</returns>
        public LLaMARunner CreateRunner(int threads = 4) => new (this, threads);

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            // If already disposed, do nothing.
            if (isDisposed)
            {
                return;
            }

            // Dispose of unmanaged resources.
            if (ctx.IsValueCreated && ctx.Value != IntPtr.Zero)
            {
                LLaMANativeMethods.llama_free(ctx.Value);
            }

            isDisposed = true;
        }
    }
}