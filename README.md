# LLaMA.NET
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

.NET library to run [LLaMA](https://arxiv.org/abs/2302.13971) using [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). 

## Build 🧰
To build the library, you need to have [CMake](https://cmake.org/) and Python installed. Then, run the following commands at the root of the repository.

```bash
# Pull the submodules
git submodule update --init --recursive

# Build and prepare the C++ library
python scripts/build_llama_cpp.py
```

Then, build the .NET library using `dotnet`:

```bash
# Build the .NET library
dotnet build LLaMA.NET/LLaMA.NET.csproj
```

The built library should be located at `LLaMA.NET/bin/Debug/netXXXX/LLaMA.NET.dll`.

Currently only Linux is supported. Work is being done to dynamically load the C++ library on other platforms.

## Usage 📖

### Model Preparation
To use the library, you need to have a model. It needs to be converted to a binary format that can be loaded by the library. See [llama.cpp/README.md](llama.cpp/README.md) for more information on how to convert a model.

The model directory should contain the following files:
- `ggml-model-q4_0.bin`: The model file.
- `params.json`: The model parameters.
- `tokenizer.model`: The tokenizer model.

### Inference
To run inference, you need to load a model and create a runner. The runner can then be used to run inference on a prompt.
```csharp
using LLaMA.NET;

LLaMAModel model = LLaMAModel.FromPath("/path/to/your/ggml-model-q4_0.bin");
LLaMARunner runner = model.CreateRunner()
    .WithThreads(8);

var res = runner.WithPrompt(" This is the story of a man named ")
    .Infer(out _, nTokensToPredict = 50);
Console.Write(res);

model.Dispose();
```

## License 📜
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++.
- [sandrohanea/whisper.net](https://github.com/sandrohanea/whisper.net) as the reference on loading ggml models and libraries into .NET.