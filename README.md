# LLaMA.NET
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

.NET library to run [LLaMA](https://arxiv.org/abs/2302.13971) using [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). 

## Build 🧰
To build the library, you need to have [CMake](https://cmake.org/) and Python installed. Then, run the following commands at the root of the repository:

```bash
# Build and prepare the C++ library
python scripts/build_llama_cpp.py

# Build the .NET library
dotnet build LLaMA.NET/LLaMA.NET.csproj
```

The built library should be located at `LLaMA.NET/bin/Debug/netXXXX/LLaMA.NET.dll`.

Currently only Linux is supported. Work is being done to dynamically load the C++ library on other platforms.

## Usage 📖
```csharp
using LLaMA.NET;

LLaMAModel model = LLaMAModel.FromPath("/mnt/d/LLaMA/7B-LoRA/ggml-model-q4_0.bin");
LLaMARunner runner = model.CreateRunner()
    .WithThreads(8)
    .WithPrompt(" This is the story of a man named ");

var res = runner.Infer(out _, nTokensToPredict = 50);
Console.Write(res);

model.Dispose();
```

## License 📜
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.