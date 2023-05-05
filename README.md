# LLaMA.NET - Rewritten
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

.NET library to run [LLaMA](https://arxiv.org/abs/2302.13971) using [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). 

## Build 🧰
To build the library, you need to have [CMake](https://cmake.org/) and Python installed. Then, run the following commands at the root of the repository.

```bash

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

```csharp
public static class Program
{
    static Stopwatch sw = Stopwatch.StartNew();
    private static void Main()
    {
        var runner = new LLM("/models/ggml-alpaca-7b-native-q4.bin");

        runner.Instruction($"Write a C# script that displays the current date", "");
        foreach(var token in runner.InferenceStream(150)) 
            Console.Write(token);
        Console.WriteLine();

        runner.ClearContext();

        runner.Instruction($"Extract favorite food, Age, Name, City", "Hello my name is Trbl and I am 30 years old. I live in vienna austria. I like pizza. I hate fish. Spaghetti is good too");
        foreach(var token in runner.InferenceStream(150))
            Console.Write(token);

        runner.ClearContext();
       
        runner.Continuation($"A long time ago, in a galaxy far far away... ");
        foreach(var token in runner.InferenceStream(150)) 
            Console.Write(token);

        runner.ClearContext();

        var story = @"""We buried my brother with his dreams. On colored scraps of paper my young son, Teddy, and I scrawled all the fantasies Abe never achieved for lack of trying: hero, quarterback, singer, actor and more and crammed them in the satin folds of his coffin along with his favorite bottle of Jack and a pack of Camels. Teddy, a budding artist, sketched Abe throwing a football.
            “Can you imagine Uncle Abe throwing long on a cloud?” Teddy asked as he gingerly dropped in the drawing.
            “Might piss off the angels if he gets too rowdy,” I shrugged. “Same goes for showing off his bravery or acting like he’s better than all the other souls.”
            “Everyone sings in heaven. He can sing, huh?” Teddy pressed.
            “Not off-key. God has sensitive ears.”
            “So, Uncle Abe doesn’t get to live his dreams after all? That sucks,” Teddy gathered his crayons and paper, sat on the floor of the funeral parlor and begin drawing in earnest.
            “What the hell are you doing, Teddy?”
            Teddy put a finishing flourish on a portrait of himself painting.
            “Going for my dreams while I can just in case I run out of time and end up in heaven.”""";

        runner.Instruction($"What is this story about?", story);
        foreach(var token in runner.InferenceStream(150))
            Console.Write(token);
```

## License 📜
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏
- [hpretila/llama.net](https://github.com/hpretila/llama.net) The original author
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++.
- [sandrohanea/whisper.net](https://github.com/sandrohanea/whisper.net) as the reference on loading ggml models and libraries into .NET.