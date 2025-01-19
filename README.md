# important 7.01.2025
sorry guys for the confusion i could make with my reddit post about this repository. It has right now a lot of mistakes and now I am trying to fix this. 

# zig-pytorch
As a man who is trying to understand zig and write usefull code in other language than python (or try to write something usefull), I get an idea of creating something what is using zig to optize pytorch, make is faster, more reliable. I don't know if it's going to work, but I need to do it my own as a motivation do study. 
# I've seen this in my dream
# 0. Basic know-how
PyTorch is written in C++ and gives API in this language, BUT ZIG GIVES US built in tools to compile code in C/C++. I can define functions from C++ in zig and use it in my code. [000_example.zig](mm.zig). As you see, something works, and the plan is  rewrite/build from scrach pytorch functions in zig and do something with it. 
## What do I need
- Write pytorch functions in zig.
- Write cpp api extension with obv declaration  `extern "C" .. ` my Zig functions.
- Build real extension to pytorch with `torch.utilis.cpp_extension`. Create `.o` file.
- Check it everything work on python 

# 1. Creating basic functions from pytorch
## `torch.mm(input, mat2) -> Tensor`
This function takes two tensors 2d (matrix).




# 00. notes

zig build command
` zig build-obj -OReleaseFast -fPIC mm.zig`
`zig build-obj -fcompiler-rt mm.zig -fPIC  -lpthread`
important to create module 
`pip install -e .` // `python setup.py install`





but there is a progres:
```bash
                          (base) 
Generowanie macierzy...

Liczenie z zigtorch:
Starting multiplication: 10x10 * 10x10
Spawning thread 0: 0-5
thread 43064 panic: reached unreachable code
aborting due to recursive panic
fish: Job 1, 'python testmm.py' terminated by signal SIGABRT (Abort)
```
![image](https://github.com/user-attachments/assets/c5803144-53ac-427b-bf94-8dd3bcd84fe9)

my 6 core procesor is crying 
![image](https://github.com/user-attachments/assets/e1fd7747-1a49-495b-aa31-a92af5dc1ee6)

# Updated Progress 8.01.2025
The issues mentioned above have been fixed. The current implementation is still not faster than PyTorch, but significant improvements have been made. The `mm.zig` file has been optimized, and the multithreading issues have been resolved. The setup process has been streamlined, and the performance comparison with PyTorch is now more accurate.

# Next Steps
- Continue optimizing the `mm.zig` file to further improve performance.
- Explore additional techniques for optimizing matrix multiplication.
- Keep the community updated on progress and share any new findings.
