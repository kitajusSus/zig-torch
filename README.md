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

important to create module
`pip install -e .`

