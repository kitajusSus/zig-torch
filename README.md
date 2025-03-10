# important 27.02.2025
Ive changed few things, now I preffere to do firstyly something in zig and later on I will look for options to make bindings to python. 

mm.zig is main file with matrix multiplication function. the main problem is that i dont know how to build this, to make it work as a independent library in python, I lost an idea to make "faster pytorch", now im focused to something that can be used with or without pytorch. 


# zig-pytorch
As a man who is trying to understand zig and write usefull code in other language than python (or try to write something usefull), I get an idea of creating something what is using zig to optize pytorch, make is faster, more reliable. I don't know if it's going to work, but I need to do it my own as a motivation do study. 
# I've seen this in my dream
# 0. Basic know-how
PyTorch is written in C++ and gives API in this language, BUT ZIG GIVES US built in tools to compile code in C/C++. I can define functions from C++ in zig and use it in my code. [000_example.zig](mm.zig). As you see, something works, and the plan is  rewrite/build from scrach pytorch functions in zig and do something with it. 
## What do I need
- Write pytorch functions in zig.
- Check it everything work on python 

# 1. Creating basic functions from pytorch
## `torch.mm(input, mat2) -> Tensor`
This function takes two tensors 2d (matrix).









## Development Workflow

1. **Plan**: Define the functionality you want to implement
2. **Test-Driven Development**: Write tests in Zig first
3. **Implement**: Create the Zig implementation
4. **Benchmark**: Compare performance with PyTorch
5. **C API**: Expose functionality through C API
6. **Python Bindings**: Create Python wrappers
7. **Document**: Update documentation

## Building the Project

### Building the Zig Library

```bash
# Build the Zig library
zig build

# Run Zig tests
zig build test

# Build in release mode for better performance
zig build -Drelease-fast
```
# 00. notes

**zig build command**
- ` zig build-obj -OReleaseFast -fPIC mm.zig`
- `zig build-obj -fcompiler-rt mm.zig -fPIC  -lpthread`

**important to create module**
- `pip install -e .` // `python setup.py install`


**Building the Python Package**
```bash
# Install in development mode
pip install -e .

# Build and install
python setup.py install
```


# Adding New functions


ex: Creating a New Operation

- create a new file in src/ (e.g., src/add.zig)  [example add.zig](src/add.zig)
- Implement your function
- Create a test file in tests/ (e.g., tests/testadd.zig)
- Update build.zig to include your new files
- Expose through C API in src/native.zig

```bash
# Basic build
zig build

# Build with optimization for release
zig build -Drelease-fast

# Run tests
zig build test

# Clean build artifacts
rm -rf zig-out/
```
