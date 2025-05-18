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




## Testing 18.05.2025
ZIG 0.15.Development
```bash

Przygotowywanie macierzy A (254x140) i B (140x54) typu float32...

Mnożenie macierzy przy użyciu biblioteki Zig...
Wywoływanie zig_mm z M=254, N=54, K=140
Mnożenie macierzy (254x140) x (140x54) przez Zig zajęło: 11.442 ms
[[33.951393 35.248943 34.04216  ... 33.04547  30.706268 31.629448]
 [30.861626 32.428852 33.789036 ... 30.069324 30.743837 30.006912]
 [34.62321  37.256306 36.88081  ... 36.6761   35.902122 32.418636]
 ...
 [33.08246  34.44827  35.824577 ... 31.893604 33.326576 31.194418]
 [33.301018 34.927197 37.236103 ... 32.154854 33.64552  32.54703 ]
 [37.280735 36.268288 38.55098  ... 35.220108 35.889587 33.731087]]

Mnożenie macierzy przy użyciu Numpy (dla weryfikacji)...
Mnożenie przez Numpy zajęło: 16.756 ms

Sprawdzanie poprawności wyniku...
Wyniki z Zig i Numpy są ZGODNE (w granicach tolerancji).

Fragment macierzy wynikowej C (pierwsze 5x5 lub mniej):
[[33.951393 35.248943 34.04216  32.671425 35.211803]
 [30.861626 32.428852 33.789036 26.134836 31.356468]
 [34.62321  37.256306 36.88081  33.398884 34.236626]
 [38.153404 39.542816 41.599964 35.657734 37.00282 ]
 [35.02774  37.746273 39.938614 31.795218 36.557247]]

Test FFI zakończony.

# TEST 2 
Ładowanie biblioteki z: /zig-torch/src/libmm.so
Przygotowywanie macierzy A (254x140) i B (140x54) typu float32...

Mnożenie macierzy przy użyciu biblioteki Zig...
Wywoływanie zig_mm z M=254, N=54, K=140
Mnożenie macierzy (254x140) x (140x54) przez Zig zajęło: 11.426 ms
[[37.28844  35.93189  37.627705 ... 35.715103 38.775482 35.44787 ]
 [38.677364 36.8895   37.710854 ... 33.269947 38.094646 35.526203]
 [38.677082 34.454033 38.12574  ... 34.723488 37.697567 35.672318]
 ...
 [34.749523 32.73221  36.878777 ... 31.674236 36.24429  33.483032]
 [36.10837  34.349216 37.285397 ... 32.74428  38.247486 35.067253]
 [35.338726 33.949272 32.48261  ... 32.66015  33.821045 33.340446]]

Mnożenie macierzy przy użyciu Numpy (dla weryfikacji)...
Mnożenie przez Numpy zajęło: 2.142 ms

Sprawdzanie poprawności wyniku...
Wyniki z Zig i Numpy są ZGODNE (w granicach tolerancji).

Fragment macierzy wynikowej C (pierwsze 5x5 lub mniej):
[[37.28844  35.93189  37.627705 35.706707 36.20624 ]
 [38.677364 36.8895   37.710854 35.05887  37.59802 ]
 [38.677082 34.454033 38.12574  36.012115 36.267544]
 [34.89467  31.197737 36.601784 31.63742  32.303074]
 [36.511215 33.19432  35.42715  35.11192  33.692394]]

Test FFI zakończony.
```
    ~/github/zig-torch/src  on   main !4                   
