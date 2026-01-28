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




## Testing 19.05.2025
ZIG 0.15.Development
```bash
Ładowanie biblioteki z: /home/kitajussus/github/zig-torch/src/libmm.so
Creating Matrix A (50x50) and B (50x50) type float32...

 ZIG MULTIPLYING : !!!!!!!!!!!!!!!!!
Multiply with zig (50x50) x (50x50) took : 0.713 ms

 Multiply with Numpy (for comparison):
Time for numpy: 2.430 ms

Checking...
ZIG IS NOT MULTYPLYING ONLY ZEROES!!!! IT IS WORKING

 Zig matrix  ( first 5x5 OR LESS):
[[13.51593   10.569702  13.364006  12.218889   9.840366 ]
 [12.826249   9.629739  14.659329  10.829753  10.374571 ]
 [13.49235   10.976502  14.516691  12.262626  11.237597 ]
 [11.469961   9.874568  11.2643795 10.609412  10.377689 ]
 [12.426827  10.05384   12.000959  11.205995   9.461213 ]]
Numpy Matrix ( first 5x5 OR LESS):
[[13.51593   10.569702  13.364005  12.218888   9.840365 ]
 [12.826249   9.629739  14.659329  10.829753  10.374571 ]
 [13.49235   10.976501  14.516692  12.262625  11.2375965]
 [11.469961   9.874568  11.2643795 10.609412  10.377689 ]
 [12.426826  10.053841  12.000959  11.205995   9.461213 ]]

Test FFI zakończony.
```

##  26.01.2026
I've changed my mind, now everytime i will update project to the newest version.
### zig loses agains numpy with torch
> [!IMPORTANT]
> with torch


```bash
Size M×K × K×N         Torch (ms)   NumPy (ms)   Zig (ms)     Zig vs Torch   Zig vs NumPy   Correct
---------------------------------------------------------------------------------------------------
32×32 × 32×32            0.003        0.003        0.017          0.21x          0.18x True
64×64 × 64×64            0.008        0.016        0.085          0.09x          0.19x True
128×128 × 128×128           6.511        0.627        0.462         14.09x          1.36x True
256×256 × 256×256           7.226        0.111        2.773          2.61x          0.04x True
512×512 × 512×512           1.516        0.630       22.216          0.07x          0.03x True
1024×1024 × 1024×1024          7.648        4.324      187.883          0.04x          0.02x False
1024×512 × 512×256           1.249        0.549       21.713          0.06x          0.03x True
```

### zig wins against solo numpy
>[!IMPORTANT]
>without TORCH, ONLY NUMPY
```bash
python tests/benchmark.py
Size M×K × K×N         Torch (ms)   NumPy (ms)   Zig (ms)     Zig vs Torch   Zig vs NumPy   Correct
---------------------------------------------------------------------------------------------------
32×32 × 32×32              n/a        0.019        0.018            n/a          1.04x True
64×64 × 64×64              n/a        0.143        0.055            n/a          2.62x True
128×128 × 128×128             n/a        1.178        0.358            n/a          3.29x True
256×256 × 256×256             n/a        9.086        2.778            n/a          3.27x True
512×512 × 512×512             n/a       69.996       23.019            n/a          3.04x True
1024×1024 × 1024×1024            n/a      553.027      196.582            n/a          2.81x True
1024×512 × 512×256             n/a       72.071       22.113            n/a          3.26x True

```

# Roadmap Checklist

## Phase 1: Foundation & Correctness

### [ ] Fix Boundary Conditions
**Inspirational Links & References:**
- [BLIS micro-kernel handling](https://github.com/flame/blis/blob/master/kernels/zen/3/bli_gemm_zen_asm_d6x8.c#L200-L250)
- [Eigen block size logic](https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/products/GeneralBlockPanelKernel.h#L1850)

**Notes:**
<!-- skibidi -->
---

### [ ] Integer Overflow Protection
**Inspirational Links & References:**
- [Rust overflow handling](https://doc.rust-lang.org/std/primitive.usize.html#method.checked_mul)
- [BLIS dimension validation](https://github.com/flame/blis/blob/master/frame/3/bli_l3_check.c)

**Notes:**
<!-- skibidi -->
---

### [ ] Thread-Safe Global Initialization
**Inspirational Links & References:**
- [Zig thread-local storage](https://ziglang.org/documentation/master/#threadlocal)
- [NumPy thread-safe config](https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/multiarraymodule.c#L4500)

**Notes:**
<!-- skibidi -->
---

### [ ] Unit Test Framework
**Inspirational Links & References:**
- [Eigen test suite](https://gitlab.com/libeigen/eigen/-/tree/master/test)
- [NumPy test patterns](https://github.com/numpy/numpy/blob/main/numpy/core/tests/test_multiarray.py)
- [BLIS test framework](https://github.com/flame/blis/tree/master/testsuite)

**Notes:**
<!-- skibidi -->
---

### [ ] Benchmark Comparison Framework
**Inspirational Links & References:**
- [NumPy benchmark suite](https://github.com/numpy/numpy/tree/main/benchmarks)
- [PyTorch benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks)
- [Google Benchmark library](https://github.com/google/benchmark)

**Notes:**
<!-- skibidi -->
---

### [ ] Input Validation Layer
**Inspirational Links & References:**
- [Zig error handling](https://ziglang.org/documentation/master/#Errors)
- [Rust validation patterns](https://doc.rust-lang.org/book/ch09-00-error-handling.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Memory Alignment Utilities
**Inspirational Links & References:**
- [C++ std::aligned_alloc](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)
- [NumPy alignment](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Inline Documentation
**Inspirational Links & References:**
- [Zig doc comments](https://ziglang.org/documentation/master/#Doc-Comments)
- [Doxygen style guide](https://www.doxygen.nl/manual/docblocks.html)
- [Rust documentation guide](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html)

**Notes:**
<!-- skibidi -->
---

### [ ] API Documentation Website
**Inspirational Links & References:**
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Rust book structure](https://doc.rust-lang.org/book/)
- [NumPy docs](https://numpy.org/doc/stable/)

**Notes:**
<!-- skibidi -->
---

## Phase 2: Performance Engineering

### [ ] Dynamic Block Size Selection
**Inspirational Links & References:**
- [Anatomy of High-Performance GEMM (Goto & van de Geijn)](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)
- [BLIS auto-tuning](https://github.com/flame/blis/wiki/Autotuning)
- [Halide autoscheduler](https://halide-lang.org/papers/autoscheduler2019.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Empirical Auto-Tuner
**Inspirational Links & References:**
- [OpenTuner framework](https://github.com/jansel/opentuner)
- [ATLAS auto-tuning](http://math-atlas.sourceforge.net/)
- [FFTW planner](http://www.fftw.org/fftw3_doc/Planner-Flags.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Transpose
**Inspirational Links & References:**
- [Cache-oblivious algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
- [Efficient matrix transpose (CS:APP)](http://csapp.cs.cmu.edu/3e/labs.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Element-wise Operations
**Inspirational Links & References:**
- [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html)
- [SLEEF vectorized math](https://sleef.org/)

**Notes:**
<!-- skibidi -->
---

### [ ] Reduction Operations
**Inspirational Links & References:**
- [CUB reductions](https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html)
- [Thrust reductions](https://thrust.github.io/doc/group__reductions.html)

**Notes:**
<!-- skibidi -->
---

### [ ] Built-in Profiler
**Inspirational Links & References:**
- [Chrome Trace Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [Tracy Profiler](https://github.com/wolfpld/tracy)
- [Perfetto](https://perfetto.dev/)

**Notes:**
<!-- skibidi -->
---

## Phase 3: API & Usability

### [ ] Core Tensor Implementation
**Inspirational Links & References:**
- [PyTorch Tensor API](https://pytorch.org/docs/stable/tensors.html)
- [NumPy ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html)
- [JAX DeviceArray](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.DeviceArray.html)

**Notes:**
<!-- skibidi -->

---

### [ ] PyTorch Integration
**Inspirational Links & References:**
- [PyTorch custom functions](https://pytorch.org/docs/stable/notes/extending.html)
- [JAX pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)

**Notes:**
<!-- skibidi -->

---

### [ ] Example Gallery
**Inspirational Links & References:**
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [Scikit-learn examples](https://scikit-learn.org/stable/auto_examples/)
- [JAX tutorials](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

**Notes:**
<!-- skibidi -->

---

## Phase 4: Advanced Features

### [ ] FP16 Support
**Inspirational Links & References:**
- [NVIDIA Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [ARM FP16 instructions](https://developer.arm.com/documentation/dui0801/latest/A64-Floating-point-Instructions)
- [Intel AVX-512 FP16](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-fp16-instruction-set.html)

**Notes:**
<!-- skibidi -->

---

### [ ] INT8 Quantization
**Inspirational Links & References:**
- [TensorFlow Lite quantization](https://www.tensorflow.org/lite/performance/quantization_spec)
- [PyTorch quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/quantization.html)

**Notes:**
<!-- skibidi -->

---
>>>>>>> 803966edaf013b7844c16741d8fb03c5c26703ac
