<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/images/logo.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/images/logo.png">
  <img src="/.github/images/logo.png" width="100%" alt="Fast and Fourier" >
</picture>	
</p>

A high-performance Digital Signal Processing library with JIT compilation and multi-architecture SIMD support.

FastAndFourier provides a complete set of discrete transforms with an Intermediate Representation (IR) that enables portable, high-performance execution across x86_64, AArch64, and CUDA platforms. The library features a compiler-based JIT system that generates optimized native code at runtime.

## Features

- **JIT Compilation**: Runtime code generation using system compiler (GCC/Clang) for maximum performance
- **Multi-Architecture SIMD**: Automatic vectorization for SSE4.2, AVX2, AVX-512, NEON, and SVE
- **Chirp DSL**: Scheme-like language with Smalltalk-style keywords for DSP pipelines
- **Unified IR**: Single bytecode format powers all transform types
- **Multi-Precision**: FP8, FP16, BF16, FP32, and FP64 support
- **Direct-Threaded VM**: High-performance interpreter with computed goto dispatch
- **Extensible**: Easy to add new transforms via the IR opcode system

## Supported Transforms

| Transform | Forward | Inverse | Notes |
|-----------|---------|---------|-------|
| FFT | Yes | Yes | Radix-2/4 Cooley-Tukey |
| DCT | Yes | Yes | Types I, II, III, IV |
| DST | Yes | Yes | Types I, II, III, IV |
| STFT | Yes | Yes | Overlapping windows |
| MDCT | Yes | Yes | Used in AAC, Opus |
| Haar Wavelet | Yes | Yes | Lifting scheme |
| Daubechies-4 | Yes | Yes | D4 wavelet |
| CDF 9/7 | Yes | Yes | JPEG 2000 wavelet |

## Quick Start

```bash
# Clone and build
git clone https://github.com/yourusername/fastandfourier.git
cd fastandfourier
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
make -j$(nproc)

# Run tests
make test

# Run benchmarks
./benchmarks/fastandfourier_benchmarks
```

## Example

### Basic C API

```c
#include <fastandfourier.h>
#include <math.h>
#include <stdlib.h>

int main() {
    dspir_init();
    
    // Create 1024-point FFT
    dspir_transform* fft = dspir_create_fft(1024, false, DSPIR_PREC_FP32, 0);
    
    float* in = aligned_alloc(64, 2 * 1024 * sizeof(float));
    float* out = aligned_alloc(64, 2 * 1024 * sizeof(float));
    
    // Generate complex sine wave
    for (int i = 0; i < 1024; i++) {
        in[2*i] = sinf(2.0f * M_PI * 4.0f * i / 1024.0f);
        in[2*i + 1] = 0.0f;
    }
    
    // Execute using JIT-compiled kernel (default)
    dspir_execute_jit(fft, out, in);
    
    // Cleanup
    free(in); free(out);
    dspir_destroy_transform(fft);
    dspir_cleanup();
    
    return 0;
}
```

### Chirp DSL

Chirp is a Scheme-like DSL with Smalltalk-style keyword arguments for describing DSP pipelines:

```c
#include <chirp.h>
#include <fastandfourier.h>

// Register custom functions (like CUDA kernels or OpenCL devices)
chirp_register("gaussian", (void(*)(void))my_gaussian_impl);
chirp_register("softmax", (void(*)(void))my_softmax_impl);

// Compile a Chirp program
dspir_transform *t = chirp_compile(
    "(pipeline "
    "  (fft :size 1024) "
    "  twiddle "
    "  (bfly 4) "
    "  (lift :predict gaussian :update softmax) "
    "  (custom softmax) "
    "  reduce-sum)"
);

// Execute the compiled transform
dspir_execute_jit(t, out, in);
dspir_destroy_transform(t);
```

**Chirp Syntax:**
- `(fft :size N)` - FFT with Smalltalk-style keyword argument
- `(bfly N)` - Radix-N butterfly (2, 4, or 8)
- `(lift :predict fn :update fn)` - Lifting scheme wavelet
- `(custom name)` - Call registered built-in function
- `reduce-sum`, `reduce-max`, `reduce-min` - Reduction operations
- `(pipeline ...)` - Sequence multiple operations

Compile with:
```bash
gcc -O3 -o fft_example example.c -lfastandfourier -lm -ldl
```

## API Reference

### Transform Creation

| Function | Description |
|----------|-------------|
| `dspir_create_fft(n, inverse, precision, flags)` | Create FFT transform |
| `dspir_create_dct(n, type, precision, flags)` | Create DCT transform (type 1-4) |
| `dspir_create_dst(n, type, precision, flags)` | Create DST transform (type 1-4) |
| `dspir_create_haar(n, levels, precision, flags)` | Create Haar wavelet |
| `dspir_create_mdct(n, precision, flags)` | Create MDCT transform |
| `dspir_destroy_transform(t)` | Free transform and associated memory |

### Execution Functions

| Function | Description |
|----------|-------------|
| `dspir_execute(t, out, in)` | Auto-select best backend |
| `dspir_execute_f32(t, out, in)` | FP32 execution |
| `dspir_execute_f64(t, out, in)` | FP64 execution |
| `dspir_execute_jit(t, out, in)` | JIT-compiled execution |
| `dspir_execute_jit_ex(t, out, in, flags)` | JIT with extended options |

### JIT Compilation

| Function | Description |
|----------|-------------|
| `dspir_jit_create()` | Create JIT context |
| `dspir_jit_compile(ctx, t)` | Compile transform to native code |
| `dspir_jit_get_kernel(ctx)` | Get compiled kernel function |
| `dspir_jit_destroy(ctx)` | Free JIT context |

### Chirp DSL API

Include `<chirp.h>` to use the Chirp DSL.

| Function | Description |
|----------|-------------|
| `chirp_register(name, fn)` | Register a custom C function |
| `chirp_compile(source)` | Compile Chirp S-expression to transform |

### JIT Flags

| Flag | Description |
|------|-------------|
| `DSPIR_FLAG_JIT_SIMD` | Enable SIMD intrinsics (default on supported platforms) |
| `DSPIR_FLAG_JIT_INPLACE` | Work in-place (overwrites input) |
| `DSPIR_FLAG_JIT_SPLIT_PLANE` | Use split-plane mode (default, faster for most sizes) |

### Precision Levels

| Enum | Description |
|------|-------------|
| `DSPIR_PREC_FP8` | 8-bit floating point (E4M3/E5M2) |
| `DSPIR_PREC_FP16` | 16-bit IEEE 754 half precision |
| `DSPIR_PREC_BF16` | 16-bit brain floating point |
| `DSPIR_PREC_FP32` | 32-bit single precision |
| `DSPIR_PREC_FP64` | 64-bit double precision |

### Data Layout

- Complex data is stored as interleaved real/imaginary pairs (`float[2*n]`)
- All buffers must be 64-byte aligned for SIMD performance
- Transform size N must be a power of 2 for FFT
- **Split-plane mode** (default for FP32/FP64): Internally deinterleaves to separate real/imaginary arrays for optimal vectorization

## Building

### Basic Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with Tests and Benchmarks

```bash
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DBUILD_EXAMPLES=ON
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | OFF | Build test suite (requires Google Test) |
| `BUILD_BENCHMARKS` | OFF | Build benchmarks (requires Google Benchmark) |
| `BUILD_EXAMPLES` | OFF | Build example programs |
| `ENABLE_CUDA` | OFF | Enable CUDA support |
| `ENABLE_COVERAGE` | OFF | Enable code coverage reporting |

### Compiler Flags

The library automatically detects and uses the best available SIMD instruction set. For optimal performance:

| Platform | Recommended Flags |
|----------|-------------------|
| x86_64 | `-march=native -O3` |
| AArch64 | `-march=armv8.2-a+fp16+simd -O3` |

## Benchmark Results

ARM Cortex-A78 (6 cores @ 1728 MHz), GCC 11.4.0, -O3 -march=armv8.2-a+fp16+simd+crypto

### JIT vs VM Performance (Split-Plane)

Split-plane mode provides significantly better vectorization by storing real and imaginary parts in separate arrays.

| Transform | Size | VM (us) | JIT (us) | Speedup |
|-----------|------|---------|----------|---------|
| FFT | 16 | 0.985 | 0.079 | **12.5x** |
| FFT | 32 | - | 0.235 | - |
| FFT | 64 | 2.82 | 0.735 | **3.8x** |
| FFT | 128 | 5.44 | - | - |
| DCT-II | 16 | 0.970 | - | - |
| DCT-II | 64 | 2.80 | - | - |
| DCT-II | 128 | 5.42 | - | - |
| Haar | 16 | 0.791 | - | - |
| Haar | 64 | 1.44 | - | - |
| Haar | 128 | 2.17 | - | - |
| MDCT | 16 | 0.968 | - | - |
| MDCT | 64 | 2.80 | - | - |
| MDCT | 128 | 5.42 | - | - |

### VM Execution Throughput (Interleaved)

| Transform | Size | Throughput (M/s) | Performance |
|-----------|------|------------------|-------------|
| FFT (FP32) | 16 | 16.24 | ~33 GFLOPS |
| FFT (FP32) | 64 | 22.71 | ~69 GFLOPS |
| FFT (FP32) | 128 | 23.54 | ~85 GFLOPS |
| DCT-II | 16 | 16.52 | ~33 GFLOPS |
| DCT-II | 64 | 22.85 | ~69 GFLOPS |
| DST-II | 64 | 22.84 | ~69 GFLOPS |
| Haar | 16 | 20.25 | ~41 GFLOPS |
| Haar | 64 | 44.41 | ~135 GFLOPS |
| Haar | 1024 | 78.12 | ~281 GFLOPS |
| MDCT | 64 | 22.87 | ~69 GFLOPS |

### JIT Throughput by Transform Size

| Transform | Size | Throughput | Bandwidth |
|-----------|------|------------|-----------|
| FFT (FP32) | 16 | **203 M FFTs/sec** | 1.51 GiB/s |
| FFT (FP32) | 32 | **136 M FFTs/sec** | 1.02 GiB/s |
| FFT (FP32) | 64 | **87 M FFTs/sec** | 665 MiB/s |

### Scaling Performance (N log N complexity)

| Transform | N | Time (us) | GFLOPS |
|-----------|---|-----------|--------|
| FFT | 16 | 0.98 | 0.33 |
| FFT | 64 | 2.80 | 0.69 |
| FFT | 256 | 12.0 | 0.85 |
| FFT | 1024 | 52.9 | 0.97 |
| FFT | 4096 | 247 | 1.00 |
| FFT | 8192 | 316 | 1.69 |

*Complexity verified: O(N log N) with 41% RMS deviation*

### Library Comparison (Release Build)

Throughput comparison with popular FFT libraries on ARM Cortex-A78 (all using complex-to-complex):

| Size | FAF (VM) | FAF (JIT) | KissFFT | NotoriousFFT | FFTW3 |
|------|----------|-----------|---------|--------------|-------|
| 16 | 16.1 M/s | **203 M/s** | 122.6 M/s | 197.0 M/s | **275 M/s** |
| 64 | 23.1 M/s | **87 M/s** | 83.1 M/s | 130.0 M/s | **238 M/s** |
| 256 | 21.5 M/s | - | 62.6 M/s | 94.7 M/s | **206 M/s** |
| 1024 | 19.4 M/s | - | 51.7 M/s | 74.2 M/s | **175 M/s** |
| 4096 | 16.6 M/s | - | 43.3 M/s | 60.0 M/s | **155 M/s** |

**Notes:**
- All benchmarks measure **in-cache performance** (repeated transforms on same buffers)
- FFTW3's lead comes from highly optimized assembly kernels with better scaling
- **JIT scaling**: Our JIT drops from 202 M/s (16-point) to 60 M/s (128-point) - 70% decrease
- **FFTW3 scaling**: FFTW3 drops from 275 M/s (16-point) to 206 M/s (256-point) - only 25% decrease
- JIT compilation fails above 128 points due to excessive compile times
- Real-world throughput streaming from memory will be bandwidth-limited (~10-20 GB/s)

## Architecture Support

| Platform | SSE4.2 | AVX2 | AVX-512 | NEON | SVE | CUDA |
|----------|--------|------|---------|------|-----|------|
| x86_64 | Yes | Yes | Yes | No | No | Yes |
| AArch64 | No | No | No | Yes | Yes | No |

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Chirp DSL](CHIRP.md) - Scheme-like DSL for DSP pipelines with 140+ builtins
- [Performance Analysis](ANALYSIS.md) - Detailed benchmark analysis
- [Architecture Overview](docs/ARCHITECTURE.md) - IR and VM design
- [Building Guide](docs/BUILDING.md) - Advanced build options

## Algorithm

FastAndFourier uses a flexible Intermediate Representation (IR) with 31 opcodes that can express multiple transform types. The JIT compiler generates optimized C code from this IR, which is then compiled to native machine code using the system compiler.

For FFT, the library uses radix-2 and radix-4 Cooley-Tukey algorithms with bit-reversal permutation. DCT and DST are computed via real FFT with pre/post-processing twiddles. Wavelet transforms use the lifting scheme for in-place computation.

All SIMD kernels use architecture-specific intrinsics (NEON on ARM, AVX/AVX-512 on x86) for butterfly operations and twiddle factor multiplication.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- FFT algorithms based on Cooley-Tukey
- Wavelet transforms using lifting scheme
- JIT approach inspired by GNU Lightning and LLVM ORC
- Benchmark comparisons use [Kiss FFT](https://github.com/mborgerding/kissfft), [Notorious FFT](https://github.com/NotoriousLOB/Notorious-FFT), and [FFTW3](http://www.fftw.org/)
- Benchmark comparisons use Kiss FFT, PocketFFT, and minFFT
