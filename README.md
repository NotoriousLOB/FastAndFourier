# FastAndFourier

A high-performance Digital Signal Processing library with JIT compilation and multi-architecture SIMD support.

FastAndFourier provides a complete set of discrete transforms with an Intermediate Representation (IR) that enables portable, high-performance execution across x86_64, AArch64, and CUDA platforms. The library features a compiler-based JIT system that generates optimized native code at runtime.

## Features

- **JIT Compilation**: Runtime code generation using system compiler (GCC/Clang) for maximum performance
- **Multi-Architecture SIMD**: Automatic vectorization for SSE4.2, AVX2, AVX-512, NEON, and SVE
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

### JIT Flags

| Flag | Description |
|------|-------------|
| `DSPIR_FLAG_JIT_SIMD` | Enable SIMD intrinsics (default on supported platforms) |
| `DSPIR_FLAG_JIT_INPLACE` | Work in-place (overwrites input) |

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

ARM Cortex-A78 (6 cores @ 1728 MHz), GCC 11.4.0, -O3 -march=armv8.2-a+crypto+fp16+rcpc

### JIT vs VM Performance

| Transform | Size | VM (us) | JIT (us) | Speedup |
|-----------|------|---------|----------|---------|
| FFT | 16 | 20.1 | 0.087 | 231x |
| FFT | 64 | 21.6 | 0.675 | 32x |
| FFT | 256 | 24.5 | 3.23 | 8x |
| DCT-II | 64 | 22.3 | 0.71 | 31x |
| Haar | 64 | 8.5 | 0.45 | 19x |

### Throughput (JIT with SIMD)

| Transform | Size | Throughput |
|-----------|------|------------|
| FFT (FP32) | 16 | 183 M FFTs/sec |
| FFT (FP32) | 64 | 95 M FFTs/sec |
| FFT (FP32) | 256 | 31 M FFTs/sec |

## Architecture Support

| Platform | SSE4.2 | AVX2 | AVX-512 | NEON | SVE | CUDA |
|----------|--------|------|---------|------|-----|------|
| x86_64 | Yes | Yes | Yes | No | No | Yes |
| AArch64 | No | No | No | Yes | Yes | No |

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
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
- Benchmark comparisons use Kiss FFT, PocketFFT, and minFFT
