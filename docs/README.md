# FastAndFourier

A high-performance Digital Signal Processing (DSP) library featuring Intermediate Representation (IR) for transform description, Just-In-Time (JIT) compilation, and multi-architecture vectorization.

## Features

- **Intermediate Representation (IR)**: Transform-agnostic bytecode for describing DSP operations
- **Direct-Threaded VM**: High-performance interpreter using computed goto for minimal dispatch overhead
- **JIT Compilation**: Compile transforms to native code for maximum performance
- **Multi-Architecture Support**:
  - x86_64: SSE4.2, AVX2, AVX-512
  - AArch64: NEON, SVE (Scalable Vector Extension)
  - CUDA: GPU acceleration
- **Multiple Precision Levels**: FP8, FP16, BF16, FP32, FP64
- **Comprehensive Transform Support**:
  - FFT (Fast Fourier Transform) - forward and inverse
  - DCT (Discrete Cosine Transform) - Types I-IV
  - DST (Discrete Sine Transform) - Types I-IV
  - STFT (Short-Time Fourier Transform)
  - MDCT (Modified Discrete Cosine Transform)
  - Wavelet Transforms (Haar, Daubechies-4, CDF 5/3, CDF 9/7, Symlet-4) — forward and inverse

## Building

### Requirements

- CMake 3.18+
- C11/C++17 compiler (GCC, Clang, MSVC)
- Optional: CUDA Toolkit 11.0+
- Optional: Doxygen for documentation

### Basic Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with All Options

```bash
cmake .. \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_EXAMPLES=ON \
    -DENABLE_COVERAGE=ON \
    -DENABLE_CUDA=ON
make -j$(nproc)
```

### Running Tests

```bash
make test
# or
ctest --output-on-failure
```

### Running Benchmarks

```bash
./benchmarks/fastandfourier_benchmarks
```

## Usage

### Basic FFT Example

```c
#include <fastandfourier.h>
#include <stdlib.h>
#include <math.h>

int main() {
    /* Initialize library */
    faf_init();
    
    /* Create a 1024-point FFT */
    size_t n = 1024;
    faf_transform* fft = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    
    /* Allocate aligned buffers */
    float* in = aligned_alloc(64, 2 * n * sizeof(float));
    float* out = aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Interleaved complex: real/imag pairs */
    for (size_t i = 0; i < n; i++) {
        in[2 * i]     = sinf(2.0f * M_PI * 4.0f * i / n);
        in[2 * i + 1] = 0.0f;
    }
    
    /* Execute FFT */
    faf_execute_f32(fft, out, in);
    
    /* Clean up */
    free(in);
    free(out);
    faf_destroy_transform(fft);
    faf_cleanup();
    
    return 0;
}
```

### Using JIT Compilation

```c
#include <fastandfourier.h>

int main() {
    faf_init();
    
    /* Create transform */
    faf_transform* fft = faf_create_fft(1024, false, FAF_PREC_FP32, 0);
    
    /* Create JIT context and compile */
    faf_jit_ctx* jit = faf_jit_create();
    faf_jit_compile(jit, fft);
    
    /* Get compiled kernel */
    faf_kernel_fn kernel = faf_jit_get_kernel(jit);
    
    /* Execute compiled kernel */
    float in[1024], out[1024];
    /* ... initialize input ... */
    kernel(out, in, 1024, fft->twiddles[0]);
    
    /* Clean up */
    faf_jit_destroy(jit);
    faf_destroy_transform(fft);
    faf_cleanup();
    
    return 0;
}
```

## Architecture

### IR Design

The library uses a custom Intermediate Representation (IR) for describing DSP transforms:

- **Opcodes**: 31 different operations covering arithmetic, memory, and control flow
- **Registers**: 256 virtual registers
- **Twiddles**: Pre-computed trigonometric tables
- **Instructions**: Packed 16-byte format for efficient execution

### Execution Paths

1. **VM (Interpreter)**: Direct-threaded dispatch for portable execution
2. **SIMD Kernels**: Hand-optimized vectorized implementations
3. **JIT**: Compile to native machine code
4. **CUDA**: GPU acceleration for large transforms

## Performance

Typical performance on modern hardware:

| Transform | Size | Throughput | Latency |
|-----------|------|------------|---------|
| FFT (FP32) | 1K | ~10 GSamples/s | ~5 µs |
| FFT (FP32) | 4K | ~15 GSamples/s | ~25 µs |
| FFT (FP64) | 1K | ~5 GSamples/s | ~10 µs |
| DCT-II | 1K | ~8 GSamples/s | ~6 µs |
| Haar | 1K | ~12 GSamples/s | ~4 µs |

*Results measured on Intel Core i9-12900K with AVX-512*

## API Reference

See the [full API documentation](https://fastandfourier.readthedocs.io) for detailed information.

### Core Functions

- `faf_init()` / `faf_cleanup()` - Library initialization
- `faf_create_fft()` / `faf_create_dct()` / `faf_create_dst()` - Transform creation
- `faf_execute_f32()` / `faf_execute_f64()` - Execute transforms
- `faf_destroy_transform()` - Cleanup

### JIT Functions

- `faf_jit_create()` - Create JIT context
- `faf_jit_compile()` - Compile transform
- `faf_jit_get_kernel()` - Get compiled function
- `faf_jit_destroy()` - Cleanup

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- FFT algorithms based on Cooley-Tukey radix-2/4
- Wavelet implementations using lifting scheme
- Inspired by FFTW, Intel IPP, and Apple vDSP
