<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/images/logo.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/images/logo.png">
  <img src="/.github/images/logo.png" width="100%" alt="Fast and Fourier" >
</picture>
</p>

## FastAndFourier is a JIT-compiled signal processing library with its own scripting language

Think an FFT engine in a library-sized package, with a Scheme-like language for composing DSP pipelines — and a compiler that emits AVX2 butterfly kernels at runtime.

```scheme
(pipeline
  (fft :size 1024)
  twiddle
  (bfly 4)
  (lift :predict gaussian :update softmax)
  reduce-sum)
```

That's [Chirp](#chirp-dsl). It compiles directly to a `faf_transform *` you execute with `faf_execute_jit`. No boilerplate, no bytecode you have to hand-assemble — just a pipeline description and a pointer.

- **FastAndFourier speaks Chirp.** A Scheme-like DSL with Smalltalk-style keyword arguments sits at the heart of the library. Describe a DSP pipeline in a single expression; get back a fully JIT-compiled transform ready to run at SIMD speed. Register your own C functions — CUDA kernels, custom filters, anything — and call them from the same script.

- **FastAndFourier is fast.** A JIT compiler turns the internal IR into optimized C, hands it to the system compiler, and links the result back in at runtime. At large transform sizes the AVX2 path overtakes KissFFT by **3.2×**. It is fast, but not magic — there is a ~1 µs VM floor for small N, and FFTW3 with full planning wins on a benchmark. We are honest about this.

- **FastAndFourier knows its transforms.** FFT, DCT (types I–IV), DST (types I–IV), STFT, MDCT, Haar, Daubechies-4, CDF 9/7 — all forward, all inverse, all driven by the same 31-opcode IR. Adding a new transform means adding opcodes, not rewriting the engine.

- **FastAndFourier goes everywhere.** SSE4.2, AVX2, AVX-512, NEON, SVE, and CUDA — SIMD dispatch is automatic. Write your pipeline in Chirp once; the right kernel fires on whatever hardware shows up.

- **FastAndFourier is precise.** FP8, FP16, BF16, FP32, and FP64 are first-class citizens. The same pipeline description works across all of them.

- **FastAndFourier is a library.** It compiles as C99, has no mandatory runtime dependencies, exposes a clean C API, and links as a static archive. Embed it in an audio codec, a radar preprocessor, or a weekend project — the footprint stays small.

---

## Chirp DSL

Chirp is the scripting language built into FastAndFourier. Include `<chirp.h>` and you get a compiler that turns S-expressions into native transform objects. The syntax is Scheme-flavored, the keyword arguments are Smalltalk-style, and the output is a pointer you hand directly to `faf_execute_jit`.

Hello world lives in [`examples/que_onda_mundo.c`](examples/que_onda_mundo.c) — yes, that is the canonical name.

```c
#include <chirp.h>
#include <fastandfourier.h>

chirp_register_standard_builtins();

/* Denoise with a real wavelet family */
faf_transform *dwt = chirp_compile(
    "(pipeline "
    "  (dwt :family cdf97 :size 1024 :levels 5)"
    "  (threshold :mode soft :lambda 0.08)"
    "  (idwt :family cdf97 :size 1024 :levels 5))"
);
faf_execute_f32(dwt, out, in);
faf_destroy_transform(dwt);
```

**Chirp syntax at a glance:**

| Expression | What it does |
|---|---|
| `(fft :size N)` | Full staged FFT of size N |
| `(dwt :family haar :size N :levels L)` | Forward DWT (haar, d4, cdf53, cdf97, sym4) |
| `(idwt :family … :size N :levels L)` | Inverse DWT |
| `(threshold :mode soft :lambda λ)` | Soft/hard threshold of detail bands |
| `(lift :predict haar :update haar)` | Named-family lifting → real `DWT_STAGE` |
| `(custom name)` | Call a registered C function |
| `reduce-sum` / `reduce-max` / `reduce-min` | Reduction operations |
| `(pipeline ...)` | Sequence multiple operations |

The full language reference — including 140+ built-in operations — lives in [CHIRP.md](CHIRP.md).

---

## Supported Transforms

| Transform | Forward | Inverse | Notes |
|-----------|---------|---------|-------|
| FFT | Yes | Yes | Radix-2/4 Cooley-Tukey |
| DCT | Yes | Yes | Types I, II, III, IV |
| DST | Yes | Yes | Types I, II, III, IV |
| STFT | Yes | Yes | Overlapping windows |
| MDCT | Yes | Yes | Used in AAC, Opus |
| Haar Wavelet | Yes | Yes | Orthonormal lifting / Mallat |
| Daubechies-4 (D4 / db2) | Yes | Yes | 4-tap orthonormal filter bank |
| CDF 5/3 (LeGall) | Yes | Yes | JPEG 2000 lossless |
| CDF 9/7 | Yes | Yes | JPEG 2000 lossy |
| Symlet-4 | Yes | Yes | 8-tap orthonormal |

---

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

---

## C API Example

If you prefer to build transforms by hand, the C API is straightforward:

```c
#include <fastandfourier.h>
#include <math.h>
#include <stdlib.h>

int main() {
    faf_init();

    // Create 1024-point FFT
    faf_transform* fft = faf_create_fft(1024, false, FAF_PREC_FP32, 0);

    float* in  = aligned_alloc(64, 2 * 1024 * sizeof(float));
    float* out = aligned_alloc(64, 2 * 1024 * sizeof(float));

    // Generate complex sine wave
    for (int i = 0; i < 1024; i++) {
        in[2*i]     = sinf(2.0f * M_PI * 4.0f * i / 1024.0f);
        in[2*i + 1] = 0.0f;
    }

    // Execute using JIT-compiled kernel (default)
    faf_execute_jit(fft, out, in);

    free(in); free(out);
    faf_destroy_transform(fft);
    faf_cleanup();
    return 0;
}
```

Compile with:
```bash
gcc -O3 -o fft_example example.c -lfastandfourier -lm -ldl
```

---

## API Reference

### Transform Creation

| Function | Description |
|----------|-------------|
| `faf_create_fft(n, inverse, precision, flags)` | Create FFT transform |
| `faf_create_dct(n, type, precision, flags)` | Create DCT transform (type 1-4) |
| `faf_create_dst(n, type, precision, flags)` | Create DST transform (type 1-4) |
| `faf_create_haar(n, levels, precision, flags)` | Create Haar wavelet |
| `faf_create_daubechies4(n, levels, precision, flags)` | Create Daubechies-4 / D4 |
| `faf_create_cdf53` / `faf_create_cdf97` / `faf_create_sym4` | JPEG 2000 and Symlet families |
| `faf_create_dwt(family, n, levels, inverse, prec, flags)` | Generic DWT / IDWT |
| `faf_create_mdct(n, precision, flags)` | Create MDCT transform |
| `faf_destroy_transform(t)` | Free transform and associated memory |

### Execution Functions

| Function | Description |
|----------|-------------|
| `faf_execute(t, out, in)` | Auto-select best backend |
| `faf_execute_f32(t, out, in)` | FP32 execution |
| `faf_execute_f64(t, out, in)` | FP64 execution |
| `faf_execute_jit(t, out, in)` | JIT-compiled execution |
| `faf_execute_jit_ex(t, out, in, flags)` | JIT with extended options |

### JIT Compilation

| Function | Description |
|----------|-------------|
| `faf_jit_create()` | Create JIT context |
| `faf_jit_compile(ctx, t)` | Compile transform to native code |
| `faf_jit_get_kernel(ctx)` | Get compiled kernel function |
| `faf_jit_destroy(ctx)` | Free JIT context |

### Chirp DSL API

Include `<chirp.h>` to use the Chirp DSL.

| Function | Description |
|----------|-------------|
| `chirp_register(name, fn)` | Register a custom C function |
| `chirp_compile(source)` | Compile Chirp S-expression to transform |

### JIT Flags

| Flag | Description |
|------|-------------|
| `FAF_FLAG_JIT_SIMD` | Enable SIMD intrinsics (default on supported platforms) |
| `FAF_FLAG_JIT_INPLACE` | Work in-place (overwrites input) |
| `FAF_FLAG_JIT_SPLIT_PLANE` | Use split-plane mode (default, faster for most sizes) |

### Precision Levels

| Enum | Description |
|------|-------------|
| `FAF_PREC_FP8` | 8-bit floating point (E4M3/E5M2) |
| `FAF_PREC_FP16` | 16-bit IEEE 754 half precision |
| `FAF_PREC_BF16` | 16-bit brain floating point |
| `FAF_PREC_FP32` | 32-bit single precision |
| `FAF_PREC_FP64` | 64-bit double precision |

### Data Layout

- Complex data is stored as interleaved real/imaginary pairs (`float[2*n]`)
- All buffers must be 64-byte aligned for SIMD performance
- Transform size N must be a power of 2 for FFT
- **Split-plane mode** (default for FP32/FP64): Internally deinterleaves to separate real/imaginary arrays for optimal vectorization

---

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

---

## Benchmark Results

macOS Darwin 22.6.0 (Ventura), 16 × 2300 MHz (x86\_64, AVX2 + SSE4.2; no AVX-512), -O3 -march=native -ffast-math, measured 2026-02-22

### Library Comparison (complex-to-complex, FP64/double precision, M transforms/sec)

All libraries benchmarked in double precision (FP64). FAF VM and FAF JIT are measured separately.

| Size | FAF VM | FAF JIT | KissFFT | NotoriousFFT | FFTW3 (est) | FFTW3 (meas) |
|-----:|-------:|--------:|--------:|-------------:|------------:|-------------:|
|   16 | — | 0.58 | 4.75 | 4.65 | 50.1 | 49.3 |
|   64 | — | 0.23 | 0.90 | 0.68 | 6.05 | **15.5** |
|  256 | — | 0.060 | 0.18 | 0.12 | 1.29 | 3.55 |
| 1024 | — | 0.013 | 0.037 | 0.022 | 0.23 | 0.56 |
| 4096 | — | 0.0029 | 0.0080 | 0.0043 | 0.042 | 0.095 |
| 16384 | — | 0.00063 | 0.0016 | 0.00092 | 0.0072 | 0.015 |
| 65536 | — | 0.0011 | 0.00031 | 0.00020 | 0.0013 | **0.0033** |

*Higher is better. FFTW3 (est) skips planning; FFTW3 (meas) uses the full measure optimizer with planning time amortized out. M transforms/sec = items\_per\_second ÷ N.*

**Notes:**
- All benchmarks use **double precision (FP64)** across all libraries for fair comparison
- FAF VM is capped at N≤128 by the IR register file limit (— means the path is unavailable, not slow)
- FAF JIT shows decreasing throughput with N (cache effects), but spikes at N=65536 (algorithm switch)
- FFTW3 (measure) is fastest overall; at N=64 it is **67× faster** than FAF and **17× faster** than KissFFT
- KissFFT maintains consistent performance: ~2× faster than NotoriousFFT, ~3-5× faster than FAF
- NotoriousFFT now included (double precision) — bugs fixed in latest version

---

## Architecture Support

| Platform | SSE4.2 | AVX2 | AVX-512 | NEON | SVE | CUDA |
|----------|--------|------|---------|------|-----|------|
| x86_64 | Yes | Yes | Yes | No | No | Yes |
| AArch64 | No | No | No | Yes | Yes | No |

---

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Chirp DSL](CHIRP.md) - Scheme-like DSL for DSP pipelines with 140+ builtins
- [Performance Analysis](ANALYSIS.md) - Detailed benchmark analysis
- [Architecture Overview](docs/ARCHITECTURE.md) - IR and VM design
- [Building Guide](docs/BUILDING.md) - Advanced build options

---

## Algorithm

FastAndFourier uses a flexible Intermediate Representation (IR) with 31 opcodes that can express multiple transform types. The JIT compiler generates optimized C code from this IR, which is then compiled to native machine code using the system compiler.

For FFT, the library uses radix-2 and radix-4 Cooley-Tukey algorithms with bit-reversal permutation. DCT and DST are computed via real FFT with pre/post-processing twiddles. Wavelet transforms use the lifting scheme for in-place computation.

All SIMD kernels use architecture-specific intrinsics (NEON on ARM, AVX/AVX-512 on x86) for butterfly operations and twiddle factor multiplication.

---

## License

MIT License — see [LICENSE](LICENSE) file.

## Acknowledgments

- FFT algorithms based on Cooley-Tukey
- Wavelet transforms using lifting scheme
- JIT approach inspired by GNU Lightning and LLVM ORC
- Benchmark comparisons use [Kiss FFT](https://github.com/mborgerding/kissfft), [Notorious FFT](https://github.com/NotoriousLOB/Notorious-FFT), [FFTW3](http://www.fftw.org/), PocketFFT, and minFFT
