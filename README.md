<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/images/logo.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/images/logo.png">
  <img src="/.github/images/logo.png" width="100%" alt="Fast and Fourier" >
</picture>
</p>

# FastAndFourier

**A small, embeddable DSP library. Transforms are described in a 34-opcode IR, compiled at runtime into SIMD kernels, and composed through a Scheme-flavored scripting language — Chirp — that turns a string into an executable pipeline.**

```scheme
(pipeline
  (rfft :size 4096)
  (spectral notch)
  (irfft))
```

That is a complete program. `chirp_compile` turns it into a `faf_transform *`; `faf_execute` runs it real-in, real-out. There is no planner object to manage and no bytecode to assemble by hand. You describe the pipeline, you get a pointer.

The honest way to evaluate a library is the way you would evaluate a ride: what can it do, what does it cost, and where does it break. Here are the answers, in that order.

---

## What it does

- **Scripting.** Chirp is built into the library: S-expressions with Smalltalk-style keyword arguments, 140+ registered builtins (windows, distributions, activations, vector statistics), and a registry for your own C functions. A denoiser is three lines; a matched filter is one expression and one `chirp_bind`.

- **A real JIT.** The IR is lowered to C, handed to the system compiler, and linked back in at runtime with AVX2/AVX-512/NEON intrinsics where the hardware allows. Compilation is cached on the transform object (thread-safe, pay once), with an on-disk cache across process runs.

- **The whole family.** FFT (complex and real-input), DCT and DST types I–IV, STFT, MDCT, CWT/CQT filter bank, and five DWT wavelet families — Haar, Daubechies-4, CDF 5/3, CDF 9/7, Symlet-4 — forward and inverse, all from one config struct and one execute call.

- **Explicit data contracts.** Split-plane, Hermitian-packed, interleaved, and real layouts; NumPy-style, ortho, forward-scaled, and JPEG 2000 normalization conventions. The library never silently converts a layout for you. If a buffer does not match the transform, you get an error, not a wrong answer.

- **Portable C11, no required dependencies.** A static archive, four public headers, `libm`/`libdl`/`pthread` on Linux. SIMD dispatch is automatic at build time. The same source builds on x86_64 and AArch64.

## What it costs

- **The JIT needs a system C compiler at runtime** (`cc`/`gcc`/`clang`). On a locked-down target without one, every transform still runs — it just runs on the interpreter. Set `cfg.backend = FAF_BACKEND_VM` or build with `-DFASTANDFOURIER_SINGLE_HEADER=ON` to make that choice permanent.
- **Small transforms pay interpreter overhead.** Below roughly N = 512 there is a ~1 µs dispatch floor on the VM path. The JIT threshold defaults to N ≥ 128 (`FAF_JIT_AUTO_THRESHOLD`) and can be tuned.
- **The register file is sized for transforms up to 2^16 points.** That is a design choice, not a bug: it keeps the VM state small and the JIT output cache-friendly.

## Where it breaks

If your problem is "the fastest possible FFT of a fixed size, plan reused for years," the correct answer is FFTW3 with `FFTW_MEASURE`. It wins on this machine at nearly every size, and we publish our own numbers saying so (see [Quarter-mile times](#quarter-mile-times)). We're not here to race the reigning champion at the light.

The case for FastAndFourier is different: you want transforms you can *compose, script, and embed* — a filter chain described in a config file, a wavelet denoiser tuned without recompiling, a spectral pipeline that fuses R2C → pointwise C → C2R into one kernel. That is the trade-off we make, and we make it openly.

---

## What's new in 1.1.0

- **CWT / CQT filter bank.** `faf_create_cwt` builds an analytic, zero-phase,
  geometrically spaced bank (Morse default, MATLAB TimeBandwidth=60), certifies
  the Littlewood–Paley sum, and refuses to ship a bank that does not tile.
  Dual-frame inversion via `faf_create_inverse` reconstructs; an L1
  one-integral inverse is there when you want Calderón instead of a dual.
  Details, and the theorem the checks are special cases of, in
  [docs/CWT.md](docs/CWT.md). Chirp: `(cwt :n 4096 :wavelet morse)`.
- **Two DWT backends.** Lifting for the five short PR families, polyphase FIR
  for custom taps and analysis-only atoms. AUTO is a rule, not a heuristic.
  See [docs/DWT.md](docs/DWT.md).
- **Optional Bluestein.** Non-5-smooth FFT lengths only with
  `FAF_FLAG_BLUESTEIN` / Chirp `:bluestein`. Off by default. RFFT and DWT
  refuse it. See [docs/SIZES.md](docs/SIZES.md).
- **Split-plane correlator.** Packed Hermitian multiply and
  [`examples/split_corr.c`](examples/split_corr.c). Layout FAQ in
  [docs/LAYOUT.md](docs/LAYOUT.md).

## What's new in 1.0.0

- **Config-based creation.** One `faf_config` struct drives every transform type: `faf_config_init(n)`, set the knobs you care about, call `faf_create_*(&cfg)`. Defaults are sensible (FP32, split layout, NumPy norm) and every default is visible in `t->cfg`.
- **Real-input FFT.** `faf_create_rfft` gives a true R2C with a packed Hermitian spectrum of `n/2 + 1` bins — not a C2C with a zeroed imaginary plane. `faf_create_inverse(fwd)` hands back the matching C2R with every knob inherited. Details in [docs/RFFT.md](docs/RFFT.md).
- **Mixed-radix sizes.** FFT length must be 5-smooth (`2^a · 3^b · 5^c`), not a power of two. `faf_is_size_supported` and `faf_get_recommended_size` answer the practical questions. Optional Bluestein (`FAF_FLAG_BLUESTEIN`) is opt-in for primes; see [docs/SIZES.md](docs/SIZES.md).
- **Fused spectral pipelines.** `(pipeline (rfft) (spectral f) (irfft))` compiles to a single fused REAL → REAL transform. Bind an external spectrum with `chirp_bind` and `(mul-spectrum)` gives you convolution and matched filtering without touching a twiddle.
- **Cached JIT execution.** `faf_execute_jit_cached` compiles once, pins the kernel to the transform, and takes a compare-and-swap fast path thereafter.

---

## Getting started

### Requirements

- CMake ≥ 3.18, a C11 compiler (and C++17 for the tests)
- Linux, macOS, or Windows; x86_64 or AArch64
- Nothing else. Tests and benchmarks fetch Google Test / Google Benchmark automatically. The JIT additionally wants a system C compiler at runtime, and works without one (see above).

### Build

```bash
git clone https://github.com/NotoriousLOB/FastAndFourier.git
cd FastAndFourier
cmake -S . -B build
cmake --build build -j
```

Tests, benchmarks, and examples build by default. Then:

```bash
ctest --test-dir build --output-on-failure   # run the test suite
./build/examples/que_onda_mundo              # the canonical first program
```

`que_onda_mundo` compiles `(fft :size 8)` from a string, runs an impulse through it, and prints the flat spectrum. If you see eight ones, everything works.

If you only want the library — no test dependencies, no network fetch:

```bash
cmake -S . -B build \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF
cmake --build build -j
```

### Use it from your code

Link the static archive and go:

```bash
gcc -O3 -o my_app my_app.c -I/path/to/FastAndFourier/include \
    -L/path/to/FastAndFourier/build -lfastandfourier -lm -ldl -lpthread
```

(On macOS, drop `-ldl`. On Windows, link `fastandfourier.lib` alone.)

Or install system-wide:

```bash
cmake --install build   # static lib + headers + CMake package config
```

---

## The C API in five minutes

Every transform follows the same four steps: init a config, create the transform, wrap your buffers, execute.

```c
#include <fastandfourier.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    faf_init();

    const size_t n = 1024;                      /* any 5-smooth size */
    faf_config cfg = faf_config_init(n);        /* FP32, split, norm NONE */
    faf_transform *fft = faf_create_fft(&cfg);
    if (!fft) { fprintf(stderr, "%s\n", faf_get_error()); return 1; }

    /* Default C2C layout is split-plane: separate re[] and im[] arrays. */
    float *re_in  = faf_aligned_alloc(n * sizeof(float));
    float *im_in  = faf_aligned_alloc(n * sizeof(float));
    float *re_out = faf_aligned_alloc(n * sizeof(float));
    float *im_out = faf_aligned_alloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        re_in[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        im_in[i] = 0.0f;
    }

    faf_buffer in  = faf_buffer_split(re_in,  im_in,  n);
    faf_buffer out = faf_buffer_split(re_out, im_out, n);
    faf_execute(fft, &out, &in);                /* AUTO: JIT if n >= 128, else VM */

    faf_aligned_free(re_in); faf_aligned_free(im_in);
    faf_aligned_free(re_out); faf_aligned_free(im_out);
    faf_destroy_transform(fft);
    faf_cleanup();
    return 0;
}
```

Three things to know:

1. **Buffers are descriptors, not allocations.** `faf_buffer_*` wraps memory you own. The library reads `in`, writes `out`, and allocates its own scratch.
2. **Layouts are checked.** A split transform given an interleaved buffer returns an error. Convert explicitly at the starting line with `faf_deinterleave_f32` / `faf_interleave_f32`.
3. **Alignment matters for speed.** Use `faf_aligned_alloc` (64-byte) for hot buffers. `faf_execute_f32` / `faf_execute_f64` remain as interleaved convenience wrappers if you prefer the classic style.

### Real input

Half the spectrum of a real signal is redundant, so don't pay for it:

```c
faf_config cfg = faf_config_init(4096);         /* n even and 5-smooth */
faf_transform *fwd = faf_create_rfft(&cfg);
faf_transform *inv = faf_create_inverse(fwd);   /* C2R, same knobs, flipped direction */

size_t nb = faf_spectrum_len(fwd);              /* 2049 = n/2 + 1 */
faf_buffer x    = faf_buffer_real(time, 4096);
faf_buffer spec = faf_buffer_hermitian(Xr, Xi, nb);
faf_buffer y    = faf_buffer_real(back, 4096);

faf_execute(fwd, &spec, &x);                    /* REAL -> HERMITIAN */
faf_execute(inv, &y, &spec);                    /* HERMITIAN -> REAL, 1/n applied */
```

DC and Nyquist are real by construction (`im[0] == im[nb-1] == 0` within rounding). The full contract — layouts, norms, what this is *not* — is in [docs/RFFT.md](docs/RFFT.md).

---

## Chirp in five minutes

Chirp is the same engine, driven by a string instead of a config struct. Include `<chirp.h>`, register the builtins, compile, execute.

```c
#include <chirp.h>
#include <chirp_builtins.h>

faf_init();
chirp_register_standard_builtins();

/* A complete wavelet denoiser: decompose, soft-threshold the detail
 * bands, reconstruct. */
faf_transform *denoise = chirp_compile(
    "(pipeline "
    "  (dwt :family cdf97 :size 1024 :levels 5)"
    "  (threshold :mode soft :lambda 0.08)"
    "  (idwt :family cdf97 :size 1024 :levels 5))"
);

faf_execute_f32(denoise, out, in);
faf_destroy_transform(denoise);
```

### Your C, inside the pipeline

The interesting case is mixing your own code with the built-ins. Register a spectral function — one call with live split-plane pointers over the Hermitian bins — and Chirp will fuse it between an R2C and its inverse:

```c
struct notch_ctx { size_t bin; };

static void notch(float *re, float *im, size_t n_bins, void *ctx) {
    size_t k = ((struct notch_ctx *)ctx)->bin;
    if (k < n_bins) { re[k] = 0.0f; im[k] = 0.0f; }
}

struct notch_ctx ctx = { .bin = 16 };
chirp_register_spectral("notch", notch, &ctx);

faf_transform *t = chirp_compile(
    "(pipeline (rfft :size 256) (spectral notch) (irfft))");

faf_buffer in  = faf_buffer_real(x, 256);
faf_buffer out = faf_buffer_real(y, 256);
faf_execute(t, &out, &in);        /* real in, real out, tone at bin 16 gone */
```

For convolution and matched filtering, precompute a spectrum once, bind it, and multiply:

```c
faf_transform *corr = chirp_compile(
    "(pipeline (rfft :size 128) (mul-spectrum) (irfft))");
chirp_bind(corr, "H", Hr, Hi, nb);   /* conjugate Hi first to correlate */
```

`chirp_register_unary` covers scalar elementwise functions; `chirp_register` / `chirp_register_ex` cover the rest. The full language reference — 140+ builtins, lifting composition, error handling — is [CHIRP.md](CHIRP.md). The examples directory runs the tour: [`que_onda_mundo.c`](examples/que_onda_mundo.c) (hello world), [`spectral_notch.c`](examples/spectral_notch.c), [`matched_filter.c`](examples/matched_filter.c), [`chirp_dwt_denoise.c`](examples/chirp_dwt_denoise.c), and fourteen more.

---

## Reference

### Transforms

| Transform | Forward | Inverse | Size rule | Notes |
|-----------|:-------:|:-------:|-----------|-------|
| FFT (C2C) | Yes | Yes | 5-smooth | Radix-2/3/4/5 mixed-radix |
| RFFT (R2C/C2R) | Yes | Yes | even, 5-smooth | Packed Hermitian, `n/2+1` bins |
| DCT I–IV | Yes | Yes | — | `cfg.dct_type`, 0 means II |
| DST I–IV | Yes | Yes | — | `cfg.dct_type`, 0 means II |
| STFT | Yes | Yes | — | `cfg.hop_length`, `cfg.win_length` |
| MDCT | Yes | Yes | even | AAC/Opus-style |
| Haar | Yes | Yes | power of 2 | Orthonormal lifting (default); FIR override |
| Daubechies-4 (db2) | Yes | Yes | power of 2 | 4-tap orthonormal |
| CDF 5/3 | Yes | Yes | power of 2 | JPEG 2000 lossless |
| CDF 9/7 | Yes | Yes | power of 2 | JPEG 2000 lossy |
| Symlet-4 | Yes | Yes | power of 2 | 8-tap orthonormal |
| CWT (filter bank) | Yes | Yes | even, 5-smooth | Fourier-domain CQT, LP-certified, [docs/CWT.md](docs/CWT.md) |

DWT has two backends. Lifting (Backend L) is the default for the five short
PR families; polyphase FIR (Backend F) is for custom taps and analysis-only
atoms. AUTO selection and the Haar convention table live in
[docs/DWT.md](docs/DWT.md). Inverse is `faf_create_inverse` / Chirp
`(inverse)`, not a second keyword pile.

All of the above, plus fused pipelines, run through the same 34-opcode IR. Adding a transform means adding opcodes, not forking the engine.

### Layouts, norms, precisions, backends

| `faf_layout` | Meaning |
|---|---|
| `FAF_LAYOUT_SPLIT` | `re[n]`, `im[n]` — default for C2C |
| `FAF_LAYOUT_HERMITIAN` | packed R2C spectrum, `n/2+1` split bins — default for RFFT |
| `FAF_LAYOUT_REAL` | `float[n]` — native for DWT/DCT, time-domain side of R2C |
| `FAF_LAYOUT_INTERLEAVED` | `float[2n]`, opt-in convenience — slower; see [docs/LAYOUT.md](docs/LAYOUT.md) |

| `faf_norm` | Forward | Inverse | Used by |
|---|---|---|---|
| `FAF_NORM_NONE` (FFT default) | unscaled | `1/n` | NumPy convention |
| `FAF_NORM_ORTHO` | `1/√n` | `1/√n` | Haar/D4/Sym4 default |
| `FAF_NORM_FORWARD` | `1/n` | unscaled | opt-in |
| `FAF_NORM_LAZY` / `FAF_NORM_JPEG2000` | family conventions | Haar split / CDF |

Precisions: `FAF_PREC_FP8`, `FP16`, `BF16`, `FP32` (default), `FP64` (plus integer enum values). FP8/FP16 compile-time gates: `-DENABLE_FP8=ON/OFF`, `-DENABLE_FP16=ON/OFF` (both default ON).

Backends: `FAF_BACKEND_AUTO` (default — JIT when `n ≥ FAF_JIT_AUTO_THRESHOLD` = 128, else VM), `FAF_BACKEND_VM`, `FAF_BACKEND_JIT`.

### API summary

| Group | Functions |
|---|---|
| Lifecycle | `faf_init`, `faf_cleanup`, `faf_version`, `faf_arch_name` |
| Config | `faf_config_init`, `faf_config_from`, `faf_config_inverse` |
| Buffers | `faf_buffer_real`, `faf_buffer_split`, `faf_buffer_interleaved`, `faf_buffer_hermitian` |
| Creation | `faf_create`, `faf_create_fft`, `faf_create_rfft`, `faf_create_dct`, `faf_create_dst`, `faf_create_stft`, `faf_create_mdct`, `faf_create_haar`, `faf_create_daubechies4`, `faf_create_cdf53`, `faf_create_cdf97`, `faf_create_sym4`, `faf_create_dwt`, `faf_create_inverse` |
| Execution | `faf_execute` (preferred), `faf_execute_f32/f64/f16/fp8`, `faf_execute_vm`, `faf_execute_jit`, `faf_execute_jit_cached`, `faf_execute_jit_ex` |
| Layout conversion | `faf_deinterleave_f32/f64`, `faf_interleave_f32/f64` |
| JIT context | `faf_jit_create`, `faf_jit_compile`, `faf_jit_compile_ex`, `faf_jit_get_kernel`, `faf_jit_set_verbose`, `faf_jit_destroy` |
| Sizing | `faf_spectrum_len`, `faf_is_size_supported`, `faf_get_recommended_size`, `faf_get_alignment`, `faf_precision_size` |
| Chirp | `chirp_compile`, `chirp_register`, `chirp_register_ex`, `chirp_register_unary`, `chirp_register_spectral`, `chirp_register_spectral_ex`, `chirp_register_standard_builtins`, `chirp_bind`, `chirp_cleanup` |
| Errors | `faf_get_error`, `faf_clear_error` |
| Memory | `faf_aligned_alloc`, `faf_aligned_free` |

---

## How execution works

Every transform is compiled to IR: 34 opcodes covering butterflies, twiddles, staged FFT/DCT/DWT steps, lifting, reductions, permutations, and builtin calls. From there, three paths:

1. **VM** — a direct-threaded interpreter over the IR. Always available, no runtime dependencies, bounded register file (N ≤ 65536).
2. **JIT** — the IR is emitted as C with architecture intrinsics, compiled by your system compiler, and `dlopen`ed. With `faf_execute_jit_cached` the compiled kernel is pinned to the transform: first call pays the compile, every call after takes the fast path.
3. **Fused pipeline** — a Chirp `(rfft … (spectral …) (irfft))` chain is lowered to a single generated kernel with your spectral call inside it. One pass, real to real.

The JIT is the NOS button: a one-time hit of compile overhead at the start of the run, in exchange for a faster trip down the strip on every execution after. Whether the bottle is worth it depends on how many times you run the same transform — which is exactly why `FAF_BACKEND_AUTO` exists.

---

## Quarter-mile times

Everyone in this business has a quarter-mile time. Here are ours, with the track conditions: macOS Ventura, x86_64 at 2.3 GHz (AVX2, no AVX-512), `-O3 -march=native -ffast-math`, double precision across all libraries, measured 2026-02-22.

**Single-transform latency, C2C, FP64 (µs — lower is better):**

| Size | FastAndFourier | KissFFT | NotoriousFFT | FFTW3 (est) | FFTW3 (measure) |
|-----:|---------------:|--------:|-------------:|------------:|----------------:|
|   64 |           4.40 |    1.11 |         1.47 |       0.165 |           0.065 |
| 1024 |           75.3 |    27.1 |         45.7 |        4.28 |            1.79 |
| 16384 |          1595 |     634 |         1088 |         140 |            67.0 |
| 65536 |           907 |    3253 |         5055 |         748 |             301 |

What to take from this table:

- **Small N is not our race.** Below ~512 points the VM dispatch floor (~1 µs) dominates; at N = 8 that is 24× behind KissFFT. If you need a fast 64-point transform, use KissFFT and keep the change.
- **The gap closes with N, then inverts.** Crossover against KissFFT is ≈ N = 32768 on this machine; at N = 65536 FastAndFourier is **3.2–3.6× faster than KissFFT**, because the JIT engages and the AVX2 kernels do the work the interpreter used to schedule.
- **FFTW3 wins overall.** `FFTW_MEASURE` is 3–10× ahead of everything on this table. It has twenty-five years of codelet tuning behind it; respect where respect is due.

Reproduce or extend: `cmake -S . -B build -DBUILD_BENCHMARKS=ON`, then `./build/benchmarks/fastandfourier_benchmarks`. Comparisons against KissFFT, PocketFFT, MinFFT, muFFT, NotoriousFFT, and FFTW3 build with `-DENABLE_EXTERNAL_BENCHMARKS=ON`. Full methodology and more tables: [benchmarks/README.md](benchmarks/README.md).

---

## Build options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Test suite (Google Test, fetched automatically) |
| `BUILD_BENCHMARKS` | ON | Benchmarks (Google Benchmark, fetched automatically) |
| `BUILD_EXAMPLES` | ON | 18 example programs |
| `BUILD_DOCS` | ON | Doxygen docs, if Doxygen is found |
| `ENABLE_EXTERNAL_BENCHMARKS` | OFF | Also fetch and build KissFFT, PocketFFT, MinFFT, muFFT, NotoriousFFT, FFTW3 |
| `ENABLE_CUDA` | OFF | CUDA backend |
| `ENABLE_FP8` / `ENABLE_FP16` | ON / ON | Half- and quarter-precision support |
| `ENABLE_ASAN` / `UBSAN` / `TSAN` / `MSAN` | OFF | Sanitizers (mutually exclusive as noted in CMake) |
| `ENABLE_WERROR` | OFF | Warnings as errors |
| `ENABLE_COVERAGE` | OFF | Coverage instrumentation + `make coverage` |
| `FASTANDFOURIER_SINGLE_HEADER` | OFF | No-JIT portability mode (VM only) |
| `FASTANDFOURIER_PRECOMPILE` | OFF | Pre-generate codelets for common sizes |

Release builds use `-O3 -march=native -ffast-math -funroll-loops`; the CMake configure step detects SSE4.2 / AVX2 / AVX-512 on x86_64 and NEON / SVE on AArch64 and prints what it found.

## Platform support

| Platform | SSE4.2 | AVX2 | AVX-512 | NEON | SVE | CUDA |
|----------|:------:|:----:|:-------:|:----:|:---:|:----:|
| x86_64 | Yes | Yes | Yes | — | — | Experimental |
| AArch64 | — | — | — | Yes | Yes | — |

A word on the CUDA backend: it is behind `-DENABLE_CUDA=ON` and is not yet the fast line on this track. The CPU paths are where the development miles have gone for 1.1.0.

---

## Repository map

```
include/    fastandfourier.h (public API) · chirp.h · chirp_builtins.h · faf.h (internal IR)
src/        core, VM, transforms, twiddles, wavelets, RFFT, pipeline
src/jit/    the JIT compiler
src/arch/   x86 (SSE/AVX2/AVX-512) · arm (NEON/SVE) · cuda
examples/   18 runnable programs, from que_onda_mundo to matched_filter
tests/      Google Test suite (ctest)
benchmarks/ Google Benchmark suite + comparison harness
docs/       RFFT contract · CWT contract · Doxygen config
CHIRP.md    the language reference
```

## License

MIT — see [LICENSE](LICENSE). Take it racing.

## Acknowledgments

Family matters, and this one has roots: Cooley and Tukey for the algorithm; FFTW, Intel IPP, and Apple vDSP for showing what fast looks like; GNU Lightning and LLVM ORC for the JIT inspiration; and the libraries we benchmark against — [KissFFT](https://github.com/mborgerding/kissfft), [FFTW3](http://www.fftw.org/), [PocketFFT](https://github.com/mreineck/pocketfft), [MinFFT](https://github.com/aimukhin/minfft), [muFFT](https://github.com/Themaister/muFFT), and [NotoriousFFT](https://github.com/NotoriousLOB/Notorious-FFT) — for keeping the timing sheets honest.
