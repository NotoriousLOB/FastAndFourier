# Chirp DSL Documentation

> *"You want functions? We got functions."*

Chirp is a Scheme-like domain-specific language (DSL) with Smalltalk-style keyword arguments for describing DSP pipelines. It compiles to FastAndFourier's Intermediate Representation (IR) for high-performance execution via JIT compilation.

## Table of Contents

- [Quick Start](#quick-start)
- [Language Overview](#language-overview)
- [Syntax Reference](#syntax-reference)
- [Builtin Functions](#builtin-functions)
- [Examples](#examples)
- [C API](#c-api)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

The canonical first program is [`examples/que_onda_mundo.c`](examples/que_onda_mundo.c)
(*¿Qué onda mundo?* — Chirp does not say hello, it asks what's up).

```c
#include <chirp.h>
#include <chirp_builtins.h>

chirp_register_standard_builtins();

faf_transform *t = chirp_compile("(fft :size 8)");
faf_execute_f32(t, out, in);   /* impulse in → flat spectrum out */
faf_destroy_transform(t);
```

From there you can grow a pipeline:

```c
faf_transform *dwt = chirp_compile(
    "(pipeline "
    "  (dwt :family cdf97 :size 1024 :levels 5)"
    "  (threshold :mode soft :lambda 0.08)"
    "  (inverse))"
);
```

---

## Language Overview

Chirp uses S-expressions with Smalltalk-style keyword arguments:

```scheme
(operator :keyword1 value1 :keyword2 value2 arg1 arg2)
```

### Design Philosophy

1. **Composition over monoliths**: Complex operations are built from simple primitives
2. **First-class functions**: Registered builtins can be used directly without `(custom ...)` wrapper
3. **Keyword clarity**: Smalltalk-style `:keywords` make arguments self-documenting
4. **Pipeline-centric**: The `(pipeline ...)` form sequences operations naturally

---

## Syntax Reference

### Literals

```scheme
42          ; Integer
3.14        ; Float (becomes f32 or f64 based on precision)
symbol      ; Symbol (function name, variable)
```

### Keywords

```scheme
:size       ; Used for specifying sizes (fft, arrays)
:predict    ; Prediction function (lifting scheme)
:update     ; Update function (lifting scheme)
```

### Core Forms

#### `(let NAME expr [body…])` / `(inverse)` / `(inverse NAME)`

`(let)` binds a compiled form to a name. Body is optional: a following
top-level form can use the name too.

```scheme
(let W (dwt :family haar :size 64 :levels 3)
  (pipeline W (threshold :mode soft :lambda 0.1) (inverse W)))

(let F (rfft :size 3840 :norm ortho :layout hermitian))
(pipeline F (spectral corr) (inverse F))
```

- `(inverse)` inverts the nearest preceding invertible stage in the same pipeline.
- `(inverse NAME)` inverts a `let` binding. Repeating `:size` is a compile error.

#### `(bind NAME :h h :g g [:h-syn hs :g-syn gs])`

Binds a custom PR (or analysis-only) tap set from vectors registered in C
with `chirp_register_vector`. Then `(dwt :taps NAME :conv custom-pr :family custom …)`.

```c
chirp_register_vector("h", h, 2);
chirp_register_vector("g", g, 2);
chirp_register_vector("ht", ht, 2);
chirp_register_vector("gt", gt, 2);
chirp_compile(
    "(bind lazy :h h :g g :h-syn ht :g-syn gt)"
    "(dwt :family custom :size 64 :levels 2 :backend fir :conv custom-pr :taps lazy)");
```

Or skip the bind form and call `chirp_register_taps("lazy", h, g, 2, ht, gt, 2)`
directly. `ht`/`gt` omitted ⇒ analysis-only (no inverse).

#### `(pipeline expr1 expr2 ...)`

Sequences multiple operations. Returns a transform that executes each expression in order.

```scheme
(pipeline
  (fft :size 1024)
  twiddle
  (bfly 4)
  reduce-sum)
```

#### `(fft :size N)` / `(ifft :size N)`

Full staged Fast Fourier Transform (bit-reversal + mixed-radix stages).

- `:size N` — Transform size (must be 5-smooth: `2^a · 3^b · 5^c`, unless `:bluestein`)
- `:layout split|interleaved` — Buffer layout (default: interleaved in Chirp)
- `:norm none|ortho|forward` — Same contract as C `faf_norm`
- `:precision f32|f64`
- `:bluestein` — Opt-in chirp-z for non-5-smooth `N`. Explicit; never implied. See [docs/SIZES.md](docs/SIZES.md).

```scheme
(fft :size 256)      ; 256-point FFT
(fft :size 3840)     ; 3840 = 2^8 · 3 · 5 — legal, mixed-radix
(fft :size 4096)     ; 4096-point FFT
(ifft :size 1024)    ; inverse FFT
(fft :size 307 :bluestein :layout split)  ; prime length, opt-in
```

#### `(dwt :family name :size N :levels L)` / `(idwt …)`

Discrete wavelet transform. Families: `haar`, `d4` / `daubechies4` / `db2`, `cdf53` / `legall`, `cdf97`, `sym4`.

- `:size N` — power of 2 (DWT sizes must be powers of two, unlike FFT; see [docs/DWT.md](docs/DWT.md))
- `:levels L` — Mallat levels (`0` or omitted = full `log2(N)`)
- `:conv name` — Haar-ortho / Haar-lazy / Haar-mean / JPEG 5/3 / … (see docs/DWT.md)
- `:backend auto|lift|fir` — AUTO is the §6.3 rule; `cdf53` + `:backend fir` is a compile error
- Periodic boundaries, real-valued (imaginary plane left untouched)

`(dwt :size 3840)` is **illegal** — DWT requires dyadic sizes. Use
`(fft :size 3840)` or `(rfft :size 3840)` for 5-smooth sizes.

```scheme
(dwt :family cdf97 :size 1024 :levels 5)
(idwt :family cdf97 :size 1024 :levels 5)
(dwt :family haar :size 1024 :levels 4 :conv haar-ortho :backend fir)
```

#### `(rfft :size N …)` / `(irfft :size N …)`

Real-to-complex FFT. Standalone or as the endpoints of a fused spectral
pipeline. See [docs/RFFT.md](docs/RFFT.md) for the full contract.

| Keyword | Default | Description |
|---------|---------|-------------|
| `:size N` | *required* | Signal length (even, 5-smooth) |
| `:norm kind` | `none` | `none`, `ortho`, `forward` |
| `:layout kind` | `hermitian` | `hermitian` (split-plane) or `interleaved` |
| `:precision p` | `f32` | `f32` or `f64` |

`(irfft)` accepts the same keywords. In a pipeline, use `(inverse)` instead
to inherit all settings from the preceding `(rfft)`. `:bluestein` on
`(rfft)` is a compile error (v1).

```scheme
(rfft :size 4096 :norm ortho :layout hermitian)
(irfft :size 4096)

; Fused spectral pipeline — real in, real out:
(pipeline (rfft :size 256) (spectral notch) (irfft))
```

`:norm lazy` and `:norm jpeg2000` are rejected at compile time (they belong to
DWT conventions, not RFFT).

#### `(cwt :n N …)` / `(icwt :n N …)`

Continuous Wavelet Transform (Fourier-domain CQT filter bank). **Standalone
only** — cannot appear inside a `(pipeline …)`.

Keywords:

| Keyword | Default | Description |
|---------|---------|-------------|
| `:n N` | *required* | Signal length (even, 5-smooth) |
| `:fs F` | `1.0` | Sample rate |
| `:wavelet name` | `morse` | `morlet`, `morse`, `bump`, `shannon`, `meyer` |
| `:voices V` | `10` | Voices per octave |
| `:fmin F` | auto | Lowest center frequency |
| `:fmax F` | auto | Highest center frequency |
| `:gamma G` | `3.0` | Morse γ parameter |
| `:beta B` | `20.0` | Morse β (Lilly–Olhede). MATLAB TimeBandwidth=60 is β=20, not 60. |
| `:mu M` | `6.0` | Morlet μ parameter |
| `:norm kind` | `l1` | `l1`, `l2`, `bandpass` |
| `:precision p` | `f32` | `f32` or `f64` |
| `:lowpass off` | on | Omit the residual lowpass row |
| `:allow-untiled` | — | Permit LP certification to fail |

Inverse (`icwt`) accepts the same keywords plus `:inverse dual|l1` (default `dual`).

```scheme
(cwt :n 4096 :wavelet morse)
(cwt :n 1024 :wavelet morlet :mu 6.0 :voices 12 :precision f64)
(icwt :n 4096 :inverse dual)
```

#### `(cwt :n N …)` / `(icwt :n N …)`

Continuous Wavelet Transform (Fourier-domain CQT filter bank). **Standalone
only** — cannot appear inside a `(pipeline …)`.

Keywords:

| Keyword | Default | Description |
|---------|---------|-------------|
| `:n N` | *required* | Signal length (even, 5-smooth) |
| `:fs F` | `1.0` | Sample rate |
| `:wavelet name` | `morse` | `morlet`, `morse`, `bump`, `shannon`, `meyer` |
| `:voices V` | `10` | Voices per octave |
| `:fmin F` | auto | Lowest center frequency |
| `:fmax F` | auto | Highest center frequency |
| `:gamma G` | `3.0` | Morse γ parameter |
| `:beta B` | `20.0` | Morse β (Lilly–Olhede). MATLAB TimeBandwidth=60 is β=20, not 60. |
| `:mu M` | `6.0` | Morlet μ parameter |
| `:norm kind` | `l1` | `l1`, `l2`, `bandpass` |
| `:precision p` | `f32` | `f32` or `f64` |
| `:lowpass off` | on | Omit the residual lowpass row |
| `:allow-untiled` | — | Permit LP certification to fail |

Inverse (`icwt`) accepts the same keywords plus `:inverse dual|l1` (default `dual`).

```scheme
(cwt :n 4096 :wavelet morse)
(cwt :n 1024 :wavelet morlet :mu 6.0 :voices 12 :precision f64)
(icwt :n 4096 :inverse dual)
```

#### `(cwt :n N …)` / `(icwt :n N …)`

Continuous Wavelet Transform (Fourier-domain CQT filter bank). **Standalone
only** — cannot appear inside a `(pipeline …)`.

Keywords:

| Keyword | Default | Description |
|---------|---------|-------------|
| `:n N` | *required* | Signal length (even, 5-smooth) |
| `:fs F` | `1.0` | Sample rate |
| `:wavelet name` | `morse` | `morlet`, `morse`, `bump`, `shannon`, `meyer` |
| `:voices V` | `10` | Voices per octave |
| `:fmin F` | auto | Lowest center frequency |
| `:fmax F` | auto | Highest center frequency |
| `:gamma G` | `3.0` | Morse γ parameter |
| `:beta B` | `20.0` | Morse β (Lilly–Olhede). MATLAB TimeBandwidth=60 is β=20, not 60. |
| `:mu M` | `6.0` | Morlet μ parameter |
| `:norm kind` | `l1` | `l1`, `l2`, `bandpass` |
| `:precision p` | `f32` | `f32` or `f64` |
| `:lowpass off` | on | Omit the residual lowpass row |
| `:allow-untiled` | — | Permit LP certification to fail |

Inverse (`icwt`) accepts the same keywords plus `:inverse dual|l1` (default `dual`).

```scheme
(cwt :n 4096 :wavelet morse)
(cwt :n 1024 :wavelet morlet :mu 6.0 :voices 12 :precision f64)
(icwt :n 4096 :inverse dual)
```

#### `(threshold :mode soft|hard :lambda λ)`

Apply a hard or soft threshold to **detail** coefficients (everything after the coarsest `N / 2^L` approximation).

```scheme
(pipeline
  (dwt :family haar :size 512 :levels 4)
  (threshold :mode soft :lambda 0.15)
  (idwt :family haar :size 512 :levels 4))
```

#### `(bfly N)`

Radix-N butterfly operation.

- `N` - Radix (2, 4, or 8)

```scheme
(bfly 2)     ; Radix-2 butterfly
(bfly 4)     ; Radix-4 butterfly
(bfly 8)     ; Radix-8 butterfly
```

#### `(lift :predict fn :update fn)`

If `fn` names a known family (`haar`, `d4`, `cdf53`, `cdf97`, `sym4`, or the
legacy aliases `haar_predict` / `predict_linear` / `predict_cubic`), Chirp
emits a real `DWT_STAGE`. It does **not** stuff a builtin index into the
coefficient-based `LIFT_PRED` opcode.

```scheme
(lift :predict haar :update haar)
(lift :predict cdf97 :update cdf97)
```

#### `(custom name)` *(legacy syntax)*

Explicit wrapper for custom functions. Equivalent to bare `name` in first-class syntax.

```scheme
(custom sigmoid)     ; Old syntax
sigmoid              ; New first-class syntax (preferred)
```

### Bare Symbols

#### `twiddle`

Twiddle factor multiplication (complex rotation).

```scheme
(pipeline (fft :size 1024) twiddle)
```

#### Reduction Operations

```scheme
reduce-sum       ; Sum all elements
reduce-max       ; Maximum element
reduce-min       ; Minimum element
```

---

## Builtin Functions

All builtins come in f32 and f64 variants. The f32 variant has `_f32` suffix, f64 has `_f64`.

### Trigonometric Functions

| Function | Description |
|----------|-------------|
| `sin` | Sine |
| `cos` | Cosine |
| `tan` | Tangent |
| `asin` | Arcsine |
| `acos` | Arccosine |
| `atan` | Arctangent |
| `atan2` | Two-argument arctangent (y, x) |
| `sinh` | Hyperbolic sine |
| `cosh` | Hyperbolic cosine |
| `tanh` | Hyperbolic tangent |

```c
float y = chirp_sin_f32(x);
double y = chirp_sin_f64(x);
```

### Exponential and Logarithmic

| Function | Description |
|----------|-------------|
| `exp` | Exponential (e^x) |
| `exp2` | Base-2 exponential (2^x) |
| `log` | Natural logarithm |
| `log2` | Base-2 logarithm |
| `log10` | Base-10 logarithm |
| `pow` | Power (x^y) |
| `sqrt` | Square root |
| `cbrt` | Cube root |
| `hypot` | Hypotenuse (sqrt(x²+y²)) |

### Special Functions

| Function | Description |
|----------|-------------|
| `erf` | Error function |
| `erfc` | Complementary error function |
| `tgamma` | Gamma function |
| `lgamma` | Log-gamma function |
| `j0` | Bessel function of first kind (order 0) |
| `j1` | Bessel function of first kind (order 1) |
| `y0` | Bessel function of second kind (order 0) |
| `y1` | Bessel function of second kind (order 1) |

### Probability Distributions (PDFs)

All PDFs take parameters specific to each distribution:

```c
float y = chirp_gaussian_pdf_f32(x, mu, sigma);
float y = chirp_cauchy_pdf_f32(x, x0, gamma);
```

| Function | Parameters | Description |
|----------|------------|-------------|
| `gaussian_pdf` | `mu, sigma` | Normal distribution |
| `cauchy_pdf` | `x0, gamma` | Cauchy/Lorentzian |
| `exponential_pdf` | `lambda` | Exponential |
| `lognormal_pdf` | `mu, sigma` | Log-normal |
| `laplace_pdf` | `mu, b` | Laplace/double exponential |
| `beta_pdf` | `alpha, beta` | Beta distribution |
| `gamma_pdf` | `k, theta` | Gamma distribution |
| `weibull_pdf` | `k, lambda` | Weibull distribution |
| `uniform_pdf` | `a, b` | Uniform distribution |
| `student_t_pdf` | `nu` | Student's t-distribution |

### Activation Functions

| Function | Description |
|----------|-------------|
| `sigmoid` | 1/(1+e^(-x)) |
| `relu` | max(0, x) |
| `gelu` | x * Φ(x) where Φ is normal CDF |
| `swish` | x * sigmoid(x) |
| `softmax` | Vector normalization to probabilities |

### Vector Operations

Vector functions take arrays and length:

```c
float sum = chirp_sum_f32(vec, n);
float dot = chirp_dot_f32(a, b, n);
```

| Function | Description |
|----------|-------------|
| `sum` | Sum of elements |
| `dot` | Dot product of two vectors |
| `norm` | L2 norm (sqrt(sum(x²))) |
| `norm1` | L1 norm (sum(|x|)) |
| `norm_inf` | Infinity norm (max(|x|)) |
| `mean` | Arithmetic mean |
| `variance` | Variance |
| `std` | Standard deviation |
| `cumsum` | Cumulative sum (prefix sum) |
| `mul` | Element-wise multiplication |
| `add` | Element-wise addition |
| `sub` | Element-wise subtraction |
| `div` | Element-wise division |
| `scale` | Multiply by scalar |
| `saxpy` | y = alpha*x + y (BLAS style) |

### Window Functions

Generate window coefficients:

```c
float window[1024];
chirp_hann_window_f32(window, 1024);
```

| Function | Description |
|----------|-------------|
| `hann_window` | Hann window (cosine, zero at edges) |
| `hamming_window` | Hamming window (non-zero at edges) |
| `blackman_window` | Blackman window (better sidelobe rejection) |
| `flattop_window` | Flat-top window (low amplitude error) |
| `gaussian_window` | Gaussian window (sigma parameter) |
| `kaiser_window` | Kaiser window (beta parameter) |

### Wavelet Functions

#### Direct Wavelet Transforms

```c
float approx[512], detail[512];
chirp_haar_f32(signal, approx, detail, 1024);
```

| Function | Description |
|----------|-------------|
| `haar` | Compiles to a Haar `DWT_STAGE` |
| `d4` / `daubechies4` | Daubechies D4 (db2) |
| `cdf53` / `cdf97` / `sym4` | JPEG 2000 and Symlet families |
| `morlet` | Morlet wavelet (CWT kernel, not a DWT) |
| `mexican_hat` | Mexican hat (Ricker) wavelet |

C helpers `chirp_haar_f32` / `chirp_daubechies4_f32` implement one-level
orthonormal analysis (Haar uses `1/√2`; D4 uses the 4-tap QMF).

#### Wavelet Composition via Lifting

```scheme
(pipeline
  (lift :predict haar :update haar)
  (threshold :mode soft :lambda 0.1)
  (idwt :family haar :size 256 :levels 1))
```

### Utility Functions

| Function | Description |
|----------|-------------|
| `clamp` | Clamp to range [min, max] |
| `lerp` | Linear interpolation |
| `smoothstep` | Smooth Hermite interpolation |
| `smootherstep` | Higher-order smooth interpolation |
| `step` | Heaviside step function |
| `sign` | Sign function (-1, 0, 1) |
| `deg2rad` | Degrees to radians |
| `rad2deg` | Radians to degrees |

---

## Examples

### Example 1: Basic FFT Pipeline

```scheme
; Simple FFT with twiddle factors
(pipeline
  (fft :size 1024)
  twiddle)
```

### Example 2: Neural Activation Chain

```scheme
; Apply multiple activation functions
(pipeline
  (fft :size 512)
  sigmoid      ; First-class function syntax
  gelu
  relu
  reduce-sum)
```

### Example 3: Distribution-Based Filtering

```scheme
; Use probability distributions in pipeline
(pipeline
  (fft :size 2048)
  (lift :predict gaussian_pdf :update laplace_pdf)
  sigmoid
  reduce-max)
```

### Example 4: Window + FFT

```scheme
; Apply window function before FFT
(pipeline
  hann_window     ; First-class
  (fft :size 4096)
  twiddle)
```

### Example 5: Custom Wavelet via Lifting

```c
// Register custom lifting primitives
chirp_register("my_predict", (void*)my_predict_impl);
chirp_register("my_update", (void*)my_update_impl);
```

```scheme
; Compose into wavelet
(pipeline
  (lift :predict my_predict :update my_update)
  scale_sqrt2
  sigmoid)
```

### Example 6: Multi-Stage Butterfly

```scheme
; Radix-2/4/8 cascade
(pipeline
  (bfly 2)
  (bfly 4)
  (bfly 8)
  reduce-max)
```

### Example 7: Vector Statistics

```c
// C API usage of vector functions
float data[1024];
// ... fill data ...

float mean = chirp_mean_f32(data, 1024);
float std = chirp_std_f32(data, 1024);
float sum = chirp_sum_f32(data, 1024);

// SAXPY: y = alpha*x + y
chirp_saxpy_f32(2.0f, x, y, result, 1024);
```

---

## C API

### Registration

```c
#include <chirp.h>

// Register a single custom function
int chirp_register(const char *name, void (*fn)(void));

// Register all 140+ standard builtins
int chirp_register_standard_builtins(void);

// List all registered builtins
void chirp_list_builtins(void);

// Cleanup: free all registered builtin names
void chirp_cleanup(void);
```

### Compilation

```c
// Compile Chirp source to transform
faf_transform *chirp_compile(const char *source);
```

Returns `NULL` on parse error. The returned transform can be executed with `faf_execute_jit()`.

### Builtin Function Signatures

When registering custom functions, match these signatures:

```c
// Scalar unary: y = f(x)
float my_func_f32(float x);
double my_func_f64(double x);

// Scalar binary: z = f(x, y)
float my_func_f32(float x, float y);

// Vector: out = f(in, n)
void my_vec_f32(const float *in, float *out, size_t n);

// In-place vector: vec = f(vec, n)
void my_inplace_f32(float *vec, size_t n);
```

---

## Advanced Topics

### Custom Lifting Primitives

The lifting scheme decomposes wavelets into three steps:

1. **Split**: Even/odd separation
2. **Predict**: Predict odd from even (produces detail coefficients)
3. **Update**: Update even using detail (produces approximation)

Implement custom predict/update:

```c
// Prediction: compute detail = odd - predict(even)
void my_predict(float *even, float *odd, float *detail, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // Linear interpolation example
        float pred = 0.5f * (even[i] + even[i+1]);
        detail[i] = odd[i] - pred;
    }
}

// Update: compute approx = even + update(detail)
void my_update(float *even, float *detail, float *approx, size_t n) {
    for (size_t i = 0; i < n; i++) {
        approx[i] = even[i] + 0.25f * detail[i];
    }
}
```

Register and use:

```c
chirp_register("my_predict", (void*)my_predict);
chirp_register("my_update", (void*)my_update);
```

```scheme
(lift :predict my_predict :update my_update)
```

### Precision Selection

The precision of generated IR is determined by the transform's `precision` field:

```c
faf_transform *t = chirp_compile("(fft :size 1024)");
t->precision = FAF_PREC_FP32;  // Use f32 variants
t->precision = FAF_PREC_FP64;  // Use f64 variants
```

### Error Handling

`chirp_compile()` returns `NULL` on parse errors. Check stderr for messages:

```c
faf_transform *t = chirp_compile(bad_source);
if (!t) {
    fprintf(stderr, "Compilation failed\n");
    // Error message already printed by Chirp
}
```

### Memory Management

Transforms created by `chirp_compile()` must be freed:

```c
faf_transform *t = chirp_compile(source);
// ... use t ...
faf_destroy_transform(t);
```

Builtin registration persists for the lifetime of the program.

---

## Implementation Notes

### Register File

- Maximum 256 builtins (expandable by recompiling)
- Names are case-sensitive
- Duplicate registrations update the existing entry

### IR Generation

Chirp compiles to FastAndFourier IR opcodes:

| Chirp Form | IR Opcode |
|------------|-----------|
| `(fft :size N)` | `FAF_FFT_STAGE` |
| `(bfly N)` | `FAF_BFLY2/4/8` |
| `twiddle` | `FAF_TWIDDLE_MUL` |
| `(lift ...)` | `FAF_LIFT_PRED/UPD` |
| `function` | `FAF_CALL_BUILTIN` |
| `reduce-*` | `FAF_REDUCE_*` |

### Performance

- Builtin calls are dispatched via function pointer
- Small functions may be inlined by the JIT compiler
- Vector operations use standard C loops (compiler can auto-vectorize)

---

## See Also

- [README.md](README.md) - Main project documentation
- [examples/chirp_builtins_example.c](examples/chirp_builtins_example.c) - Comprehensive builtin demo
- [examples/chirp_wavelet_dsl.c](examples/chirp_wavelet_dsl.c) - Lifting scheme compositions
- [include/chirp.h](include/chirp.h) - Public C API
- [include/chirp_builtins.h](include/chirp_builtins.h) - Builtin function declarations
