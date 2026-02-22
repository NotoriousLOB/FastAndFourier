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

```c
#include <chirp.h>
#include <chirp_builtins.h>

// Register all 140+ standard builtins
chirp_register_standard_builtins();

// Compile a Chirp program
dspir_transform *t = chirp_compile(
    "(pipeline "
    "  (fft :size 1024) "
    "  hann_window "
    "  sigmoid "
    "  reduce-sum)"
);

// Execute
dspir_execute_jit(t, output, input);
dspir_destroy_transform(t);
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

#### `(pipeline expr1 expr2 ...)`

Sequences multiple operations. Returns a transform that executes each expression in order.

```scheme
(pipeline
  (fft :size 1024)
  twiddle
  (bfly 4)
  reduce-sum)
```

#### `(fft :size N)`

Fast Fourier Transform.

- `:size N` - Transform size (must be power of 2)

```scheme
(fft :size 256)      ; 256-point FFT
(fft :size 4096)     ; 4096-point FFT
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

Lifting scheme step for wavelet transforms.

- `:predict fn` - Prediction function (registered builtin)
- `:update fn` - Update function (registered builtin)

```scheme
(lift :predict haar_predict :update haar_update)
(lift :predict gaussian_pdf :update laplace_pdf)
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
| `haar` | Haar wavelet transform |
| `haar_inverse` | Inverse Haar transform |
| `daubechies4` | Daubechies-4 wavelet |
| `morlet` | Morlet wavelet (complex) |
| `mexican_hat` | Mexican hat (Ricker) wavelet |
| `meyer_scaling` | Meyer scaling function |
| `shannon_wavelet` | Shannon wavelet (sinc-based) |

#### Wavelet Composition via Lifting

Wavelets can be composed using the lifting scheme:

```scheme
; Haar via lifting
(pipeline
  (lift :predict haar_predict :update haar_update)
  scale_sqrt2)

; Custom wavelet
(pipeline
  (lift :predict my_predict :update my_update)
  sigmoid
  scale_sqrt2)
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
dspir_transform *chirp_compile(const char *source);
```

Returns `NULL` on parse error. The returned transform can be executed with `dspir_execute_jit()`.

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
dspir_transform *t = chirp_compile("(fft :size 1024)");
t->precision = DSPIR_PREC_FP32;  // Use f32 variants
t->precision = DSPIR_PREC_FP64;  // Use f64 variants
```

### Error Handling

`chirp_compile()` returns `NULL` on parse errors. Check stderr for messages:

```c
dspir_transform *t = chirp_compile(bad_source);
if (!t) {
    fprintf(stderr, "Compilation failed\n");
    // Error message already printed by Chirp
}
```

### Memory Management

Transforms created by `chirp_compile()` must be freed:

```c
dspir_transform *t = chirp_compile(source);
// ... use t ...
dspir_destroy_transform(t);
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
| `(fft :size N)` | `DSPIR_FFT_STAGE` |
| `(bfly N)` | `DSPIR_BFLY2/4/8` |
| `twiddle` | `DSPIR_TWIDDLE_MUL` |
| `(lift ...)` | `DSPIR_LIFT_PRED/UPD` |
| `function` | `DSPIR_CALL_BUILTIN` |
| `reduce-*` | `DSPIR_REDUCE_*` |

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
