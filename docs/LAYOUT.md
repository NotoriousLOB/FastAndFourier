# Layouts

Four layouts. The library never converts for you.

| Layout | Storage | Default for |
|--------|---------|-------------|
| `FAF_LAYOUT_SPLIT` | `re[n]`, `im[n]` — two planes | C2C FFT |
| `FAF_LAYOUT_HERMITIAN` | `re[n/2+1]`, `im[n/2+1]` packed R2C | RFFT spectrum |
| `FAF_LAYOUT_REAL` | `re[n]`, `im == NULL` | RFFT time, DWT, DCT |
| `FAF_LAYOUT_INTERLEAVED` | `float[2n]` as `re,im,re,im,…` | opt-in convenience |

`faf_execute` rejects a mismatched buffer. Convert at the edge with
`faf_deinterleave_f32` / `faf_interleave_f32`. JIT kernels flagged
`FAF_FLAG_JIT_SPLIT_PLANE` never see interleaved pointers.

## Why split / Hermitian is the fast path

A complex multiply, a conjugate, a Hermitian multiply, and a CWT row
are all strided as `re[k]`, `im[k]`. Two contiguous planes sit in L1/L2
as two streams. Interleaved pairs force a shuffle every load: the
`re,im` pair is the wrong unit of work for AVX2/NEON, and a correlator
that unpacks into `complex float` then complains about cache paid the
tax twice.

Hermitian split is the same idea on half the bins. DC and Nyquist are
real: `im[0] == im[n/2] == 0` within rounding. Do not store a C2C of a
real signal with the imag plane zeroed and call it an RFFT.

The layout tax is the `BM_RFFT_Layout` benchmark (3840-point Hermitian
vs interleaved) in `benchmarks/benchmark_fft.cpp`. C2C split vs
interleaved is `BM_SplitPlane_*` / `BM_Standard_*`.

## Packed Hermitian multiply

Real correlation on the packed grid:

```
bins 1 .. n/2-1 : complex multiply   y = x * h   (or x * conj(h))
bin 0 and n/2   : real multiply, imag stays 0
```

`faf_herm_mul_f32` / `faf_herm_mul_conj_f32` implement that. Test DC
first, then Nyquist, then a tone at bin `k`. The split-plane correlator
in [`examples/split_corr.c`](../examples/split_corr.c) is the worked
example; it also registers `chirp_register_spectral_ex("corr", …)` so
the same kernel can sit in

```scheme
(pipeline (rfft :size 128 :layout hermitian) (spectral corr) (irfft))
```

The spectral builtin signature is `void(re, im, n_bins, ctx)` on the
Hermitian planes. It is not an interleaved `complex float *`.

## What not to do

Do not store an analytic CWT row with a zeroed imaginary plane in
`FAF_LAYOUT_INTERLEAVED`. If a row is real, it is `FAF_LAYOUT_REAL`.
Zero imag in interleaved storage is a cache-line of zeros you will
load on every scan, and it pretends the data is complex.

README examples are split / Hermitian only. Interleaved is for talking
to libraries that will not take two pointers.
