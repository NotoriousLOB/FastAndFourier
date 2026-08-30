# Continuous Wavelet Transform (CWT)

A Fourier-domain CQT/CWT filter bank with Littlewood–Paley certification,
dual-frame and L1 one-integral inverses, and SIMD-accelerated execution.

The design follows *Filter Banks That Tell the Truth* (working paper, 2026):
a bank is honest if and only if its Littlewood–Paley (LP) sum is essentially
constant. Geometric voices, analytic zero-phase prototypes, a residual
lowpass for the DC hole, and a reconstruction test are not optional garnish —
they are the theorem, typed in.

## How it differs from the DWT

The DWT (Haar, CDF 9/7, etc.) uses a lifting scheme in the time domain with
critically sampled octave bands. The CWT filter bank samples analytic wavelets
on the DFT grid, producing a redundant scalogram with arbitrary frequency
resolution. It is the right tool for time-frequency analysis at sub-octave
detail; the DWT is the right tool for compression and denoising at minimal cost.

## Algorithm

Given real input `x[n]`:

1. **RFFT** — compute `X = RFFT(x)`, yielding `n/2+1` Hermitian bins.
2. **Per-scale multiply** — for each row `j`, pointwise-multiply `X` by the
   precomputed wavelet filter `Ψ[j]` (real, zero-phase, stored on the
   Hermitian grid; DC and Nyquist bins of every wavelet are zero).
3. **IRFFT** — inverse-transform each product to get the real-valued
   coefficients `W[j, :]`. For a zero-phase analytic prototype this is
   exactly `2 Re(ψ_j ∗ x)` under the library's unnormalized RFFT convention.

The inner RFFT/IRFFT pair is created once at bank-creation time and reused
across all scales and all calls to `faf_execute`.

## Size constraints

`n` must be **even** and **5-smooth** (`2^a · 3^b · 5^c`). Use
`faf_is_size_supported(FAF_TRANSFORM_CWT, n)` or
`faf_get_recommended_size(FAF_TRANSFORM_CWT, n)`.

## Wavelet families

| Name | `faf_cwt_wavelet` | Parameters | Notes |
|------|-------------------|------------|-------|
| Morse | `FAF_CWT_WAVELET_MORSE` | `morse_gamma`, `morse_beta` | Default. γ=3, β=20 (MATLAB `TimeBandwidth=60`, P²=βγ). Log-domain evaluation. |
| Morlet | `FAF_CWT_WAVELET_MORLET` | `morlet_mu` | Analytic Morlet with zero-mean correction (μ=6). |
| Bump | `FAF_CWT_WAVELET_BUMP` | `bump_center`, `bump_width` | Compact-support bump. |
| Shannon | `FAF_CWT_WAVELET_SHANNON` | — | Ideal one-octave bandpass. |
| Meyer | `FAF_CWT_WAVELET_MEYER` | — | Zero-phase analytic Meyer (auxiliary ν). |

All prototypes are **analytic** (zero for ω ≤ 0), **zero-mean** (DC bin = 0),
and **peak-normalized** in the frequency domain. Nyquist is left unused
(the analytic projector on a length-`n` grid).

Morse β is the Lilly–Olhede exponent, **not** MATLAB's TimeBandwidth. The
MATLAB default `TimeBandwidth=60` with γ=3 is β = 60/3 = 20. Passing
`morse_beta=60` is a valid narrower wavelet (P²=180); it is not the default.

## Scale geometry

With `FAF_CWT_SCALE_GEOMETRIC` (default), center frequencies follow
`f_{j+1} / f_j = 2^{1/V}` where `V = voices`. This produces a CQT
(Constant-Q Transform) tiling of the frequency axis. Linear scale grids do
not tile in log-frequency — that is Proposition 4 of the paper, not a taste.

`FAF_CWT_SCALE_LINEAR` is available but does not tile geometrically. The
LP sum will not certify, and `FAF_CWT_FLAG_ALLOW_UNTILED` is required.

If `fmin` is left at 0, it is set to `16 · fs / n` (sixteen cycles in the
record). That is the duration cap of §7.4: a coarser wavelet wraps the
length-`n` torus and is not the prototype you plotted. The residual
lowpass covers everything below that floor. A user-supplied `fmin` that
still wraps fails create under `VALIDATE_STRICT`.

`faf_cwt_center_kind` selects how a scale is labelled: peak frequency of
`ψ̂` (default) or the energy centroid of `|ψ̂|²`. Morse wavelets with
large β have mass to the left of the peak; quote which one you used when
you label a scalogram axis.

## Normalization

| `faf_cwt_norm_kind` | Filter scaling | Use case |
|---------------------|----------------|----------|
| `FAF_CWT_NORM_L1` | L1 dilation `ψ̂_s(ξ) = ψ̂(sξ)` | Amplitude-preserving scalograms; default; matches the one-integral inverse |
| `FAF_CWT_NORM_L2` | Global energy correction so Φ ≈ 1 | Plancherel / tight-frame energy |
| `FAF_CWT_NORM_BANDPASS` | Peak = `bandpass_peak` (default 2) | MATLAB-style display gain |

L1 vs L2 is not cosmetic. Mixing them inside one bank and then comparing
scales produces a figure that is simultaneously correct and meaningless.
Dual-frame inversion is built from the **live** analysis filters, so it
reconstructs for every kind.

## Littlewood-Paley certification

The bank computes the L2-shadow LP sum `A[k] = |φ[k]|² + Σ_j |Ψ_L2[j,k]|²`
and requires `max|A[k] − 1| ≤ lp_alpha` (default 0.05) **and**
`mean|A[k] − 1| ≤ lp_beta` over the certified band. If
`FAF_CWT_FLAG_VALIDATE_STRICT` is set (default) and certification fails,
`faf_create_cwt` returns `NULL`.

The `faf_cwt_lp_report` struct (accessed via `faf_cwt_bank_report`) contains:

- `passed` — 1 if certified
- `max_abs_dev`, `mean_abs_dev` — deviation from 1 on `[k_lo, k_hi]`
- `min_A`, `max_A`, `frame_cond` — range and `max/min` condition number
- `hole_bins_dc`, `hole_bins_nyq` — bins outside the tiled band
- `max_dc_wavelet` — largest wavelet magnitude at DC (should be ~0)
- `max_wrap_energy` — fraction of coarsest-kernel energy near `t = n/2`
- `admissibility_C` — `∫ ψ̂(ω)/ω dω` for the L1 inverse
- `A` — pointer to the LP-sum array, valid until destroy

A condition number above about 1.2 on the analysis band is a bank you
should redesign before you publish a scalogram.

## Residual lowpass

When `FAF_CWT_FLAG_INCLUDE_LOWPASS` is set (default), the bank adds a
lowpass **row 0**: `φ[k] = sqrt(max(0, 1 − Σ_j |Ψ_L2[j,k]|²))`. This
absorbs energy below the lowest wavelet (Theorem 3: no finite collection
of wavelets can see DC) and is required for dual-frame reconstruction of
signals that contain a trend. Wavelet scales then run coarse → fine in
rows `1 .. n_scales`.

Create also rejects a bank whose coarsest wavelet wraps around the length-`n`
torus (`max_wrap_energy > 1e-3`) unless `FAF_CWT_FLAG_ALLOW_UNTILED` is set.

## Inverse transforms

`faf_create_inverse(cwt)` always builds the dual-frame inverse.

### Dual-frame inverse

Let `Ψ` be the live analysis filters. The dual is
`Ψ_dual[j,k] = Ψ[j,k] / (Σ_r |Ψ[r,k]|² + ε)`, then:

1. RFFT each coefficient row `W[j]`
2. Accumulate `Σ_j Y_j[k] · Ψ_dual[j,k]`
3. IRFFT the accumulator

Reconstruction is exact to floating-point precision when the LP sum of the
live filters has no zeros on the analysis band (~1e-12 relative L2 error
in FP64 for a certified L2 bank; similarly tight for L1 + dual).

### L1 one-integral inverse

```
x̂[t] = Σ_j W[j,t] · d_log / C_ψ  +  (φ ∗ x)[t]
```

where `d_log = log(2)/V` and `C_ψ` is the admissibility constant
`∫_0^∞ ψ̂(ω)/ω dω`. This is a discretized Calderón one-integral — looser
than dual-frame. It requires an L1-normalized bank. Do not call it
perfect reconstruction.

## Memory layout

Forward output is a 2D real array of shape `[n_rows × n]`, stored row-major
in a flat `float*` (or `double*`). Row `j` starts at `out + j * n`.

- `faf_cwt_n_scales(t)` — number of wavelet scales `J`
- `faf_cwt_n_rows(t)` — `J + has_lowpass` (total rows in the output)
- `faf_cwt_n_bins(t)` — packed Hermitian length `n/2+1`
- `faf_cwt_has_lowpass(t)` — 1 if row 0 is the residual lowpass
- `faf_cwt_psi_f32/f64(t, scale_index)` — wavelet row, `scale_index=0`
  is the coarsest wavelet (not the lowpass). The pointer is valid for
  `n_bins` values; rows are internally padded, so do not stride with
  `n/2+1`.

## Quick start

```c
faf_cwt_config cfg = faf_cwt_config_init(4096);
cfg.wavelet = FAF_CWT_WAVELET_MORSE;
cfg.voices = 10;

faf_transform *cwt = faf_create_cwt(&cfg);
faf_transform *icwt = faf_create_inverse(cwt);   /* dual-frame, any norm */

float *x   = faf_aligned_alloc(4096 * sizeof(float));
float *W   = faf_aligned_alloc(faf_cwt_n_rows(cwt) * 4096 * sizeof(float));
float *out = faf_aligned_alloc(4096 * sizeof(float));

faf_buffer in_buf  = faf_buffer_real(x, 4096);
faf_buffer w_buf   = faf_buffer_real(W, 4096);
faf_buffer out_buf = faf_buffer_real(out, 4096);

faf_execute(cwt, &w_buf, &in_buf);    /* forward */
faf_execute(icwt, &out_buf, &w_buf);  /* inverse */

faf_destroy_transform(icwt);
faf_destroy_transform(cwt);
```

## Chirp DSL

CWT is available as a standalone Chirp form. It **cannot** appear inside a
pipeline — `(pipeline (cwt …) …)` is a compile error.

```scheme
(cwt :n 4096 :wavelet morse)
(cwt :n 1024 :wavelet morlet :mu 6.0 :voices 12 :precision f64)
(icwt :n 4096 :inverse dual)
```

See [CHIRP.md](../CHIRP.md) for the full keyword reference.

## SIMD acceleration

The per-scale Hermitian multiply is vectorized:

- **AArch64 NEON**: 4-wide FP32, 2-wide FP64 with FMA
- **x86_64 AVX2**: 8-wide FP32, 4-wide FP64 with FMA when `__FMA__` is set

Dispatch is compile-time via `#ifdef`. Scalar fallback is always available.

## Thread safety

A `faf_transform` created by `faf_create_cwt` is **not** thread-safe for
concurrent `faf_execute` calls — the transform owns internal scratch buffers
that are reused across calls. Create one transform per thread, or serialize
access with a mutex.

## Examples

- [`examples/cwt_filterbank.c`](../examples/cwt_filterbank.c) — create a bank, print the LP report
- [`examples/cwt_reconstruct.c`](../examples/cwt_reconstruct.c) — forward + dual inverse reconstruction
