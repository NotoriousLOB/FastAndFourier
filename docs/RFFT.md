# Real FFT (R2C / C2R)

A first-class real-to-complex transform. This is **not** a C2C of a real
signal with a zeroed imaginary plane.

## Why it exists

A length-`n` real sequence has a Hermitian spectrum: `X[n-k] = conj(X[k])`.
Only `n/2+1` complex bins are independent (DC through Nyquist). Computing a
full C2C wastes that symmetry; packing even/odd samples into an `n/2` complex
FFT and applying a super-twiddle post-pass is the standard ~2× win.

## Algorithm

Let `n` be even (this cut: power of 2, `n >= 2`).

1. Pack `z[k] = x[2k] + i x[2k+1]` for `k = 0 .. n/2-1`.
2. Run the existing power-of-2 C2C of length `n/2` (split-plane, unscaled).
3. Kiss / Sorensen post-pass with super-twiddles
   `exp(±i π ((k+1)/(n/2) + 1/2))` to produce packed `X[0 .. n/2]`.

Inverse is the reverse: pre-pass, `n/2` IFFT, unpack even/odd. The inner C2C
is **not** `1/(n/2)`-scaled; `faf_execute` applies the outer norm using the
**real** length `n`.

`n` must be even and 5-smooth (`2^a 3^b 5^c`). Odd `n` is still rejected.

## Layouts and who owns what

Default `cfg.layout` is `FAF_LAYOUT_HERMITIAN`. Time is always real.

| Direction | Input | Output |
|-----------|--------|--------|
| Forward (`FAF_TRANSFORM_RFFT`) | `FAF_LAYOUT_REAL`, length `n` | Hermitian `n/2+1` |
| Inverse (`FAF_TRANSFORM_IRFFT`) | Hermitian `n/2+1` | `FAF_LAYOUT_REAL`, length `n` |

Hermitian storage is split-plane: `re[n/2+1]`, `im[n/2+1]`. The caller owns
every buffer; the transform owns scratch and the nested `n/2` C2C.

`cfg.layout = FAF_LAYOUT_INTERLEAVED` packs the spectrum as
`float[2*(n/2+1)]` (`re,im,re,im,...`). Time is still `REAL`. There is no
silent convert: `faf_execute` rejects a mismatched buffer.

```c
faf_config cfg = faf_config_init(4096);
faf_transform *fwd = faf_create_rfft(&cfg);
faf_transform *inv = faf_create_inverse(fwd);

size_t nb = faf_spectrum_len(fwd); /* 2049 */
faf_buffer in   = faf_buffer_real(x, 4096);
faf_buffer spec = faf_buffer_hermitian(Xr, Xi, nb);
faf_buffer out  = faf_buffer_real(y, 4096);
faf_execute(fwd, &spec, &in);
faf_execute(inv, &out, &spec);
```

Chirp:

```scheme
(rfft :size 4096 :norm none :layout hermitian)
(irfft :size 4096)
(fft :size 1024 :layout split)
```

Interleaved is an opt-in convenience. Convert at the edge with
`faf_deinterleave_f32` / `faf_interleave_f32`; `faf_execute` will not.

A fused `(pipeline (rfft) … (irfft))` is a later phase (Fourier-domain C).
Compile the two forms separately for now.

## DC and Nyquist

`X[0]` (DC) and `X[n/2]` (Nyquist) are real: `im[0] == im[n/2] == 0` within
rounding. Bins `1 .. n/2-1` are complex. Reconstructing the full-length
spectrum is `X[n-k] = conj(X[k])`; you do not need to store it.

## Normalization

Same enum as C2C, using the **real** length `n`:

| `faf_norm` | Forward | Inverse |
|------------|---------|---------|
| `NONE` (default) | unscaled | `1/n` |
| `ORTHO` | `1/√n` | `1/√n` |
| `FORWARD` | `1/n` | unscaled |

`LAZY` / `JPEG2000` are rejected at create.

## Inverse without retyping

`faf_create_inverse(fwd)` (or `faf_config_inverse` + `faf_create_rfft`)
flips `dir` and keeps `n`, precision, layout, norm, backend.

## What this is not

- Not a C2C with `im[:] = 0`. Compare a few bins against our own C2C if you
  want a reference; magnitudes in `0 .. n/2` should match.
- Not Bluestein / arbitrary `n`. Size must be even and 5-smooth.
- Not a framed STFT. That stays a later transform.
