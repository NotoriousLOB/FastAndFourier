# Sizes

Size is a create-time contract. `faf_execute` never pads.

## Default legal lengths

| Transform | Legal `n` |
|-----------|-----------|
| FFT C2C | 5-smooth, `n ≥ 1` (`2^a · 3^b · 5^c`) |
| RFFT / IRFFT | even 5-smooth, `n ≥ 2` |
| CWT / ICWT | even 5-smooth (rides RFFT) |
| DWT (both backends), dyadic | power of two, `n ≥ 2` |
| DCT / DST | unrestricted (already) |
| MDCT | even |

5-smooth means the mixed-radix 2/3/4/5 kernels can factor `n`. That is
the physics of the buffer, not a power-of-two religion. `3840`
(`2^8 · 3 · 5`), `4000` (`2^5 · 5^3`), and `4096` (`2^12`) are all
legal FFT sizes. `4097` is not, unless Bluestein is on.

Helpers:

```c
faf_is_size_supported(FAF_TRANSFORM_FFT, n);
faf_get_recommended_size(FAF_TRANSFORM_FFT, 3000); /* 3000 itself */
faf_get_recommended_size(FAF_TRANSFORM_RFFT, n_min); /* next even 5-smooth */
```

Chirp: `(fft :size 3840)` and `(rfft :size 3840)` compile. `(dwt :size
3840)` is illegal — DWT is a dyadic Mallat bank, see [DWT.md](DWT.md).

## When to set `FAF_FLAG_BLUESTEIN`

Almost never. Bluestein is a chirp-z: two length-`M` FFTs plus a
pointwise multiply, `M ≥ 2n−1`, `M` the next 5-smooth. You wanted
page-exact `3000` and `3000` is already 5-smooth. `3072` is
`2^{10} · 3`. Mixed-radix wins on any nearby 5-smooth.

Ship it for the cases mixed-radix cannot touch: prime `n`, `n = 2^a p`
with a large prime, protocol lengths that cannot move.

```c
#define FAF_FLAG_BLUESTEIN (1u << 12)

cfg.n = 307;                 /* prime */
cfg.flags |= FAF_FLAG_BLUESTEIN;
faf_transform *t = faf_create_fft(&cfg);
```

Rules:

- Without the flag, non-5-smooth → create error.
- With the flag, `M =` next 5-smooth `≥ 2n−1`. Scratch is on the
  transform (`t->scratch_size`), allocated at create, not execute.
- Chirp: `(fft :size 307 :bluestein)` explicit. No `:bluestein auto`.
- RFFT+Bluestein: refused in v1. Use C2C on a real buffer.
- DWT+Bluestein: refused. Nonsense.

Work estimate: `n = 307` → `2n−1 = 613` → `M = 625` (`5^4`). That is
two 625-point FFTs plus two chirp multiplies of length `n`, versus a
320-point mixed-radix FFT if you can pad. The `BM_FFT_Bluestein`
benchmark (307 vs 320 vs 512) is the number to cite.

There is no `BLUESTEIN` opcode. The transform lowers to the existing
inner FFT of length `M`.
