# Discrete Wavelet Transform

Two backends, one create-time contract. The operator is not "the Haar
function" and it is not "whatever `DWT_STAGE` felt like." It is a named
filter bank, a convention, a grid, and a backend.

## 1. What operator you asked for

Five fields, refused at create if they disagree:

| Field | Meaning |
|-------|---------|
| Filters | Family table, or custom `(h, g[, h̃, g̃])` |
| Convention | `faf_wavelet_convention` — Haar-ortho is not Haar-lazy |
| Moments | Implied by the convention (Haar-ortho: 1; D4/Sym4: 2) |
| Grid | Dyadic, `n` a power of two, periodic boundary (v1) |
| Backend | Lifting or polyphase FIR |

`FAF_NORM_*` is an FFT/RFFT/DCT knob. For DWT, **convention wins**. If both
are set and they disagree (`ORTHO` on `cdf53_int`), create fails.

Chirp: `(dwt :family haar :size 1024 :levels 4 :conv haar-ortho :backend auto)`.

## 2. Why lifting (Backend L)

Lifting is not a wavelet. It is a factorization of a known 2×2 polyphase
matrix into `LIFT_PRED` / `LIFT_UPD` / `LIFT_SCALE`. For the five short PR
families it is fewer MACs, in-place, and the inverse cannot drift from the
forward because there is no second coefficient table.

JPEG2000 5/3 is a lifting spec. Implementing it as convolution of the 5/3
taps is a different operator. That is why L exists, and why
`cdf53_int` + `:backend fir` is a create error.

Haar, D4, and Sym4 are short enough that FIR does not win. Cite
`BM_DWT_Backend` (Haar/D4/Sym4, L vs F) in `benchmarks/`.

Who uses L: Haar (all three conventions may lift), D4, Sym4, CDF 5/3 integer,
CDF 9/7 JPEG. Inverse = reverse steps, flipped signs, inverse scales.
`n ≥ 2^L`, periodic.

## 3. Why lifting is the wrong default elsewhere

- Sequential hazards on long factorizations; fat lifting weights.
- Live / per-frame / adaptive taps vs vanishing moments computed once.
- Published tap tables (bior6.8, db8, coif3) that were never factored here.
- Analysis-only atoms (Han-class): no dual, so there is nothing to undo.
- Undecimated / M-band / complex / dual-tree.

We will not run the Euclidean algorithm on Laurent polynomials in a create
path. Custom taps go to Backend F.

## 4. Backend F definition

Explicit analysis filters `(h, g)` and, if PR, synthesis `(h̃, g̃)`, executed
as polyphase FIR on even/odd phases. Mallat periodic convolution, even
decimation:

```
a[k] = Σ_i h[i] x[(2k + i) mod n]
d[k] = Σ_i g[i] x[(2k + i) mod n]
```

This matches the in-tree D4/Sym4 QMF kernels, **not** MATLAB
`dwt(..., 'per')` delay alignment. Impulse tests at index 0 and `n-1`
lock the convention.

Naive "convolve `n` then keep evens" is allowed in one place:
`faf_dwt_ref_conv_decim_*`, used by tests to prove Backend F matches Mallat.
It is not a JIT opcode.

Custom tap binding:

```c
int faf_dwt_set_taps(faf_transform *t,
                     const float *h, const float *g, int len_hg,
                     const float *ht, const float *gt, int len_syn);
/* ht, gt NULL ⇒ ANALYSIS_ONLY; inverse is then illegal */
```

Chirp: `(dwt :family haar :size 1024 :levels 3 :backend fir :conv haar-ortho)`.

Custom taps:

```scheme
(bind F :h h_re :g g_re :h-syn hs :g-syn gs)
(dwt :family custom :size 1024 :levels 3 :backend fir :conv custom-pr :taps F)
```

Register the vectors from C (`chirp_register_vector` / `chirp_register_taps`).
Backend F uses create-time scratch (`t->scratch`); an 8-tap analysis path
does not malloc on execute.

## 5. AUTO rule

```
if convention in {cdf53_int, cdf97_jpeg}           → L
else if convention in {custom_pr, analysis_only}   → F
else if family in {haar, d4, cdf53, cdf97, sym4}
        and built-in taps and support ≤ 8
        and not ANALYSIS_ONLY                      → L
else if family is cdf97 (9-tap JPEG pair)          → L
else                                               → F
```

Override with `:backend lift|fir`. Illegal override
(`cdf53_int` + FIR, `custom_pr` + LIFT) → create / compile error.

## 6. Inverse

Inverse is a derived object: `faf_create_inverse(fwd)` / Chirp `(inverse)`.
It is not "run forward and scale."

A scale *is* sufficient for Haar-ortho (the pair is its own adjoint).
It is **not** sufficient for biorthogonal 5/3 and 9/7, for Haar-lazy
(divide-by-2 on recon), or for Haar-mean. Backend F + `CUSTOM_PR` uses the
stored synthesis taps. Backend F + `ANALYSIS_ONLY` returns NULL:
`analysis-only bank has no inverse`.

## 7. Haar is not one filter

| conv | Analysis LP | Analysis HP | Inverse | Backend |
|------|-------------|-------------|---------|---------|
| `haar_ortho` | `(1/√2)(1, 1)` | `(1/√2)(1, −1)` | same pair, adjoint | L default, F allowed |
| `haar_lazy` | `(1, 1)` | `(1, −1)` | divide by 2 on recon | L or F |
| `haar_mean` | `(1/2, 1/2)` | `(1, −1)` | documented dual | F allowed, L default for support 2 |

Ortho coefficients are not lazy coefficients. Create-time validation
refuses `family=haar` + `conv=cdf53_int`.

## 8. Han / custom atoms

Han is **not** a Mallat family and must not gain `FAF_WAVELET_HAN`.
It is Backend F, `ANALYSIS_ONLY`, typically 8-tap, `decim=1`.
`faf_create_inverse` on that object fails. See [Han.md](Han.md).

Custom PR four-tuples use `FAF_CONV_CUSTOM_PR` and `faf_dwt_set_taps`.
Analysis-only user filters use `FAF_CONV_ANALYSIS_ONLY`.
`(idwt :family han)` must not compile; there is no such family.
