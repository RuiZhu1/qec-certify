# Certificates

Every certificate in this repository states: the noise model, the code, the decoder class,
its assumptions, and whether the bound is **exact**, **numerical** or **analytic**
(README Section 8).

## Tier 1: exact maximum likelihood (implemented, M1)

- **Statement.** For a detector error model (DEM) of independent error mechanisms,
  `ExactMLDecoder` (`src/qec_certify/decoders/ml_exact.py`) computes the exact joint
  distribution `P(s, l)` of the syndrome `s` and the logical class `l`, predicts
  `argmax_l P(s, l)` for every syndrome, and reports
  `optimal_logical_error_rate = sum_s (P(s) - max_l P(s, l))`. No decoder that sees only
  `s` can have a lower logical error rate under that DEM.
- **Kind.** Exact, up to float64 rounding. It is not a sampled estimate.
- **Noise models covered.** Anything Stim can express as a DEM of independent mechanisms.
  So far it has only been validated on code-capacity i.i.d. depolarizing noise, where the
  DEM reproduces the Pauli channel exactly.
- **Codes covered.** Limited by memory: `2^(D+L)` float64 entries for `D` detectors and
  `L` observables (the default cap is `D + L <= 26`). This includes the d=3 and d=5 rotated
  surface codes under code-capacity noise, and small repetition codes.
- **Validation (tests/test_ml_exact.py).** Agrees with literal enumeration of all `2^M`
  mechanism configurations (repetition codes, random DEMs), and with enumeration of all
  `4^n` physical Pauli errors (d=3 surface code, both bases). The d=3 repetition-code LER
  matches the closed form.
- **Assumptions and caveats.** The DEM must describe the noise faithfully. Optimality is
  over decoders that see this syndrome for a single shot, relative to the stated noise
  model. It says nothing about other noise models or about hardware.

## Tier 2: analytic bounds (open, M5)

Not yet defined.
