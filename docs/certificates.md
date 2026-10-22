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

## Tier 1, beyond the table: exact ML per shot (implemented)

- **Statement.** `SweepMLDecoder` (`src/qec_certify/decoders/ml_sweep.py`) computes
  `P(s, l)` exactly for the observed syndrome `s` of each shot and predicts
  `argmax_l P(s, l)`, the same decision as the full table, without tabulating all `2^D`
  syndromes. It contracts the DEM's tensor network along a sweep: mechanisms are applied one
  at a time to the joint distribution of the *open* detectors (touched by processed and
  unprocessed mechanisms) and the observables, and each detector is conditioned on its
  observed value right after its last mechanism.
- **Kind.** The decoder is exactly ML (up to float64 rounding; all terms are non-negative,
  so there is no cancellation). Its LER is *sampled*, so statements about it carry Wilson
  intervals; it is not an exact LER like `optimal_logical_error_rate`.
- **Codes covered.** Cost per shot is `O(M 2^(w + L))` for frontier width `w`. For the
  rotated surface code under code-capacity noise a column-by-column sweep gives `w <= d + 3`
  (tested for d=3 to 9), so d=11 costs about `2^15` numbers per shot.
- **Validation (tests/test_ml_sweep.py).** `log P(s, l)` agrees with the full table to
  `1e-11` on every d=3 syndrome (both bases), on repetition codes and random DEMs, and on
  sampled d=5 syndromes; decisions agree except on exact ties, where any choice is optimal.
- **Assumptions and caveats.** As for the table: the DEM must describe the noise, and
  optimality is relative to that DEM.

## Tier 2: analytic bounds (open, M5)

Not yet defined.
