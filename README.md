# qec-certify

**Learned quantum error-correction decoders with optimality certificates, grounded in the resource theory of quantum channels.**

> Status: early-stage research code. Everything below marked as a *hypothesis* is an open question, not a result.

---

## 1. Motivation

Machine-learned decoders (e.g. recurrent-transformer decoders such as AlphaQubit) now match or beat hand-designed decoders on real hardware noise. But a learned decoder only answers "how good is my decoder?", never "how far from optimal is it?". Meanwhile, quantum resource theory is good at exactly the opposite kind of statement: *no* strategy can do better than X.

This project connects the two:

- **ML gives achievable performance** (upper bounds on logical error rate).
- **Theory and exact computation give reference points and limits** (lower bounds, or the exact optimum on tractable instances).
- **Channel resource measures describe the noise structure** and let us ask which structural features of noise make learned decoders help, and by how much.

## 2. Research questions

**RQ1 (Gap to optimal).** For a given code, noise model, and decoder architecture, how large is the gap between the learned decoder's logical error rate and the maximum-likelihood (optimal) decoder's?

**RQ2 (Noise structure predicts advantage).** Does the gap between a matching-type decoder and the optimal decoder, and how much of that gap a learned decoder closes, correlate with quantitative descriptors of the noise channel (correlations, non-Pauli character, bias)?

**RQ3 (Sample complexity).** How many training samples does a learned decoder need to reach a given fraction of the optimal performance, as a function of code distance and noise structure?

**RQ4 (Certificates).** Can we produce a *provable* lower bound on logical error rate for a given noise model that is tight enough to certify a learned decoder as near-optimal? (Long-term, theory-heavy.)

### Hypotheses (to be tested, not assumed)

- H1: The advantage of learned decoders over minimum-weight perfect matching (MWPM) grows with the amount of correlated or non-Pauli structure in the noise.
- H2: On i.i.d. Pauli noise at small distance, a well-trained learned decoder approaches the ML decoder to within statistical error.
- H3: Sample complexity to reach a fixed gap-to-optimal grows with noise-structure descriptors, not only with code distance.

## 3. Approach

### 3.1 Three tiers of reference decoders

| Tier | Reference | Scope | Meaning |
|------|-----------|-------|---------|
| 0 | MWPM (PyMatching), optionally belief-matching / BP+OSD | All sizes | Practical baseline |
| 1 | Exact maximum-likelihood decoder | Small codes (d=3, 5; repetition and rotated surface code, code-capacity first) | True optimum on tractable instances |
| 2 | Analytic or information-theoretic bounds | Model-dependent | Certificates (research goal) |

### 3.2 Learned decoders

- Baseline MLP / CNN decoder
- GNN decoder over the detector graph
- Transformer decoder (attention over detector events across space and rounds)
- Recurrent-transformer variant (round-by-round processing, closer to AlphaQubit-style design)
- Optional later: RL-based decoding or decoder search (kept out of the first milestones)

### 3.3 Noise models (in order of implementation)

1. Code-capacity i.i.d. depolarizing
2. Biased Pauli noise
3. Phenomenological noise (measurement errors)
4. Correlated Pauli noise (crosstalk-style two-qubit correlations)
5. Coherent / non-Pauli noise (over-rotation), via an appropriate simulator
6. Circuit-level noise via Stim, including realistic gate-dependent parameters
7. Leakage (stretch goal)

### 3.4 Noise-structure descriptors (the "resource" side)

For each noise model compute, where well-defined:

- Distance of the channel from its Pauli twirl (e.g. process-fidelity or diamond-norm based gap)
- Pauli-error correlation measures (mutual information / total correlation between error indicators across qubits)
- Bias (ratio of Z to X/Y error rates)
- Entropic quantities: entropy of the error distribution, and the coherent information / hashing-bound-style capacity estimates for code-capacity settings

These descriptors are the independent variables in RQ2 and RQ3. Definitions must be documented in `docs/noise_descriptors.md` with precise formulas and references.

### 3.5 Evaluation metrics

- Logical error rate per round (with confidence intervals; enough shots for CI width below a target)
- **Gap ratio** `G = LER_learned / LER_ML` (Tier 1) and `LER_learned / LER_MWPM` (Tier 0)
- Fraction of the MWPM-to-ML gap closed: `(LER_MWPM - LER_learned) / (LER_MWPM - LER_ML)`
- Sample-efficiency curves: LER vs. number of training shots
- Inference latency and parameter count (for practical relevance)

## 4. Repository layout

```
qec-certify/
├── README.md
├── LICENSE                     # Apache-2.0
├── pyproject.toml
├── CITATION.cff
├── docs/
│   ├── noise_descriptors.md    # definitions of noise-structure measures
│   ├── certificates.md         # what is certified, what is not, assumptions
│   └── experiment_log.md       # running log of results and negative results
├── src/qec_certify/
│   ├── codes/                  # repetition, rotated surface code construction
│   ├── noise/                  # noise models + descriptors
│   ├── sim/                    # Stim wrappers, dataset generation
│   ├── decoders/
│   │   ├── mwpm.py             # PyMatching wrapper
│   │   ├── ml_exact.py         # exact ML decoder for small codes
│   │   ├── nn/                 # mlp.py, gnn.py, transformer.py, recurrent.py
│   ├── bounds/                 # information-theoretic / analytic bounds
│   ├── train/                  # training loops, configs
│   ├── eval/                   # metrics, gap ratio, CI computation
│   └── utils/
├── configs/                    # YAML experiment configs
├── experiments/                # scripts that reproduce each figure
├── notebooks/
├── tests/
└── .github/workflows/ci.yml
```

## 5. Milestones

**M0. Scaffolding.** Package, CI (lint + tests), config system, seeded reproducibility.

**M1. Baselines and ground truth (code capacity).**
- Repetition code and d=3 rotated surface code under i.i.d. depolarizing noise
- MWPM baseline via PyMatching
- Exact ML decoder by full syndrome-class enumeration (d=3), extended to d=5 if tractable
- Tests: ML LER must be less than or equal to MWPM LER within statistical error; ML matches brute-force on tiny instances

**M2. First learned decoder.**
- MLP and transformer decoders trained on Stim-generated data
- Report gap ratio vs. ML and vs. MWPM
- Sample-efficiency curves

**M3. Structured noise.**
- Biased and correlated noise families with a tunable structure parameter
- Compute noise descriptors; plot gap-closed vs. descriptor (RQ2)

**M4. Circuit-level noise.**
- Stim circuit-level memory experiments, d=3 to 7
- Tier 1 replaced by best available near-optimal reference (e.g. tensor-network or high-effort approximate ML) with clearly stated caveats

**M5. Theory track (parallel, not code-gated).**
- Formalize what a certificate means (`docs/certificates.md`)
- Derive and implement at least one analytic lower bound for a restricted noise family and compare against Tier 1 numerics

**M6. Write-up.** Reproducible figures, arXiv-ready draft, cleaned release.

## 6. Installation

```bash
git clone https://github.com/<your-username>/qec-certify.git
cd qec-certify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies: `stim`, `pymatching`, `numpy`, `scipy`, `torch`, `pyyaml`, `matplotlib`, `pytest`.
Optional: `jax` (alternative model backend), `ldpc` (BP+OSD baseline), `quimb` or similar (tensor-network reference).

## 7. Quickstart (target interface)

```bash
# 1. Generate data and evaluate baselines
python -m qec_certify.eval.baselines --config configs/m1_d3_depolarizing.yaml

# 2. Train a transformer decoder
python -m qec_certify.train.run --config configs/m2_transformer_d3.yaml

# 3. Compute gap-to-optimal report
python -m qec_certify.eval.report --run runs/m2_transformer_d3
```

## 8. Reproducibility and scientific standards

- Every experiment is defined by a YAML config plus a fixed seed; results are saved with git commit hash.
- All reported LERs include confidence intervals (Wilson or Clopper-Pearson).
- Train/validation/test noise seeds are separated; no test-shot leakage.
- **Negative results are logged** in `docs/experiment_log.md`.
- A "certificate" in this repo always states: noise model, code, decoder class, assumptions, and whether the bound is exact, numerical, or analytic.

## 9. Scope and honesty notes

- The exact ML reference is only feasible for small codes; conclusions about large distances are extrapolations until stated otherwise.
- A learned decoder matching MWPM or ML on a simulated noise model says nothing by itself about real hardware noise.
- Tier 2 (analytic certificates) is the open research component. Until it is delivered, the repo provides *exact reference gaps on small instances*, which is a weaker but still useful statement.

## 10. Related work

- Bausch et al., *Learning high-accuracy error decoding for quantum processors*, Nature (2024).
- Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum (2021).
- Higgott and Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching* (2023).
- Dennis, Kitaev, Landahl, Preskill, *Topological quantum memory*, J. Math. Phys. (2002).

(Verify all references and add full bibliographic details in `CITATION.cff` and `docs/`.)

## 11. Contributing

Issues and PRs welcome. Please include a config and a test for any new decoder, noise model, or metric.

## 12. License

Apache-2.0.

---

## Appendix: Instructions for Claude Code

Build this repository incrementally, milestone by milestone, starting with M0 and M1 only.

1. Create the layout in Section 4 with `pyproject.toml` (src layout, Python 3.10+), ruff + pytest config, and a GitHub Actions workflow running lint and tests.
2. Implement M1 fully before touching neural models:
   - Rotated surface code (d=3) and repetition code generation, with Stim circuits for code-capacity depolarizing noise.
   - `MWPMDecoder` wrapping PyMatching with a common `Decoder` interface: `fit(data)` (no-op for classical), `predict(detection_events) -> logical_flips`.
   - `ExactMLDecoder` that enumerates error configurations grouped by syndrome and logical class, choosing the most probable logical class per syndrome. Validate it against brute force on the repetition code.
   - Evaluation module returning LER with Wilson confidence intervals.
3. Write unit tests for every module. Do not report any result that is not produced by a script in `experiments/`.
4. Add MLP and transformer decoders (M2) only after M1 tests pass. Use PyTorch, config-driven training, deterministic seeding.
5. Never fabricate benchmark numbers in docs. Leave results tables empty until generated.
6. Keep `docs/experiment_log.md` updated after each experiment, including failures.
7. Ask before making architectural changes that deviate from this README.
