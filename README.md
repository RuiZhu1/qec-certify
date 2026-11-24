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
| 0 | MWPM and correlated MWPM (PyMatching), BP+OSD and Union-Find (DEM via ldpc), Relay-BP (public Rust implementation) | All sizes supported by each implementation | Practical baselines |
| 0 | Empirical lookup table (majority vote per seen syndrome, MWPM elsewhere) | Same data as the learned decoder | Floor for learned decoders: memorisation without generalisation |
| 1 | Exact maximum-likelihood decoder: full syndrome table (d=3, 5), or one exact DEM contraction per shot (code capacity up to d≈11) | Small to moderate codes, code-capacity first | True optimum on tractable instances |
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
│   ├── noise/                  # noise models (depolarizing, biased, correlated) + descriptors
│   ├── sim/                    # Stim wrappers, dataset generation
│   ├── decoders/
│   │   ├── mwpm.py             # PyMatching wrapper (plain and correlated matching)
│   │   ├── dem.py              # BP+OSD and Union-Find on arbitrary DEMs
│   │   ├── lookup.py           # empirical lookup-table decoder
│   │   ├── ml_exact.py         # exact ML decoder for small codes (full syndrome table)
│   │   ├── ml_sweep.py         # exact ML per shot by sweeping the DEM (moderate codes)
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
- Implemented: Stim rotated-surface memory with Clifford, idle, measurement and reset noise; channel-specific strengths, seeded qubit heterogeneity and drift; MWPM, correlated MWPM, BP+OSD and Union-Find (`configs/m4_circuit_level_d3.yaml`, `configs/m4_circuit_level_scaling.yaml`)
- qLDPC first increment: checksummed public BB [[18,4,3]] and gross [[144,12,12]] reference circuits, BP+OSD and public Rust Relay-BP (`configs/qldpc_*.yaml`)
- Next: larger statistically powered experiments, hardware-data adapters and optimized incremental streaming decoders
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

Core dependencies: `stim`, `pymatching`, `ldpc`, `relay-bp`, `numpy`, `scipy`, `torch`, `pyyaml`, `matplotlib`, `pytest`.
Optional: `jax` (alternative model backend), `quimb` or similar (tensor-network reference).

## 7. Quickstart

```bash
# 1. Generate data and evaluate baselines (MWPM, correlated MWPM, exact ML)
python -m qec_certify.eval.baselines --config configs/m1_d3_depolarizing.yaml
python -m qec_certify.eval.baselines --config configs/m1_scaling_depolarizing.yaml --workers 8

# 2. Train MLP decoders and report their exact gap to optimal, next to a lookup table
python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml --out runs/m2_mlp_d3

# 3. Structured noise (M3): the same baselines under biased and correlated noise
python -m qec_certify.eval.baselines --config configs/m3_biased.yaml --out runs/m3_biased --workers 8
python -m qec_certify.eval.baselines --config configs/m3_correlated.yaml --out runs/m3_correlated --workers 8

# 4. Circuit-level surface code (M4): MWPM, correlated MWPM, BP+OSD, Union-Find
python -m qec_certify.eval.baselines --config configs/m4_circuit_level_d3.yaml --out runs/m4_circuit_level_d3
python -m qec_certify.eval.baselines --config configs/m4_circuit_level_scaling.yaml --out runs/m4_circuit_level_scaling --workers 4
python -m qec_certify.eval.baselines --config configs/m4_streaming_window.yaml --out runs/m4_streaming_window --workers 3
python -m qec_certify.eval.baselines --config configs/m4_delayed_window.yaml --out runs/m4_delayed_window --workers 3
python -m qec_certify.eval.baselines --config configs/m4_window_sweep.yaml --out runs/m4_window_sweep --workers 4
python experiments/analyze_m4_window_sweep.py runs/m4_window_sweep/baselines.json --out runs/m4_window_sweep/paired_analysis.json

# 5. Public qLDPC reference circuits: BP+OSD and Relay-BP
python -m qec_certify.eval.qldpc --config configs/qldpc_bb18.yaml --out runs/qldpc_bb18
python -m qec_certify.eval.qldpc --config configs/qldpc_gross.yaml --out runs/qldpc_gross
python -m qec_certify.eval.qldpc_tune --config configs/qldpc_gross_tune.yaml --out runs/qldpc_gross_tune
python -m qec_certify.eval.qldpc_tune --config configs/qldpc_realtime_tune.yaml --out runs/qldpc_realtime_tune
python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_pareto.yaml --out runs/qldpc_pareto
python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_precision.yaml --out runs/qldpc_precision
python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_early_fallback.yaml --out runs/qldpc_early_fallback
python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_alpha.yaml --out runs/qldpc_alpha
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_alpha_robustness.yaml --out runs/qldpc_alpha_robustness
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_tail_budget.yaml --out runs/qldpc_tail_budget
python experiments/analyze_qldpc_precision.py runs/qldpc_precision/pareto.json --out runs/qldpc_precision/implementation_pairs.json
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_precision_robustness.yaml --out runs/qldpc_precision_robustness
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_parallel.yaml --out runs/qldpc_parallel
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_core_fresh.yaml --out runs/qldpc_core_fresh
python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_profile.yaml --out runs/qldpc_profile
python -m qec_certify.eval.qldpc_sweep --config configs/qldpc_robustness.yaml --out runs/qldpc_robustness
python experiments/analyze_qldpc_sweep.py runs/qldpc_robustness/sweep.json --out runs/qldpc_robustness/analysis.json
python -m qec_certify.eval.qldpc --config configs/qldpc_fallback.yaml --out runs/qldpc_fallback
python -m qec_certify.eval.qldpc --config configs/qldpc_retry.yaml --out runs/qldpc_retry

# Figures: scripts in experiments/ read these outputs
python experiments/plot_m3_structure.py runs/m3_biased/baselines.json runs/m3_correlated/baselines.json --out runs/m3_structure.png
```

Circuit-level JSON results include `experiment` and `rounds`. `ler_per_round` and its CI use
`1 - (1 - LER)**(1/rounds)`, explicitly labelled `independent_equal_hazard`; the original
whole-experiment `ler` remains authoritative. When `latency_shots` is set, each decoder also
reports offline batch-of-one p50/p95/p99 latency and throughput. These timings include Python
call overhead and are not per-round hardware-streaming measurements.

When `streaming_latency_shots` is set on a circuit-level experiment, a causal-replay adapter
feeds one detector round at a time and zero-fills the unknown future. Its final prediction is
tested to equal the offline decoder, and intermediate predictions do not depend on future
rounds. `streaming_performance` reports per-round p50/p95/p99 latency, throughput and, when
`round_period_us` is set, fixed-rate-arrival backlog estimates. This reference adapter replays
the full history after every round, so its work grows quadratically over a shot; it is not an
optimized incremental decoder. Union-Find is marked `unsupported_inconsistent_prefixes`
because zero-filled prefixes need not be valid DEM syndromes.

`streaming_method: sliding_window` instead builds a local DEM over the latest `window_rounds`
for MWPM and correlated MWPM. Cross-boundary mechanisms are projected onto in-window detectors
and marked `projected_dem_boundary`; a full-shot window is marked `exact_full_history` and must
match offline predictions. Approximate window accuracy is reported separately as
`streaming_ler`, while per-round timing uses `sliding_window_mwpm_w<W>` as its mode. Other
algorithms are explicitly marked unsupported rather than silently falling back to replay.

`streaming_method: delayed_commit` uses overlapping windows and matched-edge outputs to commit
only fully internal errors from the oldest round. Newer and boundary-crossing components remain
in the tail, and committed logical flips are XORed with that tail. Its partial-window mode is
`delayed_commit_boundary`, a causal heuristic rather than exact incremental MWPM; accuracy is
again reported separately as `streaming_ler`. Window widths can be lists, including per-distance
lists; all widths at a distance reuse the same test shots. `streaming_comparison` records total
prediction disagreements and the two paired directions `offline_correct_streaming_wrong` and
`offline_wrong_streaming_correct`, enabling McNemar-style comparisons against full history.
The separate `eval.qldpc` runner loads checksummed public BB reference circuits and records
their provenance. When BP+OSD and Relay-BP are both selected, `paired_comparison` reports their
per-shot transitions and exact McNemar p-value; single-decoder runs leave it null.
`eval.qldpc_tune` evaluates predeclared Relay-BP candidates only on the validation split, selects
by validation errors, measured latency, iteration budget and name in that order, and only then
runs the held-out test split. `tuning.json` preserves all candidates, the selection rule and the
separate validation/test seeds. Setting `max_validation_p99_ms` restricts selection to candidates
within that validation latency budget when possible; if none qualify, the lowest-p99 candidate is
selected with `latency_budget.met: false` rather than being presented as meeting the budget.
`eval.qldpc_pareto` instead evaluates every validation-nondominated accuracy/latency candidate
on the same held-out shots with a single baseline evaluation, preserving the full tradeoff.
With `evaluate_all: true`, even dominated candidates are tested for numerical-implementation
comparisons. Pareto candidates may set `decoder: relay_bp_fallback` to evaluate early BP+LSD
takeover at different Relay budgets with shared shots. Relay-BP supports `implementation: f32/i32/i64`
and optional `alpha`/`alpha_iteration_scaling_factor` convergence controls; unsuccessful native decodes fall
back to no-flip and are reported through `decode_failures`, `decode_success_rate` and iteration
statistics rather than being returned as valid corrections. `eval.qldpc_precision_sweep`
compares those implementations across noise strengths and independent seeds on shared shots,
with per-cell checkpoints and per-implementation decode diagnostics. Serial Relay prediction uses
a fresh native decoder per shot to prevent state carryover; `parallel: true` uses independent
clones through `ObservableDecoderRunner`. Batch `evaluation_seconds` and batch-of-one latency are
reported separately, since parallel throughput does not imply lower single-shot latency.
`profile_shots` adds iteration percentiles, success/failure-specific latency and serial
initialization/native-decode/Python-overhead breakdown; parallel clone setup is reported as
unavailable rather than guessed.
`relay_bp_fallback` invokes
BP+LSD only for unsuccessful Relay shots and reports `fallback_calls`, `fallback_rate` and the
resulting end-to-end latency. With `fallback_decoder: relay_bp`, a shorter independent-seed Relay
retry is used instead; `fallback_relay_bp` sets its parameters and final success/failure is
reported separately.
`eval.qldpc_sweep` holds those selected parameters fixed across
reference noise strengths and independent experiment seeds, with generic pairwise McNemar
comparisons; `experiments/analyze_qldpc_sweep.py` pools shots only for descriptive summaries.
The sweep CLI checkpoints every completed cell to `cells.jsonl` and automatically skips those
cells when rerun, so long experiments can resume without reusing partial decoder results.

Planned: transformer decoders (`configs/m2_transformer_d3.yaml`) and a separate report
command (`python -m qec_certify.eval.report`).

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
