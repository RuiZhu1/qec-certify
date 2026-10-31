# Noise-structure descriptors

Definitions of the noise-structure measures used as independent variables in RQ2 and RQ3
(README Section 3.4). Each entry gives a precise formula, the conventions it depends on
(normalisation, logarithm base, which qubits it is computed over) and a reference.

## Conventions

- All implemented noise models are code-capacity Pauli channels on the data qubits
  (`src/qec_certify/noise/models.py`): independent single-qubit channels, plus, for
  correlated noise, independent two-qubit channels on nearest-neighbour pairs.
- `E_q` is the Pauli error on data qubit `q` after all channels act (phases ignored), and
  `(E_a, E_b)` the joint error on two qubits. Both are computed exactly by composing every
  channel that touches those qubits and summing out all other qubits
  (`qec_certify.noise.descriptors.marginal`).
- Descriptors are reported for one *bulk* qubit `a` (most neighbours, nearest the centre)
  and one of its neighbours `b` (`bulk_pair`). For the i.i.d. models every qubit gives the
  same values; for correlated noise boundary qubits have fewer pair channels and less noise.
- Logarithms are base 2 (bits).

## Implemented

| Descriptor | Formula | Applies to | Reference | Implemented in |
|------------|---------|------------|-----------|----------------|
| Error rate | `eps = 1 - P(E_a = I)` | all | | `describe()["error_rate"]` |
| Bias | `eta = P(E_a = Z) / (P(E_a = X) + P(E_a = Y))`; 0.5 for depolarizing noise, `inf` for pure dephasing | all | Tuckett, Bartlett, Flammia, PRL 120, 050505 (2018) | `descriptors.bias`, `describe()["bias"]` |
| Pauli-error entropy | `H(E_a) = -sum_P P(E_a = P) log2 P(E_a = P)` | all | Cover and Thomas, *Elements of Information Theory* | `descriptors.entropy_bits`, `describe()["entropy_bits"]` |
| Neighbour mutual information | `I(E_a; E_b) = H(E_a) + H(E_b) - H(E_a, E_b)` for neighbouring data qubits `a`, `b` | all; zero for i.i.d. noise | Cover and Thomas | `descriptors.mutual_information_bits`, `describe()["neighbour_mutual_information_bits"]` |

Notes:

- For an i.i.d. Pauli channel, `1 - H(E_a)` is the hashing-bound rate (bits of logical
  information per physical qubit achievable with random stabilizer codes), so the entropy
  also locates the hashing bound: `H = 1` at depolarizing `p ≈ 0.189`. For correlated noise
  the single-qubit entropy alone does not give the hashing rate.
- The noise models' own parameters are the controlled variables: `bias` for
  `BiasedPauli`, and `correlation` (fraction of a bulk qubit's first-order error
  probability that comes from pair events) for `CorrelatedPairs`. The descriptors above
  are their model-independent counterparts, so different families can share one axis.

## Not yet defined

| Descriptor | Status |
|------------|--------|
| Distance from the Pauli twirl (process fidelity or diamond norm) | Zero for every implemented model, which are all Pauli channels; needs coherent or non-Pauli noise (README 3.3, item 5) |
| Total correlation over a block of qubits | TBD; the pairwise mutual information is the first step |
| Coherent information / hashing bound for correlated noise | TBD |
