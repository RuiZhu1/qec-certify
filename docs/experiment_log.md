# Experiment log

A running log of every experiment, including failures and negative results. Each entry
records the date, the command or script (from `experiments/` or a module CLI), the config,
the git commit, where the outputs live, and what was learned. Results tables stay empty
until a script generates them; numbers are never typed in by hand.

## Entry template

```
### YYYY-MM-DD: <short title>
- Command:
- Config:
- Commit:
- Outputs:
- Outcome (including failures):
- Notes:
```

## Entries

### 2026-09-30: M2 MLP vs. MWPM and exact ML, d=3 rotated surface code
- Command: `python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml --out runs/m2_mlp_d3`
- Config: `configs/m2_mlp_d3.yaml` (MLP 8-64-64-2, Adam lr 1e-3, batch 1024, at least 4000
  gradient steps, best epoch chosen on 100k validation shots; one training run per cell)
- Commit: parent of the commit that adds this entry (`git_commit` in the JSON is `03661f9a5ff777ed8ed6ebed7b58661181675977-dirty`)
- Outputs: `runs/m2_mlp_d3/mlp_report.json` (not tracked; regenerate with the command above)
- Outcome: exact LER (all 2^8 syndromes decoded) gives the gap to optimal with no sampling
  noise. `gap closed` is 0 for MWPM and 1 for exact ML; negative means worse than MWPM.
- Notes:
  - A first run with a fixed 20 epochs showed that at 1000 training shots the MLP's exact LER
    equalled P(logical flip), i.e. it predicted "no flip" for every syndrome, and best epochs
    sat at the epoch cap. Small sets were under-trained, not only data-limited, so
    `min_steps` was added and the run repeated; the table below is the repeat.
  - Single training seed per cell, so differences of a few percent between neighbouring
    cells are not established.
  - Statements are for d=3, code-capacity i.i.d. depolarizing noise only.

```
      p  decoder        train   exact LER   G=LER/ML  gap closed  sampled LER [CI]
   0.01  MWPM               -   7.795e-04      1.189           -  6.700e-04 [5.277e-04, 8.507e-04]
   0.01  exact ML           -   6.553e-04      1.000           -  5.900e-04 [4.575e-04, 7.609e-04]
   0.01  MLP             1000   7.327e-03     11.181     -52.743  6.910e-03 [6.415e-03, 7.443e-03]
   0.01  MLP            10000   8.831e-04      1.348      -0.835  7.200e-04 [5.718e-04, 9.065e-04]
   0.01  MLP           100000   7.390e-04      1.128       0.326  6.800e-04 [5.365e-04, 8.619e-04]
   0.01  MLP          1000000   6.656e-04      1.016       0.917  6.700e-04 [5.277e-04, 8.507e-04]
   0.02  MWPM               -   3.037e-03      1.179           -  3.320e-03 [2.982e-03, 3.696e-03]
   0.02  exact ML           -   2.575e-03      1.000           -  2.780e-03 [2.472e-03, 3.126e-03]
   0.02  MLP             1000   3.732e-03      1.449      -1.505  4.020e-03 [3.646e-03, 4.432e-03]
   0.02  MLP            10000   2.924e-03      1.135       0.245  3.110e-03 [2.783e-03, 3.475e-03]
   0.02  MLP           100000   2.651e-03      1.029       0.836  2.840e-03 [2.529e-03, 3.189e-03]
   0.02  MLP          1000000   2.575e-03      1.000       1.000  2.870e-03 [2.557e-03, 3.221e-03]
   0.05  MWPM               -   1.753e-02      1.152           -  1.803e-02 [1.722e-02, 1.887e-02]
   0.05  exact ML           -   1.521e-02      1.000           -  1.555e-02 [1.480e-02, 1.634e-02]
   0.05  MLP             1000   2.137e-02      1.405      -1.659  2.218e-02 [2.129e-02, 2.311e-02]
   0.05  MLP            10000   1.561e-02      1.026       0.829  1.601e-02 [1.525e-02, 1.681e-02]
   0.05  MLP           100000   1.524e-02      1.002       0.988  1.576e-02 [1.501e-02, 1.655e-02]
   0.05  MLP          1000000   1.524e-02      1.002       0.988  1.568e-02 [1.493e-02, 1.647e-02]
    0.1  MWPM               -   6.119e-02      1.115           -  6.117e-02 [5.970e-02, 6.267e-02]
    0.1  exact ML           -   5.489e-02      1.000           -  5.420e-02 [5.281e-02, 5.562e-02]
    0.1  MLP             1000   6.340e-02      1.155      -0.350  6.345e-02 [6.196e-02, 6.498e-02]
    0.1  MLP            10000   5.725e-02      1.043       0.625  5.751e-02 [5.608e-02, 5.897e-02]
    0.1  MLP           100000   5.489e-02      1.000       1.000  5.465e-02 [5.326e-02, 5.608e-02]
    0.1  MLP          1000000   5.489e-02      1.000       1.000  5.425e-02 [5.286e-02, 5.567e-02]
   0.15  MWPM               -   1.197e-01      1.086           -  1.200e-01 [1.180e-01, 1.220e-01]
   0.15  exact ML           -   1.102e-01      1.000           -  1.104e-01 [1.085e-01, 1.123e-01]
   0.15  MLP             1000   1.286e-01      1.166      -0.939  1.296e-01 [1.275e-01, 1.317e-01]
   0.15  MLP            10000   1.119e-01      1.015       0.820  1.128e-01 [1.108e-01, 1.148e-01]
   0.15  MLP           100000   1.106e-01      1.003       0.961  1.109e-01 [1.090e-01, 1.129e-01]
   0.15  MLP          1000000   1.102e-01      1.000       1.000  1.101e-01 [1.082e-01, 1.121e-01]
```

### 2026-09-30: M2 repeated runs (d=3) and d=5, MLP vs. MWPM and exact ML
- Command: `python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml --out runs/m2_mlp_d3`
  and the same with `configs/m2_mlp_d5.yaml`
- Config: `configs/m2_mlp_d3.yaml` (MLP 8-64-64-2, 5 repeats) and `configs/m2_mlp_d5.yaml`
  (MLP 24-128-128-2, 3 repeats). Adam lr 1e-3, batch 1024, at least 4000 gradient steps,
  at most 20 epochs beyond that floor, best epoch chosen on 100k validation shots. Each repeat
  has its own training shots and network seed; validation and test shots are shared.
- Commit: `git_commit` in each JSON (`596f2060b1448da9dda982423f1542cc9630e834` for d=3, `596f2060b1448da9dda982423f1542cc9630e834` for d=5);
  the working tree was clean apart from generated files.
- Outputs: `runs/m2_mlp_d3/mlp_report.json`, `runs/m2_mlp_d5/mlp_report.json` and the matching
  `progress.log` (untracked; regenerate with the commands above)
- Outcome: G is exact LER over exact-ML LER, so 1.000 is optimal; `gap closed` is 0 for MWPM
  and 1 for exact ML, negative means worse than MWPM.
  - d=3: with at least 1e5 training shots the MLP is within 3% of exact ML for p >= 0.02
    (p = 0.01 needs 1e6 shots for 1.013 +- 0.013). Repeats confirm the earlier single-run
    picture; the spread is large only for small training sets, where 1e3 shots at p = 0.01
    ranges from 1.54 to 11.18.
  - d=5: the d=3 result does not carry over. At 1e6 shots the MLP beats MWPM only for
    p >= 0.05, ties at p = 0.02 (G 1.80 vs 1.79) and is far worse at p = 0.01 (3.17 vs 1.84).
    No cell reaches exact ML; the closest is p = 0.15 at G = 1.09.
- Notes:
  - d=5 looks data-limited at low p (best epochs 6 to 13 of 20 at 1e6 shots, G still falling
    about 2x to 7x per 10x data) but best epochs of 15 to 20 at p >= 0.05 sit near the epoch
    cap, so those cells may also be under-trained. This is not resolved; a longer epoch budget
    and more data are the cheap next checks before changing architecture.
  - Three repeats at d=5 is a small sample. The spread is small for p >= 0.05 and large for
    p = 0.01.
  - Statements are for code-capacity i.i.d. depolarizing noise, one architecture per code.

d=3:
```
      p  decoder      train   exact LER  G = LER / LER_ML  (mean +- std, [min, max])     gap closed
   0.01  MWPM             -   7.795e-04  1.189                                                    -
   0.01  exact ML         -   6.553e-04  1.000                                                    -
   0.01  MLP           1000   2.383e-03  3.636 +- 4.222  [1.540, 11.181]                    -12.917
   0.01  MLP          10000   9.482e-04  1.447 +- 0.106  [1.348, 1.618]                      -1.359
   0.01  MLP         100000   7.344e-04  1.121 +- 0.047  [1.047, 1.175]                       0.363
   0.01  MLP        1000000   6.637e-04  1.013 +- 0.013  [1.000, 1.032]                       0.932
   0.02  MWPM             -   3.037e-03  1.179                                                    -
   0.02  exact ML         -   2.575e-03  1.000                                                    -
   0.02  MLP           1000   4.083e-03  1.585 +- 0.132  [1.449, 1.798]                      -2.263
   0.02  MLP          10000   3.160e-03  1.227 +- 0.074  [1.135, 1.316]                      -0.265
   0.02  MLP         100000   2.652e-03  1.030 +- 0.015  [1.015, 1.045]                       0.835
   0.02  MLP        1000000   2.592e-03  1.007 +- 0.014  [1.000, 1.031]                       0.963
   0.05  MWPM             -   1.753e-02  1.152                                                    -
   0.05  exact ML         -   1.521e-02  1.000                                                    -
   0.05  MLP           1000   2.075e-02  1.364 +- 0.085  [1.263, 1.486]                      -1.389
   0.05  MLP          10000   1.599e-02  1.051 +- 0.031  [1.024, 1.090]                       0.664
   0.05  MLP         100000   1.530e-02  1.006 +- 0.006  [1.000, 1.012]                       0.964
   0.05  MLP        1000000   1.525e-02  1.002 +- 0.001  [1.000, 1.004]                       0.986
    0.1  MWPM             -   6.119e-02  1.115                                                    -
    0.1  exact ML         -   5.489e-02  1.000                                                    -
    0.1  MLP           1000   6.636e-02  1.209 +- 0.103  [1.129, 1.376]                      -0.820
    0.1  MLP          10000   5.612e-02  1.022 +- 0.015  [1.003, 1.043]                       0.805
    0.1  MLP         100000   5.498e-02  1.002 +- 0.002  [1.000, 1.003]                       0.985
    0.1  MLP        1000000   5.501e-02  1.002 +- 0.001  [1.000, 1.003]                       0.980
   0.15  MWPM             -   1.197e-01  1.086                                                    -
   0.15  exact ML         -   1.102e-01  1.000                                                    -
   0.15  MLP           1000   1.286e-01  1.166 +- 0.074  [1.076, 1.280]                      -0.937
   0.15  MLP          10000   1.108e-01  1.005 +- 0.004  [1.000, 1.012]                       0.936
   0.15  MLP         100000   1.103e-01  1.001 +- 0.002  [1.000, 1.003]                       0.992
   0.15  MLP        1000000   1.102e-01  1.000 +- 0.000  [1.000, 1.000]                       1.000
```

d=5:
```
      p  decoder      train   exact LER  G = LER / LER_ML  (mean +- std, [min, max])     gap closed
   0.01  MWPM             -   8.242e-05  1.842                                                    -
   0.01  exact ML         -   4.475e-05  1.000                                                    -
   0.01  MLP          10000   2.158e-03  48.233 +- 8.524  [39.843, 56.885]                  -55.103
   0.01  MLP         100000   3.271e-04  7.309 +- 1.753  [5.882, 9.266]                      -6.494
   0.01  MLP        1000000   1.419e-04  3.171 +- 0.211  [2.927, 3.302]                      -1.578
   0.02  MWPM             -   6.272e-04  1.788                                                    -
   0.02  exact ML         -   3.508e-04  1.000                                                    -
   0.02  MLP          10000   6.313e-03  17.997 +- 2.117  [16.448, 20.409]                  -20.569
   0.02  MLP         100000   1.242e-03  3.540 +- 0.024  [3.515, 3.563]                      -2.224
   0.02  MLP        1000000   6.308e-04  1.798 +- 0.182  [1.693, 2.008]                      -0.013
   0.05  MWPM             -   8.364e-03  1.635                                                    -
   0.05  exact ML         -   5.114e-03  1.000                                                    -
   0.05  MLP          10000   2.090e-02  4.086 +- 0.130  [4.010, 4.236]                      -3.855
   0.05  MLP         100000   9.111e-03  1.781 +- 0.051  [1.723, 1.818]                      -0.230
   0.05  MLP        1000000   6.482e-03  1.267 +- 0.009  [1.258, 1.277]                       0.579
    0.1  MWPM             -   5.021e-02  1.429                                                    -
    0.1  exact ML         -   3.514e-02  1.000                                                    -
    0.1  MLP          10000   7.861e-02  2.237 +- 0.021  [2.225, 2.261]                      -1.885
    0.1  MLP         100000   4.660e-02  1.326 +- 0.013  [1.312, 1.338]                       0.240
    0.1  MLP        1000000   4.021e-02  1.144 +- 0.004  [1.140, 1.148]                       0.663
   0.15  MWPM             -   1.244e-01  1.283                                                    -
   0.15  exact ML         -   9.698e-02  1.000                                                    -
   0.15  MLP          10000   1.707e-01  1.760 +- 0.046  [1.725, 1.812]                      -1.685
   0.15  MLP         100000   1.162e-01  1.198 +- 0.007  [1.191, 1.205]                       0.300
   0.15  MLP        1000000   1.055e-01  1.088 +- 0.005  [1.083, 1.093]                       0.690
```
