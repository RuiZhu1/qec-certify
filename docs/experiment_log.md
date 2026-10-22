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

### 2026-10-22: MWPM and correlated MWPM vs. exact ML, d=3 to 11 rotated surface code
- Command: `python -m qec_certify.eval.baselines --config configs/m1_scaling_depolarizing.yaml --out runs/m1_scaling_depolarizing --workers 10`;
  figure: `python experiments/plot_m1_scaling.py runs/m1_scaling_depolarizing/baselines.json --out runs/m1_scaling_depolarizing/gap_vs_distance.png`
- Config: `configs/m1_scaling_depolarizing.yaml` (200k test shots per cell; decoders MWPM,
  correlated MWPM, and exact ML via `SweepMLDecoder`, frontier 4/8/10/12/14 detectors for
  d=3/5/7/9/11)
- Commit: code as committed in `7d7d4b49106d6460b2bac711a75e4a4cf93eaee0`; the run predates that commit, so `git_commit` in
  the JSON is its parent with uncommitted changes, `8bf689b5de0cb219d5ce01ee330e6eea744ff934-dirty`.
- Outputs: `runs/m1_scaling_depolarizing/baselines.json`, `progress.log` and
  `gap_vs_distance.png` (untracked; regenerate with the commands above, about 50 minutes on
  10 processes, dominated by d=11)
- Outcome: `G` is a decoder's sampled LER over exact ML's on the *same* shots.
  - MWPM's gap to optimal grows geometrically with distance at every p: at p = 0.1 it is
    1.12, 1.44, 1.84, 2.42, 3.15 for d = 3 to 11 (about x1.3 per step of 2), and 6.4 at
    d = 11, p = 0.05. This is the expected sign of ML's higher threshold, but it means the
    d=3 and d=5 numbers understate how much room a decoder has above MWPM.
  - Correlated MWPM closes a large part of that gap from d=5 on, yet still grows with d:
    1.45, 1.62, 1.85 at p = 0.1 for d = 7, 9, 11, and 2.7 at d = 11, p = 0.05. At d = 3 it is
    worse than plain MWPM for p <= 0.1 (also seen in exact LERs, see the M2 entry below).
  - Low-p cells have few ML failures (28 at d = 11, p = 0.05; 84 at d = 9, p = 0.05), so
    their G has a relative standard error of about 20% and 12%. The figure's error bars treat
    the two LERs as independent and are therefore conservative.
- Notes:
  - Check on Tier 1: at d = 5 the table decoder (`ExactMLDecoder`) makes the same decision as
    `SweepMLDecoder` on every one of the 200k shots of the p = 0.05 cell (958 failures each).
    That cell's sampled LER, 4.790e-3 [4.497e-3, 5.102e-3], sits about 2 sigma below the exact
    5.114e-3; 2M fresh shots give 5.061e-3 [4.963e-3, 5.160e-3], so it is a fluctuation.
  - A first attempt at this run was stopped: ties between equally likely classes were
    broken by rounding, so d=5 ML counts moved by up to 16 shots between code versions. Both
    ML decoders now share a tie rule (`TIE_RTOL`), and the numbers below come from the rerun.
  - Code capacity, i.i.d. depolarizing noise, Z-basis memory only.

```
  d       p  decoder               errors/shots         LER       G  CI
  3    0.05  mwpm                 3456/200000     1.728e-02   1.170  [1.672e-02, 1.786e-02]
  3    0.05  mwpm_correlated      3648/200000     1.824e-02   1.235  [1.766e-02, 1.884e-02]
  3    0.05  ml_sweep             2953/200000     1.477e-02   1.000  [1.425e-02, 1.530e-02]
  3    0.08  mwpm                 8213/200000     4.106e-02   1.119  [4.020e-02, 4.194e-02]
  3    0.08  mwpm_correlated      8619/200000     4.310e-02   1.174  [4.221e-02, 4.399e-02]
  3    0.08  ml_sweep             7341/200000     3.671e-02   1.000  [3.589e-02, 3.754e-02]
  3     0.1  mwpm                12244/200000     6.122e-02   1.119  [6.018e-02, 6.228e-02]
  3     0.1  mwpm_correlated     12730/200000     6.365e-02   1.163  [6.259e-02, 6.473e-02]
  3     0.1  ml_sweep            10945/200000     5.473e-02   1.000  [5.374e-02, 5.573e-02]
  3    0.12  mwpm                16764/200000     8.382e-02   1.107  [8.261e-02, 8.504e-02]
  3    0.12  mwpm_correlated     16183/200000     8.092e-02   1.069  [7.973e-02, 8.212e-02]
  3    0.12  ml_sweep            15137/200000     7.569e-02   1.000  [7.453e-02, 7.685e-02]
  3    0.15  mwpm                23905/200000     1.195e-01   1.082  [1.181e-01, 1.210e-01]
  3    0.15  mwpm_correlated     23248/200000     1.162e-01   1.052  [1.148e-01, 1.177e-01]
  3    0.15  ml_sweep            22096/200000     1.105e-01   1.000  [1.091e-01, 1.119e-01]
  5    0.05  mwpm                 1666/200000     8.330e-03   1.739  [7.941e-03, 8.738e-03]
  5    0.05  mwpm_correlated      1323/200000     6.615e-03   1.381  [6.269e-03, 6.980e-03]
  5    0.05  ml_sweep              958/200000     4.790e-03   1.000  [4.497e-03, 5.102e-03]
  5    0.08  mwpm                 5605/200000     2.803e-02   1.525  [2.731e-02, 2.876e-02]
  5    0.08  mwpm_correlated      4765/200000     2.382e-02   1.296  [2.317e-02, 2.450e-02]
  5    0.08  ml_sweep             3676/200000     1.838e-02   1.000  [1.780e-02, 1.898e-02]
  5     0.1  mwpm                10223/200000     5.112e-02   1.442  [5.016e-02, 5.209e-02]
  5     0.1  mwpm_correlated      9009/200000     4.505e-02   1.271  [4.414e-02, 4.596e-02]
  5     0.1  ml_sweep             7090/200000     3.545e-02   1.000  [3.465e-02, 3.627e-02]
  5    0.12  mwpm                15450/200000     7.725e-02   1.348  [7.609e-02, 7.843e-02]
  5    0.12  mwpm_correlated     13788/200000     6.894e-02   1.203  [6.784e-02, 7.006e-02]
  5    0.12  ml_sweep            11465/200000     5.733e-02   1.000  [5.631e-02, 5.835e-02]
  5    0.15  mwpm                24879/200000     1.244e-01   1.285  [1.230e-01, 1.258e-01]
  5    0.15  mwpm_correlated     22648/200000     1.132e-01   1.170  [1.119e-01, 1.146e-01]
  5    0.15  ml_sweep            19360/200000     9.680e-02   1.000  [9.551e-02, 9.810e-02]
  7    0.05  mwpm                  787/200000     3.935e-03   2.429  [3.670e-03, 4.219e-03]
  7    0.05  mwpm_correlated       525/200000     2.625e-03   1.620  [2.410e-03, 2.859e-03]
  7    0.05  ml_sweep              324/200000     1.620e-03   1.000  [1.453e-03, 1.806e-03]
  7    0.08  mwpm                 4091/200000     2.046e-02   2.145  [1.984e-02, 2.108e-02]
  7    0.08  mwpm_correlated      2931/200000     1.465e-02   1.537  [1.414e-02, 1.519e-02]
  7    0.08  ml_sweep             1907/200000     9.535e-03   1.000  [9.118e-03, 9.970e-03]
  7     0.1  mwpm                 8183/200000     4.091e-02   1.841  [4.006e-02, 4.179e-02]
  7     0.1  mwpm_correlated      6427/200000     3.213e-02   1.446  [3.137e-02, 3.292e-02]
  7     0.1  ml_sweep             4446/200000     2.223e-02   1.000  [2.159e-02, 2.289e-02]
  7    0.12  mwpm                14025/200000     7.013e-02   1.689  [6.901e-02, 7.125e-02]
  7    0.12  mwpm_correlated     11245/200000     5.622e-02   1.354  [5.522e-02, 5.724e-02]
  7    0.12  ml_sweep             8303/200000     4.152e-02   1.000  [4.065e-02, 4.240e-02]
  7    0.15  mwpm                25538/200000     1.277e-01   1.502  [1.262e-01, 1.292e-01]
  7    0.15  mwpm_correlated     21666/200000     1.083e-01   1.274  [1.070e-01, 1.097e-01]
  7    0.15  ml_sweep            17005/200000     8.503e-02   1.000  [8.381e-02, 8.626e-02]
  9    0.05  mwpm                  399/200000     1.995e-03   4.750  [1.809e-03, 2.200e-03]
  9    0.05  mwpm_correlated       176/200000     8.800e-04   2.095  [7.593e-04, 1.020e-03]
  9    0.05  ml_sweep               84/200000     4.200e-04   1.000  [3.393e-04, 5.199e-04]
  9    0.08  mwpm                 2824/200000     1.412e-02   2.929  [1.361e-02, 1.465e-02]
  9    0.08  mwpm_correlated      1676/200000     8.380e-03   1.739  [7.990e-03, 8.789e-03]
  9    0.08  ml_sweep              964/200000     4.820e-03   1.000  [4.526e-03, 5.133e-03]
  9     0.1  mwpm                 6577/200000     3.288e-02   2.424  [3.211e-02, 3.368e-02]
  9     0.1  mwpm_correlated      4390/200000     2.195e-02   1.618  [2.132e-02, 2.260e-02]
  9     0.1  ml_sweep             2713/200000     1.357e-02   1.000  [1.307e-02, 1.408e-02]
  9    0.12  mwpm                12741/200000     6.370e-02   2.091  [6.264e-02, 6.478e-02]
  9    0.12  mwpm_correlated      9213/200000     4.607e-02   1.512  [4.515e-02, 4.699e-02]
  9    0.12  ml_sweep             6094/200000     3.047e-02   1.000  [2.973e-02, 3.123e-02]
  9    0.15  mwpm                25818/200000     1.291e-01   1.711  [1.276e-01, 1.306e-01]
  9    0.15  mwpm_correlated     20741/200000     1.037e-01   1.374  [1.024e-01, 1.050e-01]
  9    0.15  ml_sweep            15093/200000     7.547e-02   1.000  [7.432e-02, 7.663e-02]
 11    0.05  mwpm                  178/200000     8.900e-04   6.357  [7.685e-04, 1.031e-03]
 11    0.05  mwpm_correlated        76/200000     3.800e-04   2.714  [3.036e-04, 4.755e-04]
 11    0.05  ml_sweep               28/200000     1.400e-04   1.000  [9.687e-05, 2.023e-04]
 11    0.08  mwpm                 1951/200000     9.755e-03   3.926  [9.334e-03, 1.020e-02]
 11    0.08  mwpm_correlated       966/200000     4.830e-03   1.944  [4.536e-03, 5.144e-03]
 11    0.08  ml_sweep              497/200000     2.485e-03   1.000  [2.276e-03, 2.713e-03]
 11     0.1  mwpm                 5503/200000     2.752e-02   3.148  [2.681e-02, 2.824e-02]
 11     0.1  mwpm_correlated      3230/200000     1.615e-02   1.848  [1.561e-02, 1.671e-02]
 11     0.1  ml_sweep             1748/200000     8.740e-03   1.000  [8.341e-03, 9.157e-03]
 11    0.12  mwpm                11474/200000     5.737e-02   2.647  [5.636e-02, 5.840e-02]
 11    0.12  mwpm_correlated      7566/200000     3.783e-02   1.746  [3.700e-02, 3.868e-02]
 11    0.12  ml_sweep             4334/200000     2.167e-02   1.000  [2.104e-02, 2.232e-02]
 11    0.15  mwpm                26062/200000     1.303e-01   1.979  [1.288e-01, 1.318e-01]
 11    0.15  mwpm_correlated     19995/200000     9.997e-02   1.519  [9.867e-02, 1.013e-01]
 11    0.15  ml_sweep            13167/200000     6.584e-02   1.000  [6.476e-02, 6.693e-02]
```

### 2026-10-22: M2 rerun with a lookup-table control and correlated MWPM, d=3 and d=5
- Command: `python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml --out runs/m2_mlp_d3`
  and the same with `configs/m2_mlp_d5.yaml`; figure:
  `python experiments/plot_m2_sample_efficiency.py runs/m2_mlp_d3/mlp_report.json runs/m2_mlp_d5/mlp_report.json --out runs/m2_sample_efficiency.png`
- Config: unchanged from the 2026-09-30 repeated runs (d=3: 5 repeats, d=5: 3 repeats). New in
  the runner: every MLP is paired with a `LookupTableDecoder` fitted on the same training shots
  (majority vote per seen syndrome, MWPM on unseen syndromes and ties), and correlated MWPM is
  an extra reference. All LERs in the table are exact (all 2^D syndromes decoded).
- Commit: code as committed in `7d7d4b49106d6460b2bac711a75e4a4cf93eaee0` (`git_commit` in both JSONs is
  `8bf689b5de0cb219d5ce01ee330e6eea744ff934-dirty`, the same working tree as the scaling entry).
- Outputs: `runs/m2_mlp_d3/`, `runs/m2_mlp_d5/` (`mlp_report.json`, `progress.log`) and
  `runs/m2_sample_efficiency.png` (untracked; about 50 minutes and 30 minutes on one machine)
- Outcome:
  - d=3: the lookup table matches the MLP almost everywhere. The MLP is ahead in 4 of 20
    cells, by at most 0.017 in G (p = 0.05, 1e4 shots: 1.051 vs 1.068). The table is clearly
    ahead with 1e3 shots, and at low p with more data (up to 1e5 shots at p = 0.01, 1e4 at
    p = 0.02), because it falls back on MWPM where it has no data. The earlier "MLP approaches
    exact ML at d=3" result is therefore what memorising a 256-entry table gives, not evidence
    of a learned decoding rule.
  - d=5: the table barely improves on MWPM even with 1e6 shots (G 1.82, 1.69, 1.41, 1.32,
    1.26 for p = 0.01 to 0.15, against MWPM 1.84, 1.79, 1.64, 1.43, 1.28): most of the 2^24
    syndromes are never seen. The MLP does generalise: with 1e6 shots and p >= 0.05 it reaches
    G = 1.27, 1.14, 1.09, better than both the table and correlated MWPM (1.36, 1.27, 1.17).
    At p <= 0.02 it is worse than or level with MWPM (3.27 vs 1.84 at p = 0.01, 1.80 vs 1.79
    at p = 0.02). No cell reaches exact ML.
  - Correlated MWPM, exactly: worse than plain MWPM at d=3 for p <= 0.1 (G 1.269 vs 1.189 at
    p = 0.01), better from d=5 on (1.447 vs 1.842 at p = 0.01).
- Notes:
  - Not bit-reproducible: most MLP runs reproduce the 2026-09-30 numbers exactly (e.g. d=3,
    p = 0.01, 1e3 shots, repeat 0: G = 11.181 both times), but some differ in the third
    decimal (d=3, p = 0.1, 1e5 shots, repeat 0: 1.003 now, 1.000 on 2026-09-30) and so do a
    few aggregates (d=5, p = 0.01, 1e6 shots: 3.270 +- 0.326 now, 3.171 +- 0.211 then). The
    code path, data and seeds are the same. The cause is not established; a likely candidate
    is floating-point reduction order changing with the CPU threads that PyTorch/MKL use,
    amplified by best-epoch selection on the validation loss. MWPM, lookup and exact ML
    numbers are deterministic. Conclusions above do not depend on these differences.
  - The d=5 open questions from 2026-09-30 (data- vs. epoch-limited) still stand.
  - Code capacity, i.i.d. depolarizing noise, Z-basis memory, one MLP architecture per code.

d=3:
```
      p  decoder       train   exact LER  G = LER / LER_ML  (mean +- std, [min, max])     gap closed
   0.01  MWPM              -   7.795e-04  1.189                                                    -
   0.01  corr. MWPM        -   8.313e-04  1.269                                                    -
   0.01  exact ML          -   6.553e-04  1.000                                                    -
   0.01  lookup         1000   7.795e-04  1.189 +- 0.011  [1.174, 1.205]                      -0.000
   0.01  lookup        10000   7.817e-04  1.193 +- 0.026  [1.174, 1.221]                      -0.018
   0.01  lookup       100000   7.177e-04  1.095 +- 0.022  [1.064, 1.111]                       0.498
   0.01  lookup      1000000   6.577e-04  1.004 +- 0.007  [1.000, 1.016]                       0.981
   0.01  MLP            1000   2.383e-03  3.636 +- 4.222  [1.540, 11.181]                    -12.917
   0.01  MLP           10000   9.482e-04  1.447 +- 0.106  [1.348, 1.618]                      -1.359
   0.01  MLP          100000   7.386e-04  1.127 +- 0.036  [1.079, 1.175]                       0.329
   0.01  MLP         1000000   6.637e-04  1.013 +- 0.013  [1.000, 1.032]                       0.932
   0.02  MWPM              -   3.037e-03  1.179                                                    -
   0.02  corr. MWPM        -   3.230e-03  1.254                                                    -
   0.02  exact ML          -   2.575e-03  1.000                                                    -
   0.02  lookup         1000   3.053e-03  1.185 +- 0.013  [1.179, 1.210]                      -0.034
   0.02  lookup        10000   2.978e-03  1.156 +- 0.027  [1.120, 1.196]                       0.129
   0.02  lookup       100000   2.645e-03  1.027 +- 0.013  [1.015, 1.045]                       0.848
   0.02  lookup      1000000   2.576e-03  1.000 +- 0.000  [1.000, 1.001]                       0.998
   0.02  MLP            1000   4.083e-03  1.585 +- 0.132  [1.449, 1.798]                      -2.263
   0.02  MLP           10000   3.190e-03  1.239 +- 0.067  [1.135, 1.316]                      -0.332
   0.02  MLP          100000   2.652e-03  1.030 +- 0.015  [1.015, 1.045]                       0.835
   0.02  MLP         1000000   2.593e-03  1.007 +- 0.014  [1.000, 1.032]                       0.962
   0.05  MWPM              -   1.753e-02  1.152                                                    -
   0.05  corr. MWPM        -   1.850e-02  1.216                                                    -
   0.05  exact ML          -   1.521e-02  1.000                                                    -
   0.05  lookup         1000   1.742e-02  1.145 +- 0.021  [1.126, 1.178]                       0.045
   0.05  lookup        10000   1.625e-02  1.068 +- 0.032  [1.030, 1.116]                       0.554
   0.05  lookup       100000   1.527e-02  1.004 +- 0.005  [1.000, 1.012]                       0.975
   0.05  lookup      1000000   1.522e-02  1.000 +- 0.001  [1.000, 1.002]                       0.998
   0.05  MLP            1000   2.075e-02  1.364 +- 0.085  [1.263, 1.486]                      -1.389
   0.05  MLP           10000   1.599e-02  1.051 +- 0.031  [1.024, 1.090]                       0.664
   0.05  MLP          100000   1.526e-02  1.003 +- 0.005  [1.000, 1.012]                       0.979
   0.05  MLP         1000000   1.525e-02  1.002 +- 0.001  [1.000, 1.004]                       0.986
    0.1  MWPM              -   6.119e-02  1.115                                                    -
    0.1  corr. MWPM        -   6.387e-02  1.164                                                    -
    0.1  exact ML          -   5.489e-02  1.000                                                    -
    0.1  lookup         1000   6.195e-02  1.129 +- 0.033  [1.103, 1.172]                      -0.120
    0.1  lookup        10000   5.624e-02  1.025 +- 0.016  [1.006, 1.043]                       0.785
    0.1  lookup       100000   5.489e-02  1.000 +- 0.000  [1.000, 1.000]                       1.000
    0.1  lookup      1000000   5.489e-02  1.000 +- 0.000  [1.000, 1.000]                       1.000
    0.1  MLP            1000   6.624e-02  1.207 +- 0.105  [1.118, 1.376]                      -0.800
    0.1  MLP           10000   5.612e-02  1.022 +- 0.015  [1.003, 1.043]                       0.805
    0.1  MLP          100000   5.498e-02  1.002 +- 0.002  [1.000, 1.003]                       0.985
    0.1  MLP         1000000   5.501e-02  1.002 +- 0.001  [1.000, 1.003]                       0.980
   0.15  MWPM              -   1.197e-01  1.086                                                    -
   0.15  corr. MWPM        -   1.162e-01  1.054                                                    -
   0.15  exact ML          -   1.102e-01  1.000                                                    -
   0.15  lookup         1000   1.216e-01  1.103 +- 0.030  [1.080, 1.153]                      -0.205
   0.15  lookup        10000   1.120e-01  1.016 +- 0.010  [1.003, 1.031]                       0.812
   0.15  lookup       100000   1.103e-01  1.001 +- 0.002  [1.000, 1.003]                       0.992
   0.15  lookup      1000000   1.102e-01  1.000 +- 0.000  [1.000, 1.000]                       1.000
   0.15  MLP            1000   1.284e-01  1.165 +- 0.072  [1.076, 1.273]                      -0.923
   0.15  MLP           10000   1.112e-01  1.009 +- 0.005  [1.003, 1.015]                       0.900
   0.15  MLP          100000   1.103e-01  1.001 +- 0.002  [1.000, 1.003]                       0.992
   0.15  MLP         1000000   1.102e-01  1.000 +- 0.000  [1.000, 1.000]                       1.000
```

d=5:
```
      p  decoder       train   exact LER  G = LER / LER_ML  (mean +- std, [min, max])     gap closed
   0.01  MWPM              -   8.242e-05  1.842                                                    -
   0.01  corr. MWPM        -   6.476e-05  1.447                                                    -
   0.01  exact ML          -   4.475e-05  1.000                                                    -
   0.01  lookup        10000   8.239e-05  1.841 +- 0.001  [1.841, 1.842]                       0.001
   0.01  lookup       100000   8.239e-05  1.841 +- 0.002  [1.839, 1.843]                       0.001
   0.01  lookup      1000000   8.147e-05  1.821 +- 0.009  [1.814, 1.831]                       0.025
   0.01  MLP           10000   2.158e-03  48.233 +- 8.524  [39.843, 56.885]                  -55.103
   0.01  MLP          100000   3.271e-04  7.309 +- 1.767  [5.868, 9.281]                      -6.494
   0.01  MLP         1000000   1.463e-04  3.270 +- 0.326  [2.924, 3.573]                      -1.696
   0.02  MWPM              -   6.272e-04  1.788                                                    -
   0.02  corr. MWPM        -   5.012e-04  1.429                                                    -
   0.02  exact ML          -   3.508e-04  1.000                                                    -
   0.02  lookup        10000   6.637e-04  1.892 +- 0.045  [1.865, 1.945]                      -0.132
   0.02  lookup       100000   6.328e-04  1.804 +- 0.046  [1.775, 1.857]                      -0.020
   0.02  lookup      1000000   5.911e-04  1.685 +- 0.015  [1.668, 1.695]                       0.130
   0.02  MLP           10000   6.305e-03  17.974 +- 2.078  [16.448, 20.341]                  -20.540
   0.02  MLP          100000   1.246e-03  3.552 +- 0.038  [3.530, 3.596]                      -2.239
   0.02  MLP         1000000   6.325e-04  1.803 +- 0.171  [1.692, 2.000]                      -0.019
   0.05  MWPM              -   8.364e-03  1.635                                                    -
   0.05  corr. MWPM        -   6.956e-03  1.360                                                    -
   0.05  exact ML          -   5.114e-03  1.000                                                    -
   0.05  lookup        10000   8.607e-03  1.683 +- 0.017  [1.665, 1.700]                      -0.075
   0.05  lookup       100000   8.232e-03  1.610 +- 0.013  [1.597, 1.623]                       0.041
   0.05  lookup      1000000   7.203e-03  1.408 +- 0.003  [1.406, 1.412]                       0.357
   0.05  MLP           10000   2.088e-02  4.084 +- 0.132  [4.004, 4.236]                      -3.852
   0.05  MLP          100000   9.081e-03  1.776 +- 0.049  [1.723, 1.818]                      -0.221
   0.05  MLP         1000000   6.477e-03  1.266 +- 0.008  [1.261, 1.275]                       0.581
    0.1  MWPM              -   5.021e-02  1.429                                                    -
    0.1  corr. MWPM        -   4.446e-02  1.265                                                    -
    0.1  exact ML          -   3.514e-02  1.000                                                    -
    0.1  lookup        10000   5.089e-02  1.448 +- 0.011  [1.441, 1.461]                      -0.045
    0.1  lookup       100000   5.023e-02  1.429 +- 0.005  [1.424, 1.434]                      -0.001
    0.1  lookup      1000000   4.651e-02  1.324 +- 0.001  [1.323, 1.325]                       0.245
    0.1  MLP           10000   7.881e-02  2.243 +- 0.031  [2.222, 2.278]                      -1.898
    0.1  MLP          100000   4.655e-02  1.325 +- 0.013  [1.311, 1.338]                       0.243
    0.1  MLP         1000000   4.015e-02  1.143 +- 0.005  [1.137, 1.148]                       0.668
   0.15  MWPM              -   1.244e-01  1.283                                                    -
   0.15  corr. MWPM        -   1.133e-01  1.168                                                    -
   0.15  exact ML          -   9.698e-02  1.000                                                    -
   0.15  lookup        10000   1.256e-01  1.296 +- 0.002  [1.294, 1.298]                      -0.045
   0.15  lookup       100000   1.262e-01  1.301 +- 0.001  [1.300, 1.302]                      -0.064
   0.15  lookup      1000000   1.226e-01  1.264 +- 0.001  [1.263, 1.266]                       0.066
   0.15  MLP           10000   1.709e-01  1.763 +- 0.045  [1.727, 1.813]                      -1.696
   0.15  MLP          100000   1.157e-01  1.193 +- 0.006  [1.186, 1.197]                       0.318
   0.15  MLP         1000000   1.054e-01  1.087 +- 0.003  [1.084, 1.089]                       0.693
```
