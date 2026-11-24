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

### 2026-11-23: M4 first circuit-level smoke benchmark, d=3 rotated surface code
- Command: `python -m qec_certify.eval.baselines --config configs/m4_circuit_level_d3.yaml --out runs/m4_circuit_level_d3`
- Config: `configs/m4_circuit_level_d3.yaml` (Stim standard rotated-surface memory, three rounds,
  uniform Clifford/idle/measurement/reset noise, 1000 shots per cell, MWPM, correlated MWPM,
  BP+OSD and Union-Find; 20 offline batch-of-one latency shots after five warmup calls)
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records `b0e3bf4dbe678136c170b2683fcf22277ff58258-dirty`.
- Outputs: `runs/m4_circuit_level_d3/baselines.json` (untracked; regenerate with the command above).
- Outcome: the end-to-end circuit-level path runs with Stim 1.16.0, PyMatching 2.4.0 and ldpc
  2.4.1. BP+OSD has the lowest observed error count at all three p values, but 1000 shots per cell
  are too few to establish an accuracy ranking: the Wilson intervals overlap substantially.
  Union-Find has higher observed LER in this small benchmark, as expected for a speed-oriented
  baseline, but no latency conclusion is drawn from 20 offline calls.
- Notes:
  - These are whole-experiment LERs. The JSON's `ler_per_round` is an equivalent-hazard transform,
    not a fitted physical error-per-round model.
  - Performance fields are offline batch-of-one API timings including Python overhead, not
    per-round hardware streaming results.
  - Do not use this smoke run as evidence that BP+OSD is significantly better than MWPM. A later
    experiment needs paired shots, more samples and a predeclared statistical comparison.

```
  d       p  decoder               errors/shots         LER       CI
  3   0.005  mwpm                     17/1000       1.700e-02   [1.064e-02, 2.706e-02]
  3   0.005  mwpm_correlated          18/1000       1.800e-02   [1.142e-02, 2.827e-02]
  3   0.005  bp_osd                   16/1000       1.600e-02   [9.872e-03, 2.583e-02]
  3   0.005  union_find               44/1000       4.400e-02   [3.294e-02, 5.855e-02]
  3    0.01  mwpm                     53/1000       5.300e-02   [4.075e-02, 6.867e-02]
  3    0.01  mwpm_correlated          49/1000       4.900e-02   [3.726e-02, 6.419e-02]
  3    0.01  bp_osd                   48/1000       4.800e-02   [3.639e-02, 6.307e-02]
  3    0.01  union_find              107/1000       1.070e-01   [8.932e-02, 1.277e-01]
  3    0.02  mwpm                    165/1000       1.650e-01   [1.433e-01, 1.893e-01]
  3    0.02  mwpm_correlated         161/1000       1.610e-01   [1.395e-01, 1.851e-01]
  3    0.02  bp_osd                  147/1000       1.470e-01   [1.264e-01, 1.703e-01]
  3    0.02  union_find              258/1000       2.580e-01   [2.318e-01, 2.860e-01]
```

### 2026-11-23: M4 d=3/5/7 heterogeneity and drift smoke matrix
- Command: `python -m qec_certify.eval.baselines --config configs/m4_circuit_level_scaling.yaml --out runs/m4_circuit_level_scaling --workers 4`
- Config: `configs/m4_circuit_level_scaling.yaml` (p=0.01; rounds=distance; uniform or
  lognormal qubit heterogeneity 0.5, crossed with no drift or first-to-last-round drift 0.5;
  shared noise seed 20261123; 1000/500/200 shots for d=3/5/7; four decoders and 20 offline
  batch-of-one latency shots per cell)
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance.
- Outputs: `runs/m4_circuit_level_scaling/baselines.json` (untracked; 12 cells and 48 decoder
  rows, including full LER/CI and performance fields; regenerate with the command above).
- Outcome: all d=3/5/7 cells completed with four workers. The new circuit transformation is
  suitable for reproducible scaling smoke experiments. BP+OSD evaluation took about 0.4--0.5 s
  per d=3 cell, 8 s per d=5 cell and 20 s per d=7 cell in this run, compared with about 0.1 s
  or less for MWPM-family evaluation. These are whole-cell evaluation times, not streaming
  latency measurements.
- Notes:
  - This is a functionality and scaling smoke run, not a statistically powered comparison.
    Confidence intervals are wide, especially at d=7, and observed decoder rankings must not be
    presented as established advantages.
  - Heterogeneity raises the observed LER in several d=3 and d=5 cells, but the d=7 sample is
    too small for a stable conclusion. Drift cells use different sampled shots from uniform
    cells, so differences also include sampling fluctuations.
  - The heterogeneity and drift parameters are synthetic and reproducible, not fitted to a real
    device. The same `noise_seed` fixes qubit factors across configurations; test-shot seeds
    remain separated by experiment cell.
  - The complete numerical table is the generated JSON rather than a hand-copied summary.

### 2026-11-23: Causal-replay streaming latency and arrival-backlog smoke
- Command: `python -m qec_certify.eval.baselines --config configs/m4_circuit_level_scaling.yaml --out runs/m4_circuit_level_scaling --workers 4`
- Config: the same 12-cell scaling matrix as the preceding entry, plus three streaming shots
  per cell, one warmup shot and an illustrative fixed arrival period of 1000 us per round.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance.
- Outputs: `runs/m4_circuit_level_scaling/baselines.json` (regenerated with
  `streaming_performance` alongside the unchanged sampled-LER rows).
- Outcome: causal replay preserves the final offline predictions and exposes the expected
  scaling cost. Under the illustrative 1 ms/round arrival rate, MWPM-family replay meets the
  mean input rate in the uniform cells; BP+OSD does not at d=5 or d=7. Union-Find is marked
  unsupported because zero-filled causal prefixes need not be realizable DEM syndromes.
- Notes:
  - `causal_replay` redecodes the full zero-filled history after every round, so work per shot
    grows quadratically with the number of rounds. These numbers are not an optimized
    incremental decoder or hardware online measurement.
  - Throughput uses only the sum of measured decode calls; backlog fields replay those calls
    against fixed-rate arrivals. `max_pending_rounds` includes queued and in-flight rounds.
  - Only three shots per cell were timed. Percentiles and backlog values are smoke measurements,
    not stable performance claims. The 1000 us period is illustrative, not a hardware target.

Uniform-noise cells (generated from the JSON):

```
 d  decoder           p99 round latency   rounds/s   meets 1 kHz   max pending   growth/round
 3  mwpm                   0.005503 ms   210200       yes             1          0
 3  mwpm_correlated        0.004706 ms   250400       yes             1          0
 3  bp_osd                 0.048350 ms    36630       yes             1          0
 5  mwpm                   0.011670 ms   127800       yes             1          0
 5  mwpm_correlated        0.025700 ms    67100       yes             1          0
 5  bp_osd                20.720000 ms       61.53    no             13         15.25
 7  mwpm                   0.042580 ms    57860       yes             1          0
 7  mwpm_correlated        0.071900 ms    27140       yes             1          0
 7  bp_osd               110.500000 ms        9.411   no             21        105.3
```

### 2026-11-23: Fixed-window MWPM W=3, full-window correctness and boundary failure
- Command: `python -m qec_certify.eval.baselines --config configs/m4_streaming_window.yaml --out runs/m4_streaming_window --workers 3`
- Config: `configs/m4_streaming_window.yaml` (uniform circuit-level p=0.01, rounds=distance,
  W=3, MWPM and correlated MWPM; 1000/500/200 shots for d=3/5/7; 20 streaming shots after two
  warmups; illustrative 1000 us arrival period)
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance.
- Outputs: `runs/m4_streaming_window/baselines.json` (untracked; regenerate with the command above).
- Outcome: full-window d=3 matches offline exactly for both decoders. Partial windows are fast
  enough for the illustrative arrival rate, but the projected-boundary approximation causes
  severe accuracy loss at d=5 and d=7. This implementation is a measurement baseline and a
  negative result, not a viable production window decoder.
- Notes:
  - An initial implementation split `^`-separated DEM components into independent errors and
    broke correlated matching. The corrected implementation preserves whole mechanisms and
    separators; d=3 now has zero differing predictions out of 1000 for both decoders.
  - Partial-window errors below are generated from the corrected JSON. They must not be compared
    with the earlier invalid output that was overwritten.
  - The next algorithmic requirement is boundary-aware/anchored decoding or exact incremental
    matching. Further latency optimization of the naive projection is not scientifically useful.

Corrected results:

```
 d  decoder           offline errors   W=3 errors   approximation           p99/round   meets 1 kHz
 3  mwpm                61/1000         61/1000     exact_full_history       0.007919 ms   yes
 3  mwpm_correlated      55/1000         55/1000     exact_full_history       0.010480 ms   yes
 5  mwpm                 43/500         169/500      projected_dem_boundary  0.010570 ms   yes
 5  mwpm_correlated      39/500         170/500      projected_dem_boundary  0.019970 ms   yes
 7  mwpm                 21/200          98/200      projected_dem_boundary  0.016700 ms   yes
 7  mwpm_correlated      18/200          96/200      projected_dem_boundary  0.031350 ms   yes
```

### 2026-11-23: Delayed-commit boundary anchoring improves but does not rescue W=3
- Command: `python -m qec_certify.eval.baselines --config configs/m4_delayed_window.yaml --out runs/m4_delayed_window --workers 3`
- Config: the same circuits, shots and W=3 as `m4_streaming_window.yaml`, but using
  `streaming_method: delayed_commit` and matched-edge outputs. Only fully internal errors from
  the oldest window round are committed; newer and boundary-crossing components remain in the tail.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance.
- Outputs: `runs/m4_delayed_window/baselines.json` (untracked; regenerate with the command above).
- Outcome: full-window d=3 again matches offline exactly. Delayed commit reduces partial-window
  errors relative to naive projection on the same shots, but LER remains far above offline. It
  is a useful causal baseline and negative result, not a production decoder.
- Notes:
  - d=5 errors fall from 169 to 119 (MWPM) and 170 to 115 (correlated), compared with the naive
    projection. d=7 errors fall from 98 to 79 and 96 to 78 respectively.
  - Despite the improvement, delayed-commit LER is about 0.23--0.24 at d=5 and 0.39--0.40 at d=7.
    The illustrative arrival-rate test passes, but this accuracy is unacceptable.
  - The next useful measurement is a predeclared W sweep (including a full-history point) with
    paired shots and enough samples to separate approximation loss from sampling noise. Exact
    incremental matching or a better boundary state remains necessary for a production claim.

```
 d  decoder           offline errors   naive W=3      delayed W=3   delayed LER   p99/round
 3  mwpm                61/1000         61/1000        61/1000       0.061        0.01838 ms
 3  mwpm_correlated      55/1000         55/1000        55/1000       0.055        0.01936 ms
 5  mwpm                 43/500         169/500        119/500       0.238        0.03403 ms
 5  mwpm_correlated      39/500         170/500        115/500       0.230        0.04513 ms
 7  mwpm                 21/200          98/200         79/200       0.395        0.05351 ms
 7  mwpm_correlated      18/200          96/200         78/200       0.390        0.07379 ms
```

### 2026-11-23: Predeclared paired W sweep rejects partial delayed-commit windows
- Commands: `python -m qec_certify.eval.baselines --config configs/m4_window_sweep.yaml --out runs/m4_window_sweep --workers 4` and
  `python experiments/analyze_m4_window_sweep.py runs/m4_window_sweep/baselines.json --out runs/m4_window_sweep/paired_analysis.json`
- Config: `configs/m4_window_sweep.yaml`, fixed before running. d=5 uses W=2/3/4/5 with 5000
  shots per width; d=7 uses W=2/3/4/5/7 with 2000 shots. W=distance is the full-history
  reference. Every width at a distance reuses the same shots and includes MWPM and correlated MWPM.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance.
- Outputs: `runs/m4_window_sweep/baselines.json` and `paired_analysis.json` (untracked; regenerate
  with the commands above). Analysis uses exact two-sided McNemar tests, unadjusted for multiple comparisons.
- Outcome: larger W reduces approximation loss, but every partial window remains significantly
  worse than full history. Full-history rows have zero disagreements. The delayed-commit heuristic
  is therefore stopped as a production candidate and retained only as a causal negative baseline.
- Notes:
  - This conclusion is based on paired per-shot transitions, not overlapping independent CIs.
    `full_correct_window_wrong` and `full_wrong_window_correct` are both stored in the baseline JSON.
  - At d=7, even W=5 gives LER 0.3080/0.3010 for plain/correlated MWPM against full-history
    0.1035/0.0880. Spending more effort on this boundary heuristic is not justified by the scan.
  - Latency is still measured on only 20 streaming shots per cell and is not the basis of this rejection.

Generated paired analysis (p-values are unadjusted):

```
 d  w  decoder           full LER  window LER   full->W  W->full   p-value
 5  2  mwpm               0.0828      0.2798      1165       180  8.04e-177
 5  3  mwpm               0.0828      0.2376       918       144  1.93e-138
 5  4  mwpm               0.0828      0.1800       566        80  4.45e-91
 5  5  mwpm               0.0828      0.0828         0         0  1.00e+00
 5  2  mwpm_correlated    0.0714      0.2754      1193       173  1.22e-187
 5  3  mwpm_correlated    0.0714      0.2372       971       142  2.58e-152
 5  4  mwpm_correlated    0.0714      0.1760       616        93  1.76e-95
 5  5  mwpm_correlated    0.0714      0.0714         0         0  1.00e+00
 7  2  mwpm               0.1035      0.4385       765        95  7.83e-131
 7  3  mwpm               0.1035      0.3845       677       115  1.18e-97
 7  4  mwpm               0.1035      0.3660       616        91  1.20e-96
 7  5  mwpm               0.1035      0.3080       488        79  6.28e-73
 7  7  mwpm               0.1035      0.1035         0         0  1.00e+00
 7  2  mwpm_correlated    0.0880      0.4385       775        74  3.68e-148
 7  3  mwpm_correlated    0.0880      0.3890       695        93  1.02e-114
 7  4  mwpm_correlated    0.0880      0.3470       596        78  1.08e-99
 7  5  mwpm_correlated    0.0880      0.3010       499        73  5.42e-79
 7  7  mwpm_correlated    0.0880      0.0880         0         0  1.00e+00
```

### 2026-11-23: Public qLDPC BB and gross-code Relay-BP smoke benchmarks
- Commands: `python -m qec_certify.eval.qldpc --config configs/qldpc_bb18.yaml --out runs/qldpc_bb18` and
  `python -m qec_certify.eval.qldpc --config configs/qldpc_gross.yaml --out runs/qldpc_gross`
- Config: `configs/qldpc_bb18.yaml` (1000 shots, BP+OSD product-sum with at most 100 iterations
  and OSD-CS order 3, versus Relay-BP with 60 sets) and `configs/qldpc_gross.yaml` (1000 shots,
  Relay-BP only with the same 60-set settings). Both use 20 offline latency shots after five warmups.
- Reference data: public Relay-BP test circuits at upstream commit
  `72341ef85cd45779e6483c338aef7ba85094eac1`, pinned with SHA-256 in
  `src/qec_certify/codes/data/PROVENANCE.md`. Relay-BP is not a supported IBM product.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; each JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_bb18/qldpc.json` and `runs/qldpc_gross/qldpc.json` (untracked; regenerate
  with the commands above).
- Outcome: on BB [[18,4,3]], BP+OSD has 40/1000 errors and Relay-BP 33/1000. The paired
  comparison has 12 BP-correct/Relay-wrong and 19 BP-wrong/Relay-correct shots, exact McNemar
  p=0.2810, so no accuracy advantage is established. Relay-BP takes 0.92 s for the 1000-shot
  evaluation versus 11.38 s for BP+OSD in this run, but these decoder configurations are not a
  pre-registered paper reproduction.
- Gross outcome: Relay-BP on the public gross [[144,12,12]] circuit has 0/1000 errors, with Wilson
  95% interval [2.168e-19, 3.827e-03]. Zero observed errors is not evidence of zero LER. The run
  takes 286.74 s; among the 20 timing shots, median whole-shot latency is 144.76 ms and p99 is
  1794.08 ms. Default-parameter CPU latency is therefore far from a real-time claim.
- Notes:
  - Latency fields cover an entire 3- or 12-round shot, include Python overhead and are not
    per-round hardware-streaming measurements. Twenty timing shots are smoke measurements only.
  - The small-code accuracy comparison is paired; gross-code BP+OSD was not run because a
    practical high-effort budget has not been defined. Do not claim a gross-code accuracy win.
  - These public reference circuits and decoder implementations are external scientific assets,
    not real hardware data and not a supported IBM product.

```
 code             decoder    errors/shots   LER         CI                         p50/shot    p99/shot
 BB [[18,4,3]]    bp_osd       40/1000      0.040       [0.02951, 0.05401]       8.64 ms    36.43 ms
 BB [[18,4,3]]    relay_bp     33/1000      0.033       [0.02359, 0.04598]       0.288 ms    8.93 ms
 gross [[144,12,12]] relay_bp   0/1000      0.000       [2.168e-19, 0.003827]   144.76 ms  1794.08 ms
```

### 2026-11-23: Validation-only Relay-BP selection and independent gross-code test
- Command: `python -m qec_certify.eval.qldpc_tune --config configs/qldpc_gross_tune.yaml --out runs/qldpc_gross_tune`
- Config: `configs/qldpc_gross_tune.yaml`, fixed before running. Three Relay-BP candidates:
  `paper_like` (gamma0=0.125, 80 + 60x60 iteration budget), `fast` (40 + 20x40) and `wide`
  (gamma interval [-0.4, 0.8], 80 + 60x60). Selection uses only 50 validation shots and five
  latency shots: fewest errors, then mean latency, iteration budget and name. The independent
  test uses 300 shots with Relay-BP and practical BP+OSD-0 (product-sum, at most 100 iterations).
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_gross_tune/tuning.json` (untracked; regenerate with the command above).
- Outcome: `paper_like` is selected with 0/50 validation errors, versus 1/50 for `fast` and 3/50
  for `wide`. On the independent test, Relay-BP has 0/300 errors and BP+OSD-0 has 27/300.
  All 27 prediction disagreements are BP-wrong/Relay-correct; exact McNemar
  p=1.4901e-08. Relay-BP finishes the test evaluation in 82.23 s, compared with 508.25 s for
  BP+OSD-0.
- Notes:
  - This establishes an advantage over the **specific practical BP+OSD-0 configuration**, not over
    the paper's high-effort BP+OSD-CS-10. No such stronger baseline was run.
  - Zero Relay-BP errors has Wilson 95% interval [0, 0.01264]; it does not imply zero LER.
  - Relay-BP mean/p50/p99 whole-shot latency is 260.81/164.14/1020.02 ms, versus
    1540.02/1405.02/2364.41 ms for BP+OSD-0. Twenty timing shots are smoke measurements, and all
    latencies include 12 rounds and Python overhead, not hardware streaming.
  - A single easy-shot probe took about 0.24 s for BP+OSD-0, but the 300-shot evaluation took
    508.25 s, demonstrating a heavy tail and the need for batch budget measurements.
  - Validation and test seeds are distinct and recorded in the JSON; test shots were not used for selection.

```
 candidate   validation errors   mean validation latency   iteration budget
 paper_like  0/50                86.79 ms                 3680
 fast        1/50               273.61 ms                  840
 wide        3/50              1365.65 ms                 3680

 test decoder  errors/shots   LER         CI                  mean/shot   p99/shot
 relay_bp      0/300          0.000       [0, 0.01264]       260.81 ms   1020.02 ms
 bp_osd        27/300         0.090       [0.06259, 0.12778] 1540.02 ms  2364.41 ms
```

### 2026-11-24: Gross-code Relay-BP versus BP+LSD across p and independent seeds
- Commands: `python -m qec_certify.eval.qldpc_sweep --config configs/qldpc_robustness.yaml --out runs/qldpc_robustness` and
  `python experiments/analyze_qldpc_sweep.py runs/qldpc_robustness/sweep.json --out runs/qldpc_robustness/analysis.json`
- Config: `configs/qldpc_robustness.yaml`, fixed before running. Public gross [[144,12,12]]
  circuits at p=0.001/0.003/0.005, experiment seeds 20261124/20261125, 200 shots per cell.
  Relay-BP is fixed to the earlier validation-selected `paper_like` settings with no retuning;
  BP+LSD uses product-sum, at most 100 iterations and LSD-0. Ten offline latency shots per cell.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; each JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_robustness/cells.jsonl`, `sweep.json` and `analysis.json` (untracked;
  regenerate with the commands above). Every completed cell is checkpointed, and reruns skip it.
- Outcome: the trend is consistent across both seeds. At p=0.001 both methods have zero errors
  and zero disagreements. At p=0.003 Relay-BP has zero errors in both cells and significantly
  fewer errors than BP+LSD. At p=0.005 Relay-BP has 34/200 and 32/200 errors, compared with
  99/200 and 112/200 for BP+LSD. All p=0.003 and p=0.005 paired tests favor Relay-BP.
- Notes:
  - This supports robustness of the fixed `paper_like` configuration over the tested p and seeds;
    it does not establish superiority over high-effort BP+OSD-CS-10 or other decoders.
  - Mean whole-shot Relay-BP latency ranges from 61 ms to 2597 ms across cells, with large
    heavy-tail and seed variation. BP+LSD mean latency ranges from 693 ms to 4006 ms. These are
    offline 12-round CPU measurements, not real-time or per-round hardware results.
  - The first 15-minute attempt timed out during the third cell. The runner was changed to write
    a checkpoint after every complete cell, and the full original 200-shot matrix was rerun without
    a timeout. Partial decoder output from the killed run was not reused.
  - McNemar p-values are per-cell and unadjusted for multiple comparisons. Pooled LERs below are
    descriptive combinations of two seeds, not a replacement for the paired per-cell tests.

```
 p      seed       BP+LSD errors   Relay errors   McNemar p     BP mean/shot   Relay mean/shot
 0.001  20261124   0/200           0/200          1.0           734 ms         100 ms
 0.001  20261125   0/200           0/200          1.0           693 ms          61 ms
 0.003  20261124   8/200           0/200          0.0078125    1201 ms         214 ms
 0.003  20261125  11/200           0/200          0.0009766    1421 ms         241 ms
 0.005  20261124  99/200          34/200          3.42e-15     3763 ms        2597 ms
 0.005  20261125 112/200          32/200          3.69e-22     4006 ms         823 ms
```

Descriptive pooled results:

```
 p      decoder    errors/shots   pooled LER   Wilson 95% CI
 0.001  bp_lsd     0/400          0            [0, 0.00951]
 0.001  relay_bp   0/400          0            [0, 0.00951]
 0.003  bp_lsd     19/400         0.0475       [0.0306, 0.0730]
 0.003  relay_bp   0/400          0            [0, 0.00951]
 0.005  bp_lsd     211/400        0.5275       [0.479, 0.576]
 0.005  relay_bp   66/400         0.165        [0.132, 0.205]
```

### 2026-11-24: A 100 ms Relay-BP latency budget meets speed but fails accuracy
- Command: `python -m qec_certify.eval.qldpc_tune --config configs/qldpc_realtime_tune.yaml --out runs/qldpc_realtime_tune`
- Config: `configs/qldpc_realtime_tune.yaml`, fixed before running. Gross [[144,12,12]] at
  p=0.003, 50 validation shots with ten timing shots, validation p99 budget 100 ms, and five
  Relay-BP candidates (1/5/10/20/60 ensembles). Eligible candidates are ranked by errors first;
  if none are eligible, the lowest-p99 candidate is marked `met: false`. Independent test uses
  200 shots and BP+LSD-0 as the paired baseline.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_realtime_tune/tuning.json` (untracked; regenerate with the command above).
- Outcome: only `single` meets the validation p99 budget (75.48 ms), so it is selected with
  `latency_budget.met: true`. On the independent test, however, Relay-BP has 134/200 errors
  (LER 0.670) versus 8/200 for BP+LSD (LER 0.040). BP-correct/Relay-wrong occurs 127 times and
  the reverse once, McNemar p=7.58e-37. The speed constraint is met, but accuracy is unacceptable.
- Notes:
  - This is a negative accuracy--latency result, not a real-time demonstration. Meeting
    `latency_budget.met` means only that the predeclared latency rule passed; it does not imply
    any accuracy or industrial requirement passed.
  - Validation already showed the accuracy collapse of small ensembles: 40/50, 23/50 and 1/50
    errors for single/five/ten, versus 0/50 for twenty and paper_like. Their p99 latencies all
    exceeded 100 ms, leaving `single` as the only eligible candidate. The selection rule was not
    changed after seeing this outcome.
  - Test mean/p50/p99 Relay-BP latency is 70.84/71.52/74.97 ms, versus
    1417.99/1195.24/3996.08 ms for BP+LSD. Relay is faster but not useful at this accuracy.
  - A future study must predeclare an accuracy constraint as well as latency, or report the full
    Pareto frontier. It must not silently relax the accuracy requirement after the experiment.

```
 candidate   validation errors   validation p99   iteration budget   meets 100 ms
 single      40/50               75.48 ms         40                 yes
 five        23/50              224.36 ms        120                 no
 ten          1/50              666.31 ms        440                 no
 twenty       0/50              859.52 ms        840                 no
 paper_like   0/50              338.70 ms       3680                 no

 test decoder  errors/shots   LER   CI                    mean/shot   p99/shot
 relay_bp      134/200        0.670 [0.6022, 0.7314]       70.84 ms   74.97 ms
 bp_lsd        8/200          0.040 [0.02041, 0.07693]    1417.99 ms  3996.08 ms
```

### 2026-11-24: Relay-BP Pareto frontier shows an accuracy cliff, not a real-time sweet spot
- Command: `python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_pareto.yaml --out runs/qldpc_pareto`
- Config: `configs/qldpc_pareto.yaml`, fixed before running. Gross [[144,12,12]] at p=0.003,
  50 validation shots and 200 independent test shots. Validation selects all candidates
  nondominated by error count and p99 latency. All frontier candidates share the same test shots
  and one BP+LSD-0 baseline evaluation.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_pareto/pareto.json` (untracked; regenerate with the command above).
- Outcome: the validation Pareto front is `paper_like`, `five` and `single`. On test, `single`
  has LER 0.670 with p99 72.87 ms, `five` has LER 0.385 with p99 226.92 ms, and `paper_like`
  has zero errors with p99 789.13 ms. There is no tested point that combines low LER with a
  100 ms-level p99. This is an accuracy cliff, not a real-time sweet spot.
- Notes:
  - `paper_like` versus BP+LSD: 0/200 against 8/200, all eight disagreements favor Relay-BP,
    McNemar p=0.0078125. BP+LSD p99 is 3912.25 ms. Accuracy and speed both favor `paper_like`
    over this baseline, but not a 100 ms target.
  - `single` and `five` are significantly worse than BP+LSD in accuracy (p=7.58e-37 and
    5.72e-19 respectively), so their lower latency is not useful.
  - `ten` and `twenty` were excluded before test because `paper_like` dominates them on the
    validation error count and p99. They were not removed based on test results.
  - A natural next implementation question is whether fixed-point/low-precision Relay-BP can
    preserve `paper_like` accuracy while reducing latency; reducing ensembles has already failed.

Validation:

```
 candidate   errors   p99        test errors   test LER   test p99    McNemar vs BP+LSD
 single      40/50    72.66 ms   134/200       0.670      72.87 ms    7.58e-37 (worse)
 five        23/50   214.19 ms    77/200       0.385     226.92 ms    5.72e-19 (worse)
 ten          1/50   645.59 ms    not run      dominated   --          --
 twenty       0/50   846.87 ms    not run      dominated   --          --
 paper_like   0/50   317.62 ms     0/200       0.000     789.13 ms    0.0078125 (better)
```

Baseline BP+LSD: 8/200, LER 0.040, mean 1348.24 ms, p99 3912.25 ms.

### 2026-11-24: Integer Relay-BP preserves decisions and reduces gross-code latency
- Commands: `python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_precision.yaml --out runs/qldpc_precision` and
  `python experiments/analyze_qldpc_precision.py runs/qldpc_precision/pareto.json --out runs/qldpc_precision/implementation_pairs.json`
- Config: `configs/qldpc_precision.yaml`, fixed before running. Gross [[144,12,12]] at p=0.003,
  identical `paper_like` algorithm settings for F32, I32 and I64; I32/I64 use the upstream
  `data_scale_value=8` and `max_data_value=65535`. `evaluate_all` tests every implementation on
  the same 200 shots with 20 timing shots and one BP+LSD-0 baseline.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_precision/pareto.json` and `implementation_pairs.json` (untracked;
  regenerate with the commands above).
- Outcome: all three implementations have 0/200 errors and decode success rate 100%. The separate
  per-shot comparison finds zero disagreements for F32-vs-I32, F32-vs-I64 and I32-vs-I64.
  I32 has the lowest mean and p99 latency in this run.
- Notes:
  - Mean iterations are 134.02 for F32 and 135.49 for the integer implementations, so the speedup
    is not explained by doing dramatically less work. Maximum iterations are 3287 and 2717.
  - All three have eight BP+LSD-wrong/Relay-correct transitions and McNemar p=0.0078125 over the
    shared BP+LSD-0 baseline. Accuracy comparisons between implementations show exact agreement,
    not merely equal aggregate LER.
  - A small synthetic DEM exposed unsuccessful integer candidates during development. The wrapper
    now uses detailed results, falls back to no-flip on failure and records failure rates; no
    unsuccessful candidate is returned as a valid correction. All 200 gross shots succeed here.
  - These integer/scaled implementations are not an FPGA fixed-point proof. Results are limited to
    this circuit, p, seed and `paper_like` settings, and require cross-p/seed validation.

```
 implementation  errors   success   mean iterations   mean/shot   p99/shot   decisions vs others
 f32             0/200    100%      134.02            234.44 ms   799.47 ms  0/200 disagreements
 i32_scaled      0/200    100%      135.49            155.59 ms   290.93 ms  0/200 disagreements
 i64_scaled      0/200    100%      135.49            169.55 ms   315.78 ms  0/200 disagreements
```

### 2026-11-24: I32 decision equivalence and speed advantage break down at p=0.005
- Command: `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_precision_robustness.yaml --out runs/qldpc_precision_robustness`
- Config: `configs/qldpc_precision_robustness.yaml`, fixed before running. Gross [[144,12,12]]
  at p=0.001/0.003/0.005, experiment seeds 20261124/20261125, 200 shots per cell. F32 and
  I32 scaled use identical `paper_like` settings; each cell samples shots once and compares both
  implementations on those shots, with ten latency shots.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_precision_robustness/cells.jsonl` and `precision_sweep.json` (untracked;
  regenerate with the command above). Each completed cell is checkpointed.
- Outcome: p=0.001 and p=0.003 retain zero prediction disagreements and 100% decode success.
  At p=0.005 there are 20 and 25 disagreements, so exact decision equivalence does not hold.
  I32 has fewer observed errors (28 vs 37 and 37 vs 43), but McNemar p=0.0636 and 0.3075 does
  not establish an accuracy improvement. I32 is also slower at p=0.005, with much worse tails.
- Notes:
  - At p=0.005, F32 success rates are 0.82/0.79 and I32 rates are 0.86/0.815. Failed decodes
    fall back to no-flip and remain separately reported; LER must not be interpreted without
    these success rates.
  - F32 mean/p99 latency at p=0.005 is 2556/6530 ms and 858/3399 ms; I32 is 3399/8565 ms and
    1244/7432 ms. The low-p speed advantage does not extrapolate to high p.
  - These results motivate a failure-aware fallback (for example BP+LSD on failed shots) or
    validation-only quantization tuning, not a claim that integer implementations are universally
    equivalent or faster.

```
 p      seed       F32 errors/success   I32 errors/success   disagreements   p-value   F32 mean/p99     I32 mean/p99
 0.001  20261124   0/200, 100%          0/200, 100%          0               1.0       103/364 ms       100/287 ms
 0.001  20261125   0/200, 100%          0/200, 100%          0               1.0        64/97 ms        80/143 ms
 0.003  20261124   0/200, 100%          0/200, 100%          0               1.0       204/441 ms       160/283 ms
 0.003  20261125   0/200, 100%          0/200, 100%          0               1.0       247/698 ms       267/672 ms
 0.005  20261124  37/200, 82%          28/200, 86%           20              0.0636    2556/6530 ms    3399/8565 ms
 0.005  20261125  43/200, 79%          37/200, 81.5%         25              0.3075     858/3399 ms    1244/7432 ms
```

### 2026-11-24: Failure-aware BP+LSD fallback improves accuracy but adds a large tail
- Command: `python -m qec_certify.eval.qldpc --config configs/qldpc_fallback.yaml --out runs/qldpc_fallback`
- Config: `configs/qldpc_fallback.yaml`, fixed before running. Gross [[144,12,12]] at p=0.005,
  seed 20261124, 200 shared shots. Compare BP+LSD-0, F32 Relay-BP with no-flip failure handling,
  and Relay-BP with BP+LSD invoked only on unsuccessful shots. Relay uses fixed `paper_like`
  parameters. Ten offline latency shots per decoder.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_fallback/qldpc.json` (untracked; regenerate with the command above).
- Outcome: Relay fails on 36/200 shots (18%). Fallback corrects seven previously wrong shots and
  introduces zero new errors, reducing LER from 0.185 to 0.150; exact McNemar p=0.015625 versus
  no fallback. The hybrid also beats BP+LSD-0, with all 69 baseline-wrong/hybrid-correct
  transitions in its favor and p=3.39e-21. Accuracy improves, but the latency tail is much worse.
- Notes:
  - Hybrid mean/p99 latency is 4276/14225 ms, versus 2512/6387 ms for no-fallback Relay and
    3784/8525 ms for BP+LSD. The ten-shot timing sample shows no stable latency advantage.
  - Total evaluation time is 667.94 s for hybrid, 868.87 s for BP+LSD and 439.88 s for Relay.
    Fallback uses BP+LSD on only 18% of shots, but its measured tail remains unacceptable for
    real-time claims. Total-time and ten-shot latency summaries describe different samples and
    should not be conflated.
  - A useful next step is a cheaper/faster fallback or a timeout/parallel fallback policy, not
    simply increasing the fallback rate.

```
 decoder             errors   LER     Relay failures   fallback calls   mean/shot   p99/shot   total evaluation
 bp_lsd              99/200   0.495   --               --               3784 ms     8525 ms    868.87 s
 relay_bp            37/200   0.185   36 (18%)         0                2512 ms     6387 ms    439.88 s
 relay_bp_fallback   30/200   0.150   36 (18%)         36 (18%)         4276 ms    14225 ms    667.94 s
```

### 2026-11-24: Short Relay retry is cheaper than BP fallback but not significantly better
- Command: `python -m qec_certify.eval.qldpc --config configs/qldpc_retry.yaml --out runs/qldpc_retry`
- Config: `configs/qldpc_retry.yaml`, fixed before running. Gross [[144,12,12]] at p=0.005,
  seed 20261124, 200 shared shots. Primary F32 Relay uses fixed `paper_like` settings; only its
  failed shots run a shorter F32 Relay retry with an independent seed and a 40 + 20x40 budget.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_retry/qldpc.json` (untracked; regenerate with the command above).
- Outcome: primary fails 36/200 shots. Retry fails 31 of those 36, leaving a final failure rate
  of 15.5% versus 18% for primary. It corrects five previously wrong shots and introduces zero
  new errors, reducing LER from 0.185 to 0.160. Exact McNemar p=0.0625, so the accuracy
  improvement is not statistically established.
- Notes:
  - Retry mean/p99 latency is 2948/8056 ms, versus 2575/6588 ms for no retry. Total evaluation
    time is 495.96 s versus 440.67 s. The retry adds cost rather than solving the latency problem.
  - The previous BP+LSD fallback had LER 0.150, p=0.015625 over no fallback, and total evaluation
    time 667.94 s. Relay retry is cheaper but has weaker evidence and worse observed LER.
  - This closes the short-retry configuration as a clear production recommendation. Further work
    needs a better algorithm, validation-only retry/quantization tuning, or hardware/software
    integration measurements rather than another uncontrolled fallback variant.

```
 decoder             errors   LER     final failure rate   mean/shot   p99/shot   total evaluation
 relay_bp            37/200   0.185   18%                  2575 ms     6588 ms    440.67 s
 relay_bp_fallback   32/200   0.160   15.5%                2948 ms     8056 ms    495.96 s
```

### 2026-11-24: Fresh-state serial and parallel Relay agree; parallelism improves throughput only
- Command: `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_parallel.yaml --out runs/qldpc_parallel`
- Config: `configs/qldpc_parallel.yaml`, fixed before running. Gross [[144,12,12]] at
  p=0.003/0.005, seed 20261124, 200 shared shots per cell. Serial and Rayon-parallel Relay use
  identical F32 `paper_like` parameters, with ten batch-of-one timing shots per decoder.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; the JSON
  records the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_parallel/cells.jsonl` and `precision_sweep.json` (untracked; regenerate
  with the command above).
- Outcome: serial and parallel have zero prediction disagreements and identical success rates and
  iteration counts at both p values. Parallel reduces whole-batch evaluation time from 63.08 s
  to 42.66 s at p=0.003 (about 1.48x) and from 436.45 s to 325.91 s at p=0.005 (about 1.34x).
- Notes:
  - Batch-of-one mean/p99 latency is 251/544 ms serial versus 226/529 ms parallel at p=0.003,
    and 2052/6306 ms versus 2061/6446 ms at p=0.005. There is no meaningful single-shot latency
    improvement, so this is a throughput result, not a real-time tail-latency result.
  - The public serial batch path was found to carry state between shots. The project now constructs
    a fresh native decoder per shot for serial decoding and uses independent runner clones for
    parallel decoding. Fresh, serial, reversed-order and parallel predictions are regression-tested.
  - Earlier Relay batch results were produced before this state-carryover fix and should be treated
    as requiring rerun for independent-shot semantics. The present fresh-state runs are the
    reference for subsequent comparisons; old numerical entries are retained as historical logs,
    not silently rewritten.

```
 p      mode      errors   success   mean iterations   batch time   batch1 mean   batch1 p99
 0.003  serial    0/200    100%      147.045           63.08 s      251 ms        544 ms
 0.003  parallel  0/200    100%      147.045           42.66 s      226 ms        529 ms
 0.005  serial    37/200   81.5%     1238.86          436.45 s     2052 ms       6306 ms
 0.005  parallel  37/200   81.5%     1238.86          325.91 s     2061 ms       6446 ms
```

### 2026-11-24: Fresh-state regeneration of the core Relay benchmarks
- Commands: `python -m qec_certify.eval.qldpc_tune --config configs/qldpc_gross_tune.yaml --out runs/qldpc_gross_tune_fresh`,
  `python -m qec_certify.eval.qldpc --config configs/qldpc_bb18.yaml --out runs/qldpc_bb18_fresh`, and
  `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_core_fresh.yaml --out runs/qldpc_core_fresh`
- Config: existing gross-tuning and BB18 configs, plus `configs/qldpc_core_fresh.yaml` for
  fresh F32/I32 at p=0.003/0.005, seed 20261124, 200 shared shots and ten latency shots.
- Commit: code as uncommitted changes on `b0e3bf4dbe678136c170b2683fcf22277ff58258`; new JSONs
  record the corresponding `-dirty` provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_gross_tune_fresh/tuning.json`, `runs/qldpc_bb18_fresh/qldpc.json`, and
  `runs/qldpc_core_fresh/precision_sweep.json` (untracked; regenerate with the commands above).
- Outcome: gross selection and final paired result are unchanged: `paper_like`, Relay 0/300 and
  BP+OSD-0 27/300, p=1.49e-08. Fresh BB18 Relay has 36/1000 rather than the old 33/1000; paired
  p=0.5966, so the no-significant-accuracy-advantage conclusion is unchanged. F32/I32 retain exact
  agreement at p=0.003 but disagree at p=0.005, where I32 remains slower and not significantly
  more accurate.
- Notes:
  - Fresh p=0.005 errors are F32 37/200 and I32 30/200, with 19 disagreements and McNemar
    p=0.1671. The old stateful I32 result was 28/200. The fresh result supersedes it.
  - Fresh p=0.003 F32/I32 have 0/200 errors, 100% success and zero prediction disagreements.
    I32 batch-of-one timing is lower in this small sample, but its total evaluation time is
    72.50 s versus 63.77 s for F32; no uniform speedup claim is made from these mixed summaries.
  - Fresh p=0.005 F32 mean/p99 latency is 2049/6286 ms and I32 is 2521/8335 ms; total evaluation
    time is 437.57 s versus 507.57 s. High-p integerization is neither equivalent nor faster here.
  - All old stateful-batch numerical entries remain historical logs only. The `_fresh` outputs and
    fresh-state parallel run are the current reference for subsequent claims.

```
 benchmark       old result                    fresh result                    conclusion
 gross tuning    0/300 vs 27/300, same pairing  same selected candidate/result  unchanged
 BB18 Relay      33/1000, p=0.2810              36/1000, p=0.5966              no significant advantage
 F32/I32 p=.003  0/200 each, zero disagreements  0/200 each, zero disagreements exact at low p
 F32/I32 p=.005  37/200 vs 28/200               37/200 vs 30/200, p=0.1671     not equivalent; not significant
```

### 2026-11-24: Relay profiling identifies native BP work and nonconvergence as the bottleneck
- Command: `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_profile.yaml --out runs/qldpc_profile`
- Config: `configs/qldpc_profile.yaml`, fixed before running. Gross [[144,12,12]] at
  p=0.003/0.005, seed 20261124, 100 evaluation shots and 20 profiled batch-of-one shots per
  decoder. Serial uses a fresh native decoder per shot; parallel uses independent runner clones.
  Both use identical F32 `paper_like` parameters.
- Commit: code as uncommitted changes on `2381ff5` after the baseline commit; the JSON records
  the corresponding dirty provenance.
- Outputs: `runs/qldpc_profile/cells.jsonl` and `precision_sweep.json` (untracked; regenerate
  with the command above).
- Outcome: native message passing dominates latency. At p=0.005 all three failed profiled shots
  use exactly 3680 iterations and have mean latency 6628 ms serial/6574 ms parallel. Successful
  shots also have a large tail, with up to 3608 iterations. Python overhead is below 0.5 ms and
  serial initialization is about 37 ms, so neither is the main bottleneck.
- Notes:
  - At p=0.003 serial mean latency is 247.86 ms, comprising about 36.80 ms initialization,
    210.71 ms native decode and 0.34 ms remaining overhead. The iteration p99/max is 554/603.
  - At p=0.005 serial mean latency is 2460.96 ms, with 2423.97 ms native decode, 36.63 ms
    initialization and 0.36 ms remaining overhead. Iteration p99/max is 3680/3680.
  - Parallel batch-of-one latency is essentially unchanged, as expected. Whole-batch evaluation
    is faster: 22.15 s versus 31.43 s at p=0.003 and 200.83 s versus 260.69 s at p=0.005.
    This is again throughput, not lower tail latency.
  - Optimization should target native iteration cost, nonconvergence and successful-shot tails.
    Python-only optimization or removing fresh initialization cannot meet the latency problem.

```
 p      mode      iterations p99/max   failure iterations   mean latency   init mean   native mean   Python mean
 0.003  serial    554/603             --                 247.86 ms      36.80 ms    210.71 ms     0.34 ms
 0.003  parallel  554/603             --                 210.60 ms      --          210.55 ms     0.05 ms
 0.005  serial    3680/3680           3680 (all failures) 2460.96 ms     36.63 ms   2423.97 ms    0.36 ms
 0.005  parallel  3680/3680           3680 (all failures) 2425.60 ms     --         2425.53 ms    0.07 ms
```

### 2026-11-24: Early BP+LSD takeover removes final failures but not the latency tail
- Command: `python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_early_fallback.yaml --out runs/qldpc_early_fallback`
- Config: `configs/qldpc_early_fallback.yaml`, fixed before running. Gross [[144,12,12]] at
  p=0.005, seed 20261124, 20 validation shots and 100 shared test shots. Compare 10/20/60
  Relay ensembles followed by BP+LSD-0 only on unsuccessful Relay shots, against one BP+LSD-0
  baseline evaluation. Ten latency shots per decoder.
- Commit: code as uncommitted changes on `2381ff5`; the JSON records the corresponding dirty
  provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_early_fallback/pareto.json` (untracked; regenerate with the command above).
- Outcome: all hybrids have zero final decode failures and significantly fewer errors than the
  BP+LSD baseline. More Relay work improves hybrid accuracy, but increases the tail. Ten/fallback
  has the lowest p99 among hybrids, yet 58% of shots need BP+LSD and its p99 is approximately
  the same as the baseline. The early-takeover hypothesis does not solve the latency problem.
- Notes:
  - BP+LSD baseline has 53/100 errors, mean/p99 4178/6486 ms. The hybrids have 35/100, 27/100
    and 12/100 errors, with 58%, 46% and 19% primary failure rates respectively.
  - Hybrid mean/p99 latencies are 2513/6476 ms, 2490/7475 ms and 3067/11873 ms. Accuracy and
    latency remain strongly coupled; no production sweet spot is established.
  - All baseline-wrong/hybrid-correct transitions favor the hybrids, with McNemar p values
    7.63e-06, 2.98e-08 and 9.09e-13. These accuracy gains do not imply real-time suitability.
  - Further work should target native iteration cost or a fundamentally better nonconvergence
    strategy, not continue multiplying Relay-plus-BP fallback budgets.

```
 candidate         errors   LER    primary failure rate   final failures   mean/shot   p99/shot   McNemar vs BP
 ten_fallback      35/100   0.35   58%                    0               2513 ms     6476 ms    7.63e-06
 twenty_fallback   27/100   0.27   46%                    0               2490 ms     7475 ms    2.98e-08
 paper_fallback    12/100   0.12   19%                    0               3067 ms    11873 ms   9.09e-13
```

### 2026-11-24: Alpha=0.90 improves average convergence but not the failure-driven tail
- Command: `python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_alpha.yaml --out runs/qldpc_alpha`
- Config: `configs/qldpc_alpha.yaml`, fixed before running. Gross [[144,12,12]] at p=0.005,
  seed 20261124, 50 validation shots and 200 shared test shots. Compare alpha=None, 0.95, 0.90
  and an iteration ramp (alpha=0, scaling factor 10), all with otherwise identical F32
  `paper_like` settings. `evaluate_all` includes every candidate on test. BP+LSD-0 is the shared
  baseline; 20 latency/profile shots per candidate.
- Commit: code as uncommitted changes on `2381ff5`; the JSON records the corresponding dirty
  provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_alpha/pareto.json` (untracked; regenerate with the command above).
- Outcome: alpha=0.90 gives the best accuracy and average convergence in this cell: 18/200 errors,
  91% success and 806.275 mean iterations, compared with alpha=None's 37/200, 81.5% and 1238.86.
  Mean latency falls from 1641.93 ms to 1046.35 ms. However, p99 remains about 6.3 s because
  failed decodes still exhaust the 3680-iteration budget.
- Notes:
  - Alpha=0.95 gives 21/200 errors and 89.5% success; ramp gives 26/200 and 87%. Alpha=0.90 is
    the most promising candidate in this cell, but requires cross-p/seed validation.
  - All candidates significantly outperform the shared BP+LSD-0 baseline in accuracy, with paired
    p values from 6.42e-14 to 1.42e-21. This does not imply real-time capability.
  - Failure-shot mean latency remains 6.21--6.28 s across candidates. Alpha improves average
    behavior, not the failure-driven tail; a separate tail strategy or faster native iteration
    implementation is still required.

```
 candidate    errors   success   mean iterations   mean/shot   p99/shot   failure mean
 alpha_none   37/200   81.5%     1238.86           1642 ms     6254 ms    6209 ms
 alpha_0_95   21/200   89.5%      863.21           1397 ms     6362 ms    6250 ms
 alpha_0_90   18/200   91.0%      806.28           1046 ms     6265 ms    6231 ms
 alpha_ramp   26/200   87.0%      924.81           1336 ms     6352 ms    6275 ms
```

### 2026-11-24: Alpha=0.90 accuracy and average-convergence gains replicate across seeds
- Command: `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_alpha_robustness.yaml --out runs/qldpc_alpha_robustness`
- Config: `configs/qldpc_alpha_robustness.yaml`, fixed before running. Gross [[144,12,12]]
  at p=0.001/0.003/0.005, seeds 20261124/20261125, 200 shared shots per cell. Compare
  alpha=None and alpha=0.90 with otherwise identical F32 `paper_like` settings. Ten latency
  shots and 20 profile shots per decoder.
- Commit: code as uncommitted changes on `2381ff5`; the JSON records the corresponding dirty
  provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_alpha_robustness/cells.jsonl` and `precision_sweep.json` (untracked;
  regenerate with the command above).
- Outcome: at p=0.001/0.003 both variants have zero errors and zero prediction disagreements.
  At p=0.005, alpha=0.90 improves errors from 37 to 18 and from 50 to 20 across the two seeds.
  Success rates improve from 81.5%/75.5% to 91.0%/90.0%. McNemar p values are 2.10e-05 and
  2.27e-07, so the accuracy advantage replicates across independent seeds.
- Notes:
  - High-p mean iterations fall from 1238.86 to 806.28 and from 1424.14 to 921.13. Mean latency
    also decreases. This supports a stable average-convergence benefit, not only a single-cell effect.
  - Failure-driven tail remains: all failure shots still use 3680 iterations, and p99 remains
    about 5.75--6.30 s. Low-p tail measurements also vary between seeds, so no general speedup
    percentage is claimed.
  - Alpha=0.90 is a stronger default candidate for subsequent work, but it does not solve
    real-time latency and still needs hardware-data validation.

```
 p      seed       alpha=None errors/success   alpha=0.90 errors/success   McNemar p   mean iterations none -> 0.90
 0.001  20261124   0/200, 100%                0/200, 100%                 1.0         38.61 -> 35.91
 0.001  20261125   0/200, 100%                0/200, 100%                 1.0         35.86 -> 34.58
 0.003  20261124   0/200, 100%                0/200, 100%                 1.0        147.05 -> 117.38
 0.003  20261125   0/200, 100%                0/200, 100%                 1.0        135.65 -> 132.54
 0.005  20261124  37/200, 81.5%              18/200, 91.0%                2.10e-05   1238.86 -> 806.28
 0.005  20261125  50/200, 75.5%              20/200, 90.0%                2.27e-07   1424.14 -> 921.13
```

### 2026-11-24: Iteration caps trade away accuracy for tail latency at every tested budget
- Command: `python -m qec_certify.eval.qldpc_precision_sweep --config configs/qldpc_tail_budget.yaml --out runs/qldpc_tail_budget`
- Config: `configs/qldpc_tail_budget.yaml`, fixed before running. Gross [[144,12,12]] at
  p=0.005, seeds 20261124/20261125, 200 shared shots per cell. All candidates use F32,
  alpha=0.90 and otherwise identical `paper_like` settings. Only the total iteration cap changes:
  680/1280/2480/3680. Ten latency shots and 20 profile shots per decoder.
- Commit: code as uncommitted changes on `2381ff5`; the JSON records the corresponding dirty
  provenance and relay-bp 0.2.2.
- Outputs: `runs/qldpc_tail_budget/cells.jsonl` and `precision_sweep.json` (untracked;
  regenerate with the command above).
- Outcome: lower caps reduce p99 approximately in line with the iteration budget, but every
  capped configuration has significantly more errors than the full 3680 budget on both seeds.
  There is no free tail truncation in the current implementation.
- Notes:
  - Against cap=3680, all cap=680/1280/2480 comparisons have zero `capped_correct_full_wrong`
    transitions and 36/23/7 or 56/23/6 reverse transitions. McNemar p values are
    2.91e-11/2.38e-07/0.015625 and 2.78e-17/2.38e-07/0.03125 respectively.
  - Cap=680 reaches p99 about 1.16--1.19 s, but LER is 0.27/0.38 and success only 73%/62%.
    Cap=2480 still has significantly worse accuracy than the full budget despite p99 about 4 s.
  - Stop this parameter-only tail-truncation route. Further progress needs faster native
    iterations, a better nonconvergence mechanism, or a fundamentally different decoder.

```
 cap    seed       errors   LER    success   mean iterations   mean/shot   p99/shot
 680    20261124   54/200   0.27   73.0%     350.84            601 ms      1192 ms
 1280   20261124   41/200   0.205  79.5%     490.47            922 ms      2214 ms
 2480   20261124   25/200   0.125  87.5%     681.97           1317 ms      4230 ms
 3680   20261124   18/200   0.090  91.0%     806.28           1729 ms      6266 ms
 680    20261125   76/200   0.380  62.0%     410.43            562 ms      1158 ms
 1280   20261125   43/200   0.215  78.5%     590.84            674 ms      2124 ms
 2480   20261125   26/200   0.130  87.0%     782.51            872 ms      3961 ms
 3680   20261125   20/200   0.100  90.0%     921.13           1074 ms      5809 ms
```
