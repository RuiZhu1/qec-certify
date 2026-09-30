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
