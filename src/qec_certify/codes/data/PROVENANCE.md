# Reference circuit provenance

The `.stim` files and `LICENSE.txt` in this directory are unmodified data from the public
Relay-BP repository, fixed at commit `72341ef85cd45779e6483c338aef7ba85094eac1`.
Relay-BP is not a supported IBM product. The upstream copyright notice and Apache-2.0 license
are retained in `LICENSE.txt`.

Source directory: https://github.com/trmue/relay/tree/72341ef85cd45779e6483c338aef7ba85094eac1/tests/testdata/bicycle_bivariate

| Local file | Upstream circuit | SHA-256 |
|---|---|---|
| `bb18_d3_p003.stim` | `bicycle_bivariate_18_4_3_memory_Z`, distance=3, rounds=3, p=0.003 | `3821d7f09bd931d774e777945fee0811546779830bc110d9b412465b74a63800` |
| `bb144_d12_p0001.stim` | `bicycle_bivariate_144_12_12_memory_Z`, distance=12, rounds=12, p=0.001 | `97ed68e3859e10ebdfd0a13c5c6abd623084ee1bd11d88287af6a67abfdd24f2` |
| `bb144_d12_p0002.stim` | `bicycle_bivariate_144_12_12_memory_Z`, distance=12, rounds=12, p=0.002 | `d935f128b5ea8d225f7ba523f7dab4561580e2cc6ab95942c6d15245892f5cde` |
| `bb144_d12_p003.stim` | `bicycle_bivariate_144_12_12_memory_Z`, distance=12, rounds=12, p=0.003 | `839d2fa8ce95c62e32e2d465fb646a835e184be6ceca6e4b32f046f5c506cd20` |
| `bb144_d12_p0004.stim` | `bicycle_bivariate_144_12_12_memory_Z`, distance=12, rounds=12, p=0.004 | `a4ad4c07875ff213ce2cd6dc978f9460f10783225e449f17c9039df0f1972e53` |
| `bb144_d12_p0005.stim` | `bicycle_bivariate_144_12_12_memory_Z`, distance=12, rounds=12, p=0.005 | `a75a7c020ccc28ea83549b1d742410d08d22e603b126700cea098dd5965a5a21` |

`qec_certify.codes.load_reference_circuit` verifies these hashes before parsing a circuit.
