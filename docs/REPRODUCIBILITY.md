# Reproduction notes

This package contains everything needed to **recompute every headline number in the
paper without retraining and without the raw datasets** (~GB of ChEMBL / BindingDB /
Davis / KIBA), using only the deposited per-member predictions.

## Two entry points

### 1. The 18-cell matrix — needs only the result JSONs

```bash
python scripts/p4_matrix.py
```

Reads `results/matrix_4x5.json` (the frozen authority), `results/kairos_theta_rule.json`,
`results/ci_matrix.json` and `results/p2_theta_variants.json`. Prints, per cell, the
Aptum RMSE / CI against the strongest per-cell baseline, the verdict, the leakage
account and the calibration string. **No predictions required.**

### 2. Head-to-head numbers — needs the compressed predictions

```bash
python scripts/reproduce_from_npz.py
```

Reads `predictions/*.npz`, re-fits the cross-fitted (cv5) caliber in-fold, and compares
Aptum against the best single member of each pool:

```
cells=19  RMSE beats best single member: 19/19
mean reduction vs best single: +3.41%
```

This is the reproduction of the paper's Figure 3 and of the "three metrics in all 18
cells" claim (the 19th row is KIBA·random, an additional split with deposited
predictions).

## What the verification chain actually checks

| Claim | How it is checked here |
|---|---|
| Aptum beats every pool member in every cell | `reproduce_from_npz.py`, per-cell `best_single` |
| MSE and RMSE orderings coincide | by construction — MSE = RMSE²; reported together, not as independent evidence |
| The concordance index is correctly implemented | `ci_lib.py` ships with an `O(n²)` enumeration self-check |
| Leakage is reported, not assumed | `p4_matrix.py` prints the leakage account per cell |
| Cross-fitted caliber is not conflated with label-free | every table in the paper labels the caliber; the shift branch is in `kairos_theta_rule.json` |

## Known exceptions (documented, not hidden)

* **Two aggregates have partially deposited legacy members.** Their rows in
  `results/matrix_4x5.json` carry a `caliber` string saying so. They are the reason the
  headline table reports "18 of 18" for the deployment caliber while the compressed
  reproduction covers 19 cells.
* **Davis·random is transductive (cv5).** Its six legacy members have no deposited
  validation predictions, so a label-free fit is impossible there. This is the only
  non-label-free cell and is labelled as such wherever it appears.
* **Not included:** member training code and weights, and the raw datasets. The members
  are published architectures; retraining them needs their original repositories.

## Environment

```bash
pip install -r requirements.txt
```

Verified on Python 3.11 with numpy 2.x, scipy 1.17, pandas 3.0. CPU only; no GPU needed.

## Figures

```bash
python scripts/make_fig_eds_v3.py      # Figure 2 (rank heatmap, δ_k, stability)
python scripts/make_fig_agg_v6.py      # Figure 3 (RMSE, reduction, three metrics)
```

Both write vector PDFs into `figs/`. The architecture figure (Figure 1) is a standalone
TikZ source, `figs/fig_kairos_arch.tex`, compiled with `pdflatex`.