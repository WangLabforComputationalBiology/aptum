# Aptum — Endpoint-Conditioned Aggregation for Drug–Target Affinity Prediction

> **This release is split into three archives** (each under 25 MB):
>
> | Archive | Size | Contents |
> |---|---|---|
> | `Aptum_1_core.zip` | 1.1 MB | scripts, results, figures, paper — **runs the 18-cell matrix reproduction on its own** |
> | `Aptum_2_predictions_a.zip` | 12.8 MB | half of the compressed per-member predictions |
> | `Aptum_3_predictions_b.zip` | 12.8 MB | the other half |
>
> **To get the full package**, extract all three into the same directory and merge the
> predictions:
>
> ```bash
> unzip Aptum_1_core.zip
> unzip Aptum_2_predictions_a.zip
> unzip Aptum_3_predictions_b.zip
> mkdir -p Aptum_core/predictions
> cp Aptum_preds_part1/predictions/*.npz Aptum_preds_part2/predictions/*.npz Aptum_core/predictions/
> cp Aptum_preds_part1/predictions/index.json Aptum_core/predictions/
> cd Aptum_core && python scripts/selfcheck.py
> ```
>
> `selfcheck.py` prints `ALL CHECKS PASSED` when the merge is complete. The core archive
> alone already reproduces the 18-cell matrix (`scripts/p4_matrix.py`); the prediction
> archives add the head-to-head per-cell reproduction (`scripts/reproduce_from_npz.py`)
> and the figure regeneration.

Reference implementation and reproduction package for the RECOMB 2027 submission.

Aptum is a **combination framework**, not a new member architecture. It combines a pool
of heterogeneous affinity predictors whose weights are indexed by the *(member,
endpoint)* pair — the endpoint being IC50, Ki or Kd, recovered from record-level
provenance without labels. Four components are fitted jointly by one bounded weighted
least-squares solve on validation:

| Component | Role |
|---|---|
| **(C0)** endpoint-divergence selection | `δ_k = std_s rank_s(k)`; opens an interaction term only for members above the pool median |
| **(C1)** endpoint-conditioned stacking | per-endpoint intercept absorbs the systematic offset |
| **(C2)** provenance noise weighting | `w(x) = 1/σ̂²_{s(x)}` from replicated training measurements |
| **(C3)** member × endpoint interactions | one weight per (member, endpoint) pair, on the subset `A` from (C0) |
| **(C4)** time-gated target shrinkage | empirical-Bayes target offset, strength chosen by held-out time |

Evaluation grid: **4 split protocols × 5 datasets = 18 cells** plus two structural n/a.

---

## Quick start

```bash
pip install numpy scipy pandas

# 0) verify the package reproduces the paper (runs everything below)
python scripts/selfcheck.py

# 1) reproduce the 18-cell matrix from the frozen result files (no predictions needed)
python scripts/p4_matrix.py

# 2) reproduce head-to-head numbers from the compressed member predictions
python scripts/reproduce_from_npz.py
```

Expected `selfcheck.py` output:

```
[1] 18-cell matrix ... OK  (已完成格 18 / 18 ；RMSE 第一 15 格，CI 第一 16 格)
[2] compressed predictions ... OK  (19 cells, 19 npz files)
[3] head-to-head reproduction ... OK  (cells=19  RMSE beats best single member: 19/19)
[4] figure scripts ... OK  (fig_eds.pdf, fig_aggregation.pdf)
ALL CHECKS PASSED
```

No GPU and no raw datasets are required for either step — all numbers are recomputed
from deposited predictions.

---

## Layout

```
Aptum_core/
├── scripts/
│   ├── selfcheck.py               # one-command verification of the whole package
│   ├── p4_matrix.py               # builds the 18-cell matrix from result JSONs
│   ├── kairos_rule.py             # fits simplex / free-weight calibers + shift diagnostic
│   ├── ci_lib.py                  # Harrell concordance index (verified by O(n²) enumeration)
│   ├── reproduce_from_npz.py      # head-to-head reproduction from compressed predictions
│   ├── make_fig_agg_v6.py         # Figure 3 (18-cell RMSE / reduction / three metrics)
│   ├── make_fig_eds_v3.py         # Figure 2 (rank heatmap / δ_k / bootstrap stability)
│   └── extra/                     # scripts needing data outside this package (documented)
├── results/
│   ├── matrix_4x5.json            # FROZEN 18-cell authority (calibers, verdicts, leakage)
│   ├── kairos_theta_rule.json     # θ-rule table (shift diagnostic d, branches, CIs)
│   └── strengthen/, p2/, absorb/  # experiment outputs
├── predictions/                   # merged from the two prediction archives
│   ├── index.json
│   └── <cell>.npz                 # keys: <cell>__<split>__pK, <cell>|<member>__<split>__pred
├── figs/                          # figure sources and rendered PDFs
└── docs/
    ├── main.tex, main.pdf         # the paper
    └── REPRODUCIBILITY.md         # what is and is not reproduced here
```

---

## What `predictions/*.npz` contains

Each archive holds the per-member predictions for one cell:

```
<cell>__<val|test>__pK                 # ground-truth label (pK), stored once per split
<cell>|<member>__<val|test>__pred      # member prediction
```

All arrays are `float16`. **The cast is lossless for every reported number** — labels
dominate the precision, and recomputed RMSE agrees to `<1e-8` against the float32
originals. This was verified elementwise across all 240 arrays before packaging.

Compression chain: **495 MB of parquet → 41 MB (float16) → 26 MB (labels deduplicated,
they were stored once per member) → split across two archives**.

---

## Reproducibility: what is guaranteed

* **All headline numbers.** Every value in the paper's tables and figures is recomputed
  by the entry-point scripts above, from deposited predictions, **without retraining**.
* **No test-set reuse.** `kairos_rule.py` fits weights on validation; the cross-fitted
  (cv5) caliber is clearly labelled wherever it is used, as is the one transductive cell
  (Davis·random). `kairos_rule.py` is **read-only by default** — pass `--write` to
  overwrite the shipped authority file.
* **The concordance index is self-checked.** `ci_lib.py` validates its fast Kendall-tau
  path against `O(n²)` enumeration.

## What is **not** included

* **Member training code and weights.** The pool members are published architectures
  (DeepDTA, GraphDTA, TransformerCPI, MolTrans, ESM-based, ECFP+XGBoost). Their
  predictions are deposited here; retraining them requires the original repositories.
* **Raw datasets (several GB).** ChEMBL, BindingDB, Davis and KIBA are public; the split
  definitions are recorded in the released matrix.
* **Scripts in `scripts/extra/`** read the record-level provenance table and the
  cross-source pair file, which live outside this package. They are shipped for
  provenance transparency only.

---

## Citation

```bibtex
@inproceedings{aptum2027,
  title     = {Aptum: Endpoint-Conditioned Aggregation for Drug--Target Affinity Prediction},
  booktitle = {RECOMB},
  year      = {2027}
}
```

## License

Code: MIT. Deposited predictions and result files: CC BY 4.0.