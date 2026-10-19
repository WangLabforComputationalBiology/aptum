# extra/ — scripts that need data outside this package

These scripts were used to produce parts of the paper but read inputs that are **not**
shipped here:

* `D:/protein_datasets/leap/frozen/battlefield_manifest.parquet` — the record-level
  provenance table (endpoint type, target id, publication year) used to build the
  splits and to compute the endpoint-conditioned quantities.
* `D:/protein_datasets/bindingdb/bdb_pairs.parquet` — the Davis↔BindingDB matched pair
  file used by the cross-source noise analysis.
* the original `results/p2/` parquet layout (this package ships the compressed
  `predictions/*.npz` instead).

They are included for transparency — so every number in the paper has a visible
provenance — but they will not run as-is outside the authors' environment. The two
supported entry points are `scripts/p4_matrix.py` and `scripts/reproduce_from_npz.py`,
which need nothing beyond `requirements.txt`.
