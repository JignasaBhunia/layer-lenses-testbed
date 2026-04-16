# Reproducibility Log

One subsection per config. Each subsection records the source of every hyperparameter:
- **paper**: pinned explicitly in a paper .tex or .pdf in this repo.
- **notebook**: copied from a specific legacy notebook cell.
- **inferred**: chosen by us when neither paper nor notebook pins it; treat as a deliberate default, not retrospective tuning.

All entries should cite the exact source file and, for notebooks, the cell index.

## `configs/milestone1_dlgn_sf_odt.yaml` (planned)

Goal: reproduce the shared "DLGN gating hyperplanes cluster around COB-ODT decision hyperplanes" phenomenon.

Data (Paper 1, Sec. "Synthetic Datasets" + Table 2 caption):
- `d = 100`           (paper)
- `D = 4`             (paper; tree depth)
- `n_total = 30000`   (paper, Table 2 caption)
- `seed = 365`        (paper, Table `syn-data-gen`)
- `bias = 0`          (paper, "biases kept at zero")
- `threshold = 0`     (paper, Table `syn-data-gen`)
- x sphere-uniform    (paper)
- u_i QR-orthonormal  (paper)
- y in {-1, +1} with sibling leaves opposite  (paper)

Split:
- train/val/test = 50/25/25  (paper, "Train_Validation_Test Split" appendix)

Model (DLGN_SF, i.e. Paper 1's shallow-features variant = Paper 2's DLGN):
- `L = 4`             (paper, Table 2 caption: "4 hidden layers")
- `M = 20`            (paper, Table 2 caption: "20 neurons in each layer")
- `beta = 30`         (notebook, `multi_path_dlgn.ipynb` cell 9; paper does not pin beta explicitly)
- gating bias = False (paper)
- value network: deep linear (paper, Eq. `DLGN_computation`)

Training (to be pulled from `notebooks/old_notebooks/discontinuities_clustering/code/1.DLGN_DLGN_SF_Synthetic.ipynb` during implementation):
- optimizer, learning rate, batch size, epochs: TBD (will cite notebook cell)
- loss: log-loss (BCEWithLogits on outputs remapped to {-1,+1}) (paper, Sec. 3 — "log loss for binary classification")

(Populate this block once the notebook is inspected in the implementation step.)

Snapshotting:
- save effective gating weights `V^l` at epoch 0 (init) and at the end of training, plus a handful of intermediate epochs for analysis.

Analysis outputs:
- `figures/scatter_init.pdf`, `figures/scatter_final.pdf` (Paper 2 Fig 2 style).
- `distance_table.csv` (Paper 1 Table 2 style): for each ODT internal node and each distance threshold in {0.1, 0.2, 0.3}, the count of DLGN effective hyperplanes within that distance. Also "closest distance at init" and "closest distance at final" rows.
