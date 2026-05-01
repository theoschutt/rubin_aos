# DZ-to-DOF Analysis

Inverts the OFC double-Zernike (DZ) sensitivity matrix to recover
LSST AOS degrees of freedom (DOFs) from measured wavefront DZ
coefficients. Given DZ coefficients fit to corrected wavefronts from
FAM visit pairs, this code solves the linear system
`A @ x_hat ≈ dz` (least-squares, with optional rcond / rank-truncation
of the SVD) and produces diagnostic plots and `.npy` arrays.

---

## Layout

```
dz_to_dof/
├── README.md                         # this file
├── dz_to_dof.py                      # core module: solver, smatrix
│                                     #   loaders, normalization,
│                                     #   plotting helpers
├── run_dz_to_dof.py                  # CLI: single solver run
├── run_grid.py                       # CLI: sweep DOF sets / rcond
│                                     #   / rank / norm schemes
├── combine_grid_plots.py             # CLI: collate grid PDFs into
│                                     #   one PDF
├── build_ofc_cache.py                # CLI: dump OFC smatrix +
│                                     #   weights to YAML/.npy
│                                     #   (optional, with stack)
├── test_dz_to_dof.py                 # pytest suite
├── test_gq_weights.py                # reproduce geom_gq weights
├── grid_configs/                     # JSON grid configs
├── ofc_cache/                        # cached smatrix + weights
├── input_data/                       # input DZ parquets and
│                                     #   reference weights
└── dz_to_dof_results/                # outputs (gitignored)
```

---

## Data flow

```
fit_dofs_to_double_zernikes.ipynb
   (visits → DZ fit → parquet in input_data/)
                │
                ▼
run_dz_to_dof.py        ──or──   run_grid.py
   (solve + plot single)          (sweep configs → many runs)
                │                       │
                ▼                       ▼
   dz_to_dof_results/<output>/<dataset>/<version>/
                                        │
                                        ▼
                            combine_grid_plots.py
                            (multi-page combined PDF)
```

---

## Setup

Standard scientific Python deps: `numpy`, `scipy`, `matplotlib`,
`astropy`, `pyyaml`, `pyarrow`, `pypdf` (for `combine_grid_plots.py`).

The LSST stack is only needed for `build_ofc_cache.py`, `test_gq_weights.py`
and one optional test in `test_dz_to_dof.py`. `build_ofc_cache.py` is an
optional one-time step to refresh `ofc_cache/` if the stack OFC data change.

`ofc_cache/` is committed to the repo, so the LSST stack is **not**
required for normal use — the solver reads the cached
sensitivity matrix and weights directly.

---

## Typical workflows

### Single solver run

`--smatrix_file` and `--weights_file` must be passed explicitly to
use the committed cache; otherwise `run_dz_to_dof.py` falls back to
loading from the LSST stack.

Minimal stack-free invocation:

```bash
python run_dz_to_dof.py input_data/my_dz.parquet \
    --smatrix_file ofc_cache/smatrix_cache.yaml \
    -o my_run
```

Custom DOF set, geom_gq weights, rank-25 truncation:

```bash
python run_dz_to_dof.py input_data/my_dz.parquet \
    --smatrix_file ofc_cache/smatrix_cache.yaml \
    --weights_file ofc_cache/weights_geom_gq.yaml \
    --dof_indices 0 1 2 3 4 5 6 7 8 9 \
                  10 11 12 13 14 15 16 17 18 19 \
                  20 21 22 23 24 25 26 27 28 29 \
                  31 32 33 34 35 36 37 38 39 40 \
                  41 42 43 44 45 46 47 48 49 50 \
    --rank 25 \
    -o geom_gq_50dof_rank25
```

### Filtering and grouping by alt/rot

Group by altitude only (default), keeping only `rotator_angle=0`
rows:

```bash
python run_dz_to_dof.py input_data/my_dz.parquet \
    --group_col_name alt \
    --filter_col_name rotator_angle --filter_val 0
```

Group by `(alt, rotator_angle)` combos, filter to two altitudes and
two rotator angles (OR within column, AND across):

```bash
python run_dz_to_dof.py input_data/my_dz.parquet \
    --group_col_name alt rotator_angle \
    --filter_col_name alt rotator_angle \
    --filter_val 70 75 \
    --filter_val -90 0
```

### Grid sweep + PDF combine

```bash
python run_grid.py input_data/my_dz.parquet \
    --config grid_configs/grid_config_50dof.json \
    -o grid_50dof
python combine_grid_plots.py \
    dz_to_dof_results/grid_50dof/my_dz \
    -o dz_to_dof_results/grid_50dof/grid_50dof_combined.pdf
```

A grid config JSON may set:

- `dof_sets`: `{name: [indices, ...]}`
- `rcond_values`: list of rcond floats
- `rank_values`: list of rank ints (mutually exclusive with rcond)
- `norm_schemes`: list of `null` / `"orig"` / `"geom"` / `"geom_gq"`
  - Note that if a weights YAML file is specified, this field only gets used
  as a label for plots.
- `run_args`: dict of default CLI args applied to every run (CLI
  flags override these)

See `grid_configs/grid_config_*.json` for examples.

---

## Output structure

Each run writes to:

```
dz_to_dof_results/<output>/<dataset>/<version>/
```

where `<output>` comes from `-o`, `<dataset>` from the parquet
basename (or `--dataset_name`), and `<version>` is auto-built from
the chosen DOF set, normalization, and rcond/rank, with optional
`--version` suffix.

Contents:

| File                              | Description                  |
|-----------------------------------|------------------------------|
| `dof_solution{ver}.pdf`           | Recovered DOFs per group     |
| `dz_coefficients{ver}.pdf`        | Input DZ per group           |
| `dz_reconstructed{ver}.pdf`       | A @ x_hat per group          |
| `dz_residuals{ver}.pdf`           | dz - A @ x_hat per group     |
| `sensitivity_matrix{ver}.pdf`     | Smatrix layer heatmaps       |
| `v_modes{ver}.pdf`                | SVD V-mode heatmap           |
| `dof_solution{ver}.npy`           | (n_groups, n_dof) array      |
| `dz_coefficients{ver}.npy`        | (n_groups, n_focal, n_pupil) |
| `dz_reconstructed{ver}.npy`       | same shape as dz_coeffs      |
| `dz_residuals{ver}.npy`           | same shape as dz_coeffs      |
| `group_labels{ver}.txt`           | one label per group          |

`--skip-sensitivity`, `--skip-dz`, `--skip-vmodes` skip both the PDF
and the corresponding `.npy` (where applicable).

---

## Key CLI flags (`run_dz_to_dof.py`)

| Flag                          | Purpose                            |
|-------------------------------|------------------------------------|
| `parquet_file` (positional)   | Input DZ parquet(s)                |
| `--dataset_name`              | Output subdir (req'd if >1 input)  |
| `--pupil_indices`             | Pupil Zernike indices (defaults    |
|                               | to Noll 4-19 + 22-26)              |
| `--focal_indices`             | Focal Zernike indices (default 1-6)|
| `--dof_indices`               | DOFs to solve (default skips B52)  |
| `--renorm {orig,geom}`        | Smatrix normalization              |
| `--smatrix_file`              | YAML smatrix override (cache)      |
| `--weights_file`              | YAML weights override (cache)      |
| `--group_col_name [...]`      | Group rows by these columns        |
| `--group_tolerance`           | Tolerance for grouping (deg)       |
| `--filter_col_name [...]`     | Pre-filter by these columns        |
| `--filter_val [...]`          | Target value(s); repeat per col    |
| `--filter_tolerance`          | Filter match tolerance (deg)       |
| `--rcond`                     | SVD cutoff (default 1e-4)          |
| `--rank`                      | Top-k SVs (excludes `--rcond`)     |
| `-o, --output`                | Output dir (default                |
|                               | `dz_to_dof_results`)               |
| `--version`                   | Suffix on auto version string      |
| `--dof_name`                  | Short DOF set name in version      |
| `--skip-sensitivity`          | Skip smatrix heatmap PDF + array   |
| `--skip-dz`                   | Skip DZ plots + arrays             |
| `--skip-vmodes`               | Skip V-mode heatmap                |

`run_grid.py` accepts every flag above; the grid config supplies
defaults that the CLI can override.

---

## Conventions and gotchas

- **DOF index layout** (length 51, OFC ordering):
  - `0-9`: hexapod (M2 hex z/x/y/rx/ry, then Cam hex z/x/y/rx/ry)
  - `10-29`: M1M3 bending modes B1-B20
  - `30`: M1M3_B52 (slot reserved for future compatibility, padded)
  - `31-50`: M2 bending modes B1-B20
  - The default `--dof_indices` skips index 30.
  - **Both `--renorm orig` and `--renorm geom` are invalid when B52
    is selected.** `run_dz_to_dof.py` raises a `ValueError` if
    `--renorm` is set with `30` in `--dof_indices`.

- **Sensitivity matrix shape**: `sens[k, j, dof]` with `k` = focal
  Zernike, `j` = pupil Zernike. Parquet column names are
  `z1toz6_z{j}_c{k}` — pupil index outer, focal index inner.

- **Angle units**: in the parquet, `alt` is in **radians**; on the
  CLI (`--filter_val`, group output labels) it is in **degrees**.
  `rotator_angle` is in degrees throughout.

- **Normalization schemes**:
  - `orig` — OFC `normalization_weights` (Megias+24 Eqs 9-11).
  - `geom` — `sqrt(r_i / f_i)` from range and FWHM weights.
  - `geom_gq` — `geom` variant with FWHM weights computed via
    Gauss quadrature (provided as a precomputed weights YAML in
    `ofc_cache/weights_geom_gq.yaml`).
  - To use `geom_gq`, pass it via `--weights_file` (it is not a
    `--renorm` choice; it is delivered as a weights override).

- **`rcond` vs `rank`**: mutually exclusive. `rank=k` keeps the top-k
  singular values; `rcond=r` zeroes singular values below `r * s[0]`.
