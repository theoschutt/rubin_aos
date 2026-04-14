#!/usr/bin/env python
"""
Solve the DZ-to-DOF inversion problem and produce all plots.

Loads a sensitivity matrix from lsst.ts.ofc, loads double-Zernike (DZ)
coefficients from a parquet file, groups observations by camera rotator
angle, solves for DOFs per group, and generates:

  1. Sensitivity matrix heatmaps
  2. Input DZ coefficients (per rotator-angle group)
  3. Recovered DOFs
  4. Reconstructed DZ coefficients
  5. DZ residuals

Usage
-----
    python run_dz_to_dof.py /path/to/dz_coefficients.parquet \\
        [--pupil_indices 4 5 6 ... 19 22 23 24 25 26] \\
        [--focal_indices 1 2 3 4 5 6] \\
        [--dof_indices 0 1 2 ...] \\
        [--rot_tolerance 1.0] \\
        [-o output_dir] [--version v1]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import QTable

from dz_to_dof import (
    DZtoDOFSolver,
    compact_index_str,
    load_ofc_data,
    make_dz_column_names,
    median_per_group,
    dz_matrix_to_flat,
    group_by_tolerance,
    print_dofs,
    print_residuals,
    plot_all_sensitivity_layers,
    plot_dz_datasets,
    plot_dof_datasets,
)

DEFAULT_PUPIL_INDICES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
DEFAULT_FOCAL_INDICES = [1, 2, 3, 4, 5, 6]




def load_dz_data(parquet_path):
    """Load DZ coefficient table from a parquet file, filtering bad fits.

    Returns
    -------
    dz_tab : QTable
    """
    dz_tab = QTable.read(parquet_path)
    n_before = len(dz_tab)

    if 'z1toz6_bad_fit' in dz_tab.colnames:
        dz_tab = dz_tab[dz_tab['z1toz6_bad_fit'] == 0.]
        print((f"Loaded {n_before} rows,"
               f" kept {len(dz_tab)} after filtering bad fits"))
    else:
        print(f"Loaded {n_before} rows (no z1toz6_bad_fit column)")

    return dz_tab


def group_by_rotator_angle(dz_tab, tolerance=1.0):
    """Group observations by rotator angle, sorted by angle.

    Returns
    -------
    rot_groups : list of list of int
        Index groups sorted by mean rotator angle.
    rotang_labels : list of str
        Rounded mean angle label per group.
    """
    rot_groups_unordered = group_by_tolerance(dz_tab['rotator_angle'], tolerance)

    # Sort groups by mean rotator angle
    mean_angles = [np.mean(np.asarray(dz_tab['rotator_angle'])[g])
                   for g in rot_groups_unordered]
    order = np.argsort(mean_angles)
    rot_groups = [rot_groups_unordered[i] for i in order]

    rotang_labels = []
    for group in rot_groups:
        angle = np.round(np.mean(np.asarray(dz_tab['rotator_angle'])[group]))
        rotang_labels.append(f"rot={angle:.0f}")

    print(f"Found {len(rot_groups)} rotator angle groups: "
          + ", ".join(rotang_labels))
    for i, group in enumerate(rot_groups):
        print(f"  {rotang_labels[i]}: {len(group)} observations")

    return rot_groups, rotang_labels


def main():
    parser = argparse.ArgumentParser(
        description="Solve DZ-to-DOF inversion and produce all plots."
    )
    parser.add_argument("parquet_file",
                        help="Path to DZ coefficient parquet file")
    parser.add_argument("--pupil_indices", nargs="+", type=int,
                        default=DEFAULT_PUPIL_INDICES,
                        help="Pupil Zernike indices to use")
    parser.add_argument("--focal_indices", nargs="+", type=int,
                        default=DEFAULT_FOCAL_INDICES,
                        help="Focal Zernike indices to use")
    parser.add_argument("--dof_indices", nargs="+", type=int,
                        default=None,
                        help="DOF indices (0-49) to solve. [default: [0-49]]")
    parser.add_argument("--renorm", type=str, default=None,
                        choices=["orig", "geom"],
                        help="Normalization scheme for the sensitivity matrix")
    parser.add_argument("--rot_tolerance", type=float, default=1.0,
                        help="Tolerance for grouping rotator angles (degrees)")
    parser.add_argument("-o", "--output", default="dz_to_dof_results",
                        help="Output directory")
    parser.add_argument("--version", type=str, default="",
                        help="Version string appended to output file names")
    args = parser.parse_args()

    pupil_indices = args.pupil_indices
    focal_indices = args.focal_indices
    n_focal = len(focal_indices)
    n_pupil = len(pupil_indices)
    ver = f"_{args.version}" if args.version else ""

    # Output dir: <base>/<parquet_basename>/
    parquet_basename = Path(args.parquet_file).stem
    output_dir = Path(args.output) / parquet_basename / args.version
    output_dir.mkdir(exist_ok=True, parents=True)

    # Write log of CLI options
    import sys
    log_file = output_dir / f"run_log{ver}.txt"
    with open(log_file, "w") as f:
        f.write(f"command: {' '.join(sys.argv)}\n\n")
        for key, val in vars(args).items():
            f.write(f"{key}: {val}\n")
    print(f"Run log saved to {log_file}")

    # --- Load data ---
    print("\n=== Loading OFCData ===")
    ofc_data = load_ofc_data()

    print("\n=== Building solver ===")
    solver = DZtoDOFSolver(
        ofc_data, pupil_indices, focal_indices,
        dof_indices=args.dof_indices, norm_type=args.renorm,
    )
    print(f"Design matrix A shape: {solver.A.shape}")

    print("\n=== Loading DZ coefficients ===")
    dz_tab = load_dz_data(args.parquet_file)
    dates = np.unique(dz_tab['day_obs'])

    # --- Sensitivity matrix heatmaps ---
    print("\n=== Plotting sensitivity matrix ===")
    smatrix_dir = output_dir / "sensitivity_matrix"
    smatrix_dir.mkdir(exist_ok=True)
    smat = (solver.renorm_full_coef if args.renorm
            else solver.full_coef)
    plot_all_sensitivity_layers(
        smat, pupil_indices, n_focal + 1,
        args.renorm, smatrix_dir, ver,
    )

    # --- Group by rotator angle ---
    print("\n=== Grouping by rotator angle ===")
    rot_groups, rotang_labels = group_by_rotator_angle(
        dz_tab, tolerance=args.rot_tolerance
    )

    # --- Compute median DZ per group ---
    print("\n=== Computing median DZ per group ===")
    col_names = make_dz_column_names(
        pupil_indices, focal_indices)
    dz_arr_list = median_per_group(
        dz_tab, col_names, rot_groups, n_focal, n_pupil)

    # --- Solve for DOFs per group ---
    print("\n=== Solving for DOFs ===")
    dof_hat_list = []
    rec_dz_list = []
    d_dz_list = []

    for rotang, dz_data in zip(rotang_labels, dz_arr_list):
        result = solver.solve(dz_data)
        dof_hat_list.append(result["x_hat"])
        rec_dz_list.append(result["dz_reconstructed"])
        d_dz_list.append(result["dz_residual"])

        rms = np.sqrt(np.mean(result["dz_residual"]**2))
        print(f"\n  {rotang}: rank={result['rank']}, "
              f"RMS residual={rms:.6f}")
        print_dofs(result["x_hat"])
        print()
        print_residuals(
            dz_matrix_to_flat(result["dz_residual"]),
            focal_indices, pupil_indices,
        )

    # --- Plots ---
    print("\n=== Generating plots ===")
    colors = plt.cm.tab10.colors[:len(rotang_labels)]

    renorm_str = f", Norm: {args.renorm}"
    zk_str = (f"focal k={compact_index_str(focal_indices)}"
              f", pupil j={compact_index_str(pupil_indices)}")
    if args.dof_indices is not None:
        dof_str = (f"{compact_index_str(args.dof_indices)}")
    else:
        dof_str = "[1-50]"

    plot_dof_datasets(
        dof_hat_list, rotang_labels, colors,
        (f"Reconstructed DOFs from median DZ "
         f"coeffs{renorm_str}\n Dates: {dates}\n{zk_str}"),
        output_dir / f"dof_solution{ver}.pdf",
        dof_indices=args.dof_indices,
    )

    plot_dz_datasets(
        dz_arr_list, pupil_indices,
        rotang_labels, colors,
        f"DZ Coefficients\nDates: {dates}",
        output_dir / f"dz_coefficients{ver}.pdf",
    )

    plot_dz_datasets(
        rec_dz_list, pupil_indices,
        rotang_labels, colors,
        (f"Reconstructed DZ Coefficients"
         f"{renorm_str}\n Dates: {dates}\nDOF: {dof_str}"),
        output_dir / f"dz_reconstructed{ver}.pdf",
    )

    plot_dz_datasets(
        d_dz_list, pupil_indices,
        rotang_labels, colors,
        (f"DZ Coefficient Residuals"
         f"{renorm_str}\n Dates: {dates}\nDOF: {dof_str}"),
        output_dir / f"dz_residuals{ver}.pdf",
    )

    # better viz for relative contributions
    plot_dz_datasets(
        d_dz_list, pupil_indices,
        rotang_labels, colors,
        (f"DZ Coefficient Residuals"
         f"{renorm_str}\n Dates: {dates}\nDOF: {dof_str}"),
        output_dir
        / f"dz_residuals_fixed_ylims{ver}.pdf",
        fixed_y=True,
    )

    print(f"\nAll output saved to {output_dir}")


if __name__ == "__main__":
    main()
