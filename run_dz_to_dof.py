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
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import QTable

from dz_to_dof import (
    DOF_LABELS,
    DZtoDOFSolver,
    N_DOF,
    load_ofc_data,
    load_smatrix_yaml,
    load_weights_yaml,
    make_dz_column_names,
    median_per_group,
    dz_matrix_to_flat,
    group_by_tolerance,
    format_dofs,
    format_residuals,
    plot_all_sensitivity_layers,
    plot_dz_datasets,
    plot_dof_datasets,
    plot_v_modes,
)

log = logging.getLogger("dz_to_dof")

DEFAULT_PUPIL_INDICES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
DEFAULT_FOCAL_INDICES = [1, 2, 3, 4, 5, 6]
# skips M1M3_B52 for now
DEFAULT_DOF_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, # hexapod
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, # m1m3 bending modes
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50 # m2 bending modes
]



def load_dz_data(parquet_path):
    """Load DZ coefficient table from a parquet file, filtering bad fits.

    Returns
    -------
    dz_tab : QTable
    """
    dz_tab = QTable.read(parquet_path)
    n_before = len(dz_tab)

    if 'z1toz6_bad_fit' in dz_tab.colnames:
        dz_tab = dz_tab[
            dz_tab['z1toz6_bad_fit'] == 0.]
        log.info(
            "Loaded %d rows, kept %d"
            " after filtering bad fits",
            n_before, len(dz_tab))
    else:
        log.info(
            "Loaded %d rows"
            " (no z1toz6_bad_fit column)",
            n_before)

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

    log.info(
        "Found %d rotator angle groups: %s",
        len(rot_groups),
        ", ".join(rotang_labels))
    for i, group in enumerate(rot_groups):
        log.debug(
            "  %s: %d observations",
            rotang_labels[i], len(group))

    return rot_groups, rotang_labels


def compact_index_str(indices):
    """Format a sorted list of ints as compact
    range notation, e.g. [4-19,22-26]."""
    if not indices:
        return "[]"
    s = sorted(indices)
    ranges = []
    start = end = s[0]
    for v in s[1:]:
        if v == end + 1:
            end = v
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = end = v
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")
    return "[" + ",".join(ranges) + "]"


def build_version_string(
    dof_indices, norm_type, rcond,
    dof_name=None, suffix=None, rank=None,
):
    """Build version string from solver config.

    Pattern:
      {dof}_{norm}_{mode}[_{suffix}]
    where {mode} is either rcond{e} or rank{k}.

    Examples
    -------
    >>> build_version_string(
    ...     list(range(22)), "geom", 1e-10)
    '22dof_geom_rcond-10'
    >>> build_version_string(
    ...     list(range(22)), "geom", 1e-4, rank=20)
    '22dof_geom_rank20'
    """
    if dof_name is not None:
        dof_part = dof_name
    else:
        dof_part = f"{len(dof_indices)}dof"

    if norm_type is not None:
        norm_part = norm_type
    else:
        norm_part = "no_renorm"

    if rank is not None:
        mode_part = f"rank{rank}"
    else:
        mode_part = f"rcond{int(np.log10(rcond))}"

    parts = [dof_part, norm_part, mode_part]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


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
                        default=DEFAULT_DOF_INDICES,
                        help=("DOF indices to solve."))
    parser.add_argument("--renorm", type=str, default=None,
                        choices=["orig", "geom"],
                        help="Normalization scheme for the sensitivity matrix")
    parser.add_argument("--rot_tolerance", type=float, default=1.0,
                        help="Tolerance for grouping rotator angles (degrees)")
    parser.add_argument("--rcond", type=float, default=1e-4,
                        help="Cutoff for small singular values in lstsq")
    parser.add_argument("--rank", type=int, default=None,
                        help="Keep top-k singular values "
                        "(mutually exclusive with --rcond)")
    parser.add_argument("--smatrix_file", type=str, default=None,
                        help="YAML spec for a custom smatrix "
                        "(default: OFC data, padded at B52)")
    parser.add_argument("--weights_file", type=str, default=None,
                        help="YAML with precomputed norm weights "
                        "(overrides --renorm computation)")
    parser.add_argument("-o", "--output",
                        default="dz_to_dof_results",
                        help="Output directory")
    parser.add_argument("--dof_name", type=str,
                        default=None,
                        help="Short name for DOF set "
                        "(default: '{n}dof')")
    parser.add_argument("--version", type=str,
                        default=None,
                        help="Suffix appended to "
                        "auto-generated version")
    parser.add_argument("--skip-sensitivity",
                        action="store_true",
                        help="Skip sensitivity "
                        "matrix heatmaps")
    parser.add_argument("--skip-dz",
                        action="store_true",
                        help="Skip DZ coefficient "
                        "plots")
    parser.add_argument("--skip-vmodes",
                        action="store_true",
                        help="Skip V-mode heatmap")
    args = parser.parse_args()

    if (args.rank is not None
            and args.rcond != parser.get_default("rcond")):
        raise ValueError(
            "--rank and --rcond are mutually exclusive")

    run_single(args)


def run_single(args, ofc_data=None):
    """Execute one DZ-to-DOF inversion run.

    Parameters
    ----------
    args : argparse.Namespace
        Fully-populated namespace (see `main()`
        for required fields).
    ofc_data : OFCData or None
        Optional pre-loaded OFC data; used by the
        grid runner to share the load across runs.
        Ignored if cache files cover all needs.
    """
    if args.renorm is not None and 30 in args.dof_indices:
        raise ValueError("Cannot do renormalization with M1M3_B52 selected.")

    pupil_indices = args.pupil_indices
    focal_indices = args.focal_indices
    n_focal = len(focal_indices)
    n_pupil = len(pupil_indices)

    version = build_version_string(
        args.dof_indices, args.renorm,
        args.rcond, dof_name=args.dof_name,
        suffix=args.version, rank=args.rank,
    )
    ver = f"_{version}"

    # Output dir: <base>/<parquet_basename>/
    parquet_basename = Path(args.parquet_file).stem
    output_dir = (
        Path(args.output) / parquet_basename
        / version)
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Set up logging ---
    import sys

    log_path = output_dir / f"run_log{ver}.log"
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.setLevel(logging.DEBUG)
    # Disable propagation so messages aren't
    # double-emitted when the caller (e.g. the
    # grid runner) also has root handlers.
    prev_propagate = log.propagate
    log.propagate = False

    log.debug("command: %s", " ".join(sys.argv))
    for key, val in vars(args).items():
        log.debug("  %s: %s", key, val)
    log.info("Log file: %s", log_path)

    try:
        _run_body(
            args, ofc_data, pupil_indices,
            focal_indices, n_focal, n_pupil,
            ver, output_dir)
    finally:
        log.removeHandler(fh)
        log.removeHandler(sh)
        fh.close()
        log.propagate = prev_propagate


def _run_body(
    args, ofc_data, pupil_indices, focal_indices,
    n_focal, n_pupil, ver, output_dir,
):
    """Core work of a single run (post logging
    setup).  Split out so that finally-cleanup of
    logging handlers is straightforward.
    """
    import time

    # --- Load data ---
    smatrix_override = None
    if args.smatrix_file is not None:
        smatrix_override = load_smatrix_yaml(
            args.smatrix_file)

    weights_override = None
    if args.weights_file is not None:
        weights_override = load_weights_yaml(
            args.weights_file)

    # Skip OFC load if both cache files cover it
    need_ofc = (
        smatrix_override is None
        or (args.renorm is not None
            and weights_override is None)
    )
    if need_ofc and ofc_data is None:
        log.info("Loading OFCData")
        ofc_data = load_ofc_data()
    elif not need_ofc:
        log.info(
            "Skipping OFCData load "
            "(using cached smatrix + weights)")
        ofc_data = None
    else:
        log.info("Reusing OFCData from caller")

    log.info("Building solver")
    solver = DZtoDOFSolver(
        ofc_data, pupil_indices,
        focal_indices,
        dof_indices=args.dof_indices,
        norm_type=args.renorm,
        rcond=args.rcond,
        rank=args.rank,
        smatrix_override=smatrix_override,
        weights_override=weights_override,
    )
    log.info(
        "Design matrix A: %s", solver.A.shape)
    log.debug(
        "DOF indices: %s",
        compact_index_str(args.dof_indices))
    log.debug("rcond: %s", args.rcond)
    log.debug(
        "effective rank: %d / %d",
        solver.effective_rank,
        len(args.dof_indices))
    log.debug(
        "condition number: %.2e",
        solver.condition_number)
    _, svals, Vt = solver.svd()
    log.debug("singular values: %s", svals)

    log.info("Loading DZ coefficients")
    dz_tab = load_dz_data(args.parquet_file)
    dates = np.unique(dz_tab['day_obs'])

    # --- Sensitivity matrix heatmaps ---
    if not args.skip_sensitivity:
        log.info("Plotting sensitivity matrix")
        smatrix_dir = (
            output_dir / "sensitivity_matrix")
        smatrix_dir.mkdir(exist_ok=True)
        smat = (solver.renorm_full_coef
                if args.renorm
                else solver.full_coef)
        plot_all_sensitivity_layers(
            smat, pupil_indices, n_focal + 1,
            args.renorm, smatrix_dir, ver,
        )

    # --- V-mode heatmap ---
    rank = solver.effective_rank
    if not args.skip_vmodes:
        log.info("Plotting V-mode heatmap")
        dof_str_short = compact_index_str(
            args.dof_indices)
        plot_v_modes(
            Vt, svals, args.dof_indices, rank,
            (f"V-modes of design matrix A\n"
             f"Norm: {args.renorm}, "
             f"DOF: {dof_str_short}, "
             f"rank: {rank}/"
             f"{len(args.dof_indices)}"),
            output_dir / f"v_modes{ver}.pdf",
        )

    # --- Group by rotator angle ---
    log.info("Grouping by rotator angle")
    rot_groups, rotang_labels = (
        group_by_rotator_angle(
            dz_tab,
            tolerance=args.rot_tolerance))

    # --- Compute median DZ per group ---
    log.info("Computing median DZ per group")
    col_names = make_dz_column_names(
        pupil_indices, focal_indices)
    dz_arr_list = median_per_group(
        dz_tab, col_names, rot_groups,
        n_focal, n_pupil)

    # --- Solve for DOFs per group ---
    log.info("Solving for DOFs")
    dof_hat_list = []
    rec_dz_list = []
    d_dz_list = []

    t0 = time.time()
    for rotang, dz_data in zip(rotang_labels, dz_arr_list):
        result = solver.solve(dz_data)
        dof_hat_list.append(result["x_hat"])
        rec_dz_list.append(result["dz_reconstructed"])
        d_dz_list.append(result["dz_residual"])

        rms = np.sqrt(
            np.mean(result["dz_residual"]**2))
        x = result["x_hat"]
        i_max = np.argmax(np.abs(x))
        log.debug(
            "%s: rank=%d, RMS resid=%.6f, "
            "max |DOF|=%.4f (%s)",
            rotang, result["rank"], rms,
            abs(x[i_max]), DOF_LABELS[i_max])
        log.debug("\n%s", format_dofs(x))
        log.debug(
            "\n%s", format_residuals(
                dz_matrix_to_flat(
                    result["dz_residual"]),
                focal_indices, pupil_indices))

    elapsed = time.time() - t0
    log.debug("Solve loop: %.2fs", elapsed)

    # --- Summary table ---
    log.info("RMS residuals by group:")
    for rotang, d_dz in zip(
        rotang_labels, d_dz_list
    ):
        rms = np.sqrt(np.mean(d_dz**2))
        log.info("  %s: %.6f", rotang, rms)

    # --- Plots ---
    log.info("Generating plots")
    colors = plt.cm.tab10.colors[
        :len(rotang_labels)]

    renorm_str = f"Norm: {args.renorm}"
    zk_str = (f"focal k={compact_index_str(focal_indices)}"
              f", pupil j={compact_index_str(pupil_indices)}")
    if args.dof_indices is not None:
        dof_str = (f"{compact_index_str(args.dof_indices)}")
    else:
        dof_str = f"[1-{N_DOF}]"

    n_dof_sel = len(args.dof_indices)
    if args.rank is not None:
        mode_str = f"rank: {rank}/{n_dof_sel}"
    else:
        mode_str = (f"rcond: {args.rcond}, "
                    f"rank: {rank}/{n_dof_sel}")

    plot_dof_datasets(
        dof_hat_list, rotang_labels, colors,
        (f"Reconstructed DOFs\n"
         f"{renorm_str}, {mode_str}"
         f"\nDates: {dates}\n{zk_str}"),
        output_dir / f"dof_solution{ver}.pdf",
        dof_indices=args.dof_indices,
    )

    if not args.skip_dz:
        plot_dz_datasets(
            dz_arr_list, pupil_indices,
            rotang_labels, colors,
            f"DZ Coefficients\nDates: {dates}",
            output_dir / f"dz_coefficients{ver}.pdf",
        )

    plot_dz_datasets(
        rec_dz_list, pupil_indices,
        rotang_labels, colors,
        (f"Reconstructed DZ Coefficients\n"
         f"{renorm_str}, {mode_str}"
         f"\nDates: {dates}\nDOF: {dof_str}"),
        output_dir / f"dz_reconstructed{ver}.pdf",
    )

    plot_dz_datasets(
        d_dz_list, pupil_indices,
        rotang_labels, colors,
        (f"DZ Coefficient Residuals\n"
         f"{renorm_str}, {mode_str}"
         f"\nDates: {dates}\nDOF: {dof_str}"),
        output_dir / f"dz_residuals{ver}.pdf",
        fixed_y=True,
    )

    log.info("All output saved to %s", output_dir)


if __name__ == "__main__":
    main()
