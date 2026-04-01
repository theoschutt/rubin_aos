"""
Analyze the impact on inferred double Zernike (DZ) coefficients depending on
which sensitivity matrix is used.

For each DOF, computes S_data / S_sim - 1 element-wise, where:
  - S_data : data-derived sensitivity matrix (FAM or GD measured)
  - S_sim  : simulated (design) sensitivity matrix

With dz_0 = 1, this gives the fractional disagreement between the two
sensitivity matrices at each (focal Zernike k, pupil Zernike j) position.

Loads per-DOF pkl files saved by fit_plot_dbl_zks_sens.py; the pkl path is
inferred from --state_key, --kmax, --jmax, --output, --version.

Usage (same CLI structure as fit_plot_dbl_zks_sens.py):
    python analyze_sensmat_impact.py sensitivity_analysis_*.asdf \\
        [--include_giant_donuts] [--kmax 6] [--jmax 28] [--version v3]
    python analyze_sensmat_impact.py --state_key M2_z --include_giant_donuts \\
        [--kmax 6] [--jmax 28] [--version v3]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fit_plot_dbl_zks_sens import (
    _pad_state_key, _wrap_text, find_cached_pkl,
    extract_dof_info, setup_kj_figure, finalize_kj_figure,
)


def compute_dz_residual(S_data, S_sim, S_data_err=None, threshold=1e-10):
    """Compute element-wise fractional residual S_data / S_sim - 1.

    Entries where |S_sim| <= threshold are set to NaN.

    Returns
    -------
    residuals : ndarray
    residual_errs : ndarray or None
        S_data_err / |S_sim| where defined, NaN elsewhere. None if S_data_err not provided.
    """
    mask = np.abs(S_sim) > threshold
    ratio_mask = np.abs(S_data / S_sim - 1) < 50
    residual_errs = None
    if S_data_err is not None:
        err_mask = np.abs(S_data_err / S_sim) < 50
        residuals = np.where(mask & ratio_mask & err_mask, S_data / S_sim - 1, np.nan)
        residual_errs = np.where(mask & ratio_mask & err_mask, S_data_err / np.abs(S_sim), np.nan)
    else:
        residuals = np.where(mask & ratio_mask, S_data / S_sim - 1, np.nan)
    return residuals, residual_errs


def plot_dz_residual(dof_results, output_dir, version=""):
    """Plot fractional DZ residuals (S_data/S_sim - 1) for a single DOF.

    Uses the same k-j subplot layout as create_combined_summary_plot.
    """
    (results_list, gd_results_list, state_key, unit_alpha,
     n_coefs, n_zernikes, file_keys) = extract_dof_info(dof_results)

    if not results_list and not gd_results_list:
        return

    n_datasets = len(results_list) + len(gd_results_list)
    n_subplots = n_coefs - 1
    dataset_colors = plt.cm.tab10.colors[:n_datasets]
    scatter_size = 25

    fig, axes, dataset_width = setup_kj_figure(n_coefs, n_zernikes, n_datasets)

    # Plot FAM residuals with error bars
    errorbar_marksize = 6
    for file_idx, results in enumerate(results_list):
        color = dataset_colors[file_idx]
        x_offset = (file_idx - (n_datasets - 1) / 2) * dataset_width
        x_positions = np.arange(4, n_zernikes) + x_offset

        for ii in range(n_subplots):
            k = ii + 1
            ax = axes[ii]

            S_data = results["data_linear_fits"][k, 4:n_zernikes, 0]
            S_data_err = results["data_fit_errors"][k, 4:n_zernikes, 0]
            S_sim = results["sim_linear_fits"][k, 4:n_zernikes, 0]
            residuals, residual_errs = compute_dz_residual(S_data, S_sim, S_data_err)

            ax.errorbar(
                x_positions, residuals,
                yerr=residual_errs,
                marker='o', color=color,
                capsize=3, elinewidth=1, markersize=errorbar_marksize,
                ls='', alpha=0.8
            )

    # Plot GD residuals
    n_fam = len(results_list)
    for file_idx, gd_results in enumerate(gd_results_list):
        adjusted_file_idx = file_idx + n_fam
        color = dataset_colors[adjusted_file_idx]
        x_offset = (adjusted_file_idx - (n_datasets - 1) / 2) * dataset_width
        x_positions = np.arange(4, n_zernikes) + x_offset

        for ii in range(n_subplots):
            k = ii + 1
            ax = axes[ii]

            S_data = gd_results["measured_sensitivity"][k, 4:n_zernikes]
            S_sim = gd_results["predicted_sensitivity"][k, 4:n_zernikes]
            residuals, _ = compute_dz_residual(S_data, S_sim)

            ax.scatter(x_positions, residuals, marker='^', color=color,
                       s=scatter_size - 5, alpha=0.8)

    fig.supylabel(r"$S_\mathrm{data}\,/\,S_\mathrm{sim} - 1$", fontsize=11)

    title = _wrap_text(f"{state_key}  —  sensitivity matrix fractional residual", 80)

    state_key_str = state_key
    if len(state_key) == 1:
        state_key_str = _pad_state_key(state_key[0])
    gd_fn_str = "+gd" if gd_results_list else ""
    ver_str = f"_{version}" if version else ""
    output_path = output_dir / f"{state_key_str}_sensmat_impact{gd_fn_str}{ver_str}.pdf"

    finalize_kj_figure(fig, axes, file_keys, dataset_colors, state_key,
                       title, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sensitivity matrix impact on inferred DZ coefficients."
    )
    parser.add_argument("files", nargs="*", default=[],
                        help="FAM sensitivity analysis files (used to infer state_key and pkl path)")
    parser.add_argument("--include_giant_donuts", action="store_true",
                        help="Include giant donut data (affects pkl filename via +gd suffix)")
    parser.add_argument("--gd_files", nargs="*", default=[],
                        help="Giant donut files (affects pkl filename via +gd suffix)")
    parser.add_argument("--state_key", type=str, default=None,
                        help="State key string (inferred from FAM files if not given)")
    parser.add_argument("--kmax", type=int, default=6,
                        help="Max focal Zernike index")
    parser.add_argument("--jmax", type=int, default=28,
                        help="Max pupil Zernike index")
    parser.add_argument("-o", "--output", default="sens_results",
                        help="Base output directory (same as fit_plot_dbl_zks_sens.py --output)")
    parser.add_argument("--version", type=str, default="",
                        help="Version string (matches fit_plot_dbl_zks_sens.py --version)")
    args = parser.parse_args()

    if not args.files and not args.gd_files and not args.include_giant_donuts and not args.state_key:
        parser.error("Must provide FAM files, --gd_files/--include_giant_donuts, or --state_key")

    output_dir = Path(args.output + "+sim" + f"_max-k{args.kmax}-j{args.jmax}")
    ver_str = f"_{args.version}" if args.version else ""

    state_key_str, dof_results = find_cached_pkl(
        args.files, args.gd_files, args.include_giant_donuts,
        args.state_key, output_dir, ver_str
    )

    if dof_results is None:
        print(f"No per-DOF results pkl found for state_key='{state_key_str}'.")
        print("\nRun fit_plot_dbl_zks_sens.py first to generate the pkl files.")
        return

    # --- Plot ---
    residual_dir = output_dir / "DOF_combined" / "dz_residuals"
    residual_dir.mkdir(exist_ok=True, parents=True)
    plot_dz_residual(dof_results, residual_dir, version=args.version)
    print("Done.")


if __name__ == "__main__":
    main()
