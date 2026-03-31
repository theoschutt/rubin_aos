import glob
import re
import numpy as np
import asdf
import matplotlib.pyplot as plt
import galsim
from astropy import units as u
from pathlib import Path
import argparse
import pickle
import time
import textwrap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _wrap_text(text, max_line_length):
    """Wraps text into multiple lines at word boundaries."""
    return '\n'.join(textwrap.wrap(text, width=max_line_length))

def _pad_state_key(state_key_str):
    """Adds a leading zero to single digit bending modes to help with sorting."""
    padded_str = re.sub(r'B(\d+)', lambda match: f'B{int(match.group(1)):02d}', state_key_str)
    # print(state_key_str, padded_str)
    return padded_str
    
def fit_focal_zernikes(x, y, zk_all, kmax, R_outer=np.deg2rad(1.75)):
    """Fit focal Zernike coefficients to pupil Zernike values."""
    # print(f"  Fitting focal Zernikes with x shape: {x.shape}, y shape: {y.shape}, zk_all shape: {zk_all.shape}")
    
    # Assuming x/y are in radians
    basis = galsim.zernike.zernikeBasis(kmax, x, y, R_outer=R_outer, R_inner=0.0)
    # print(f"  Basis shape: {basis.shape}")
    
    start = time.time()
    coefs_all, residuals, rank, s = np.linalg.lstsq(basis.T, zk_all, rcond=None)
    elapsed = time.time() - start
    # print(f"  Least squares fit completed in {elapsed:.2f} seconds")
    # print(f"  Coefficients shape: {coefs_all.shape}, rank: {rank}")
    
    resids_all = basis.T @ coefs_all - zk_all
    # print(f"  Residuals shape: {resids_all.shape}")
    # print(f"  RMS residual: {np.sqrt(np.mean(resids_all**2)):.6f}")
    
    return coefs_all, resids_all

def extract_data_from_file(filename):
    """Extract data from a sensitivity analysis file and convert to native Python types."""
    print(f"Opening ASDF file...")
    with asdf.open(filename) as af:
        # Extract data and convert to native Python/NumPy types
        zkTable_asdf = af["zkTable"]
        state_key = af["state_key"].copy()
        unit_alpha = af["unit_alpha"].copy()
        
        # Convert zkTable to a fully loaded structure that doesn't depend on the file
        zkTable = {}
        try:
            for key in zkTable_asdf.dtype.names:
                # Convert each column to a NumPy array
                zkTable[key] = np.array(zkTable_asdf[key])
        except AttributeError:
            for i, colname in enumerate(zkTable_asdf['colnames']):
                coldata = zkTable_asdf['columns'][i]
                if 'data' in coldata.keys():
                    zkTable[colname] = coldata['data'][:]
                elif 'value' in coldata.keys():
                    zkTable[colname] = coldata['value'][:]
                else:
                    pass
        
    # print(f"File opened and data converted to native types")
    print(f"zkTable has data for {len(zkTable['expid'])} rows")
    print(f"state_key: {state_key}")
    print(f"unit_alpha: {unit_alpha}")
    
    return zkTable, state_key, unit_alpha

def process_exposure(exp_data, jmax, kmax):
    """Process a single exposure's data."""
    # Extract required information
    thx = exp_data["thx_OCS"]
    if hasattr(thx, "to_value"):  # Handle astropy Quantity objects
        thx = thx.to_value(u.rad)  # Convert to radians
    
    thy = exp_data["thy_OCS"]
    if hasattr(thy, "to_value"):
        thy = thy.to_value(u.rad)  # Convert to radians
    
    alpha_values = exp_data["alpha"]
    zk_values = exp_data["zk_OCS"]
    zk_sim_values = exp_data["zk_sim_OCS"]
    
    # Limit to the specified pupil Zernike indices
    zk_values = zk_values[:, :jmax+1]  # +1 because we want to include jmax
    zk_sim_values = zk_sim_values[:, :jmax+1]  # +1 because we want to include jmax
    
    # print(f"  thx shape: {thx.shape}, range: [{thx.min():.5f}, {thx.max():.5f}] rad")
    # print(f"  thy shape: {thy.shape}, range: [{thy.min():.5f}, {thy.max():.5f}] rad")
    # print(f"  alpha_values shape: {alpha_values.shape}, value: {alpha_values[0]}")
    # print(f"  zk_values shape after limiting to jmax={jmax}: {zk_values.shape}")
    
    # Check that alpha is the same for all rows
    alpha_min, alpha_max = np.min(alpha_values), np.max(alpha_values)
    if not np.allclose(alpha_values, alpha_values[0]):
        print(f"  WARNING: Alpha values vary within exposure")
        print(f"  Alpha range: [{alpha_min}, {alpha_max}], std: {np.std(alpha_values)}")
        
    alpha = float(np.mean(alpha_values))  # Ensure it's a plain float
    # print(f"  Using mean alpha value: {alpha}")
    
    # Fit focal Zernikes
    # print(f"  Fitting focal Zernikes...")
    data_coefs, data_resids = fit_focal_zernikes(thx, thy, zk_values, kmax)
    sim_coefs, sim_resids = fit_focal_zernikes(thx, thy, zk_sim_values, kmax)
    
    return {
        "alpha": alpha,
        "focal_zernike_data_coeffs": data_coefs,
        "focal_zernike_sim_coeffs": sim_coefs,
        "data_residuals": data_resids,
        "sim_residuals": sim_resids
    }

def perform_linear_fits(results, unique_expids, unit_alpha, coeff_key="focal_zernike_data_coeffs"):
    """Perform linear fits across FAM exposures for each Zernike term."""
    # Get the first exposure's data to determine dimensions
    first_exp = results["expids"][unique_expids[0]]
    n_coefs = first_exp[coeff_key].shape[0]
    n_zernikes = first_exp[coeff_key].shape[1]
    
    # print(f"\nPerforming linear fits across {len(unique_expids)} exposures for {coeff_key}")
    # print(f"Number of focal Zernike coefficients: {n_coefs}")
    # print(f"Number of pupil Zernikes: {n_zernikes}")
    
    # Prepare data for linear fit
    # if multiple DOFs are moving, we need to just use the dimensionless alpha
    if len(unit_alpha) == 1:
        unit_alpha_multiplier = np.abs(unit_alpha[0])
    else:
        unit_alpha_multiplier = 1.

    # print(f"ALPHA INFO: u_a: {unit_alpha}, u_a.dtype: {unit_alpha.dtype}, type(u_a): {type(unit_alpha)}, u_a multiplier: {unit_alpha_multiplier}")

    x_values = np.array([unit_alpha_multiplier * results["expids"][eid]["alpha"]
                        for eid in unique_expids])
    y_values = np.zeros((len(unique_expids), n_coefs, n_zernikes))
    
    # print(f"x_values shape: {x_values.shape}, range: [{min(x_values)}, {max(x_values)}]")
    
    for i, eid in enumerate(unique_expids):
        y_values[i] = results["expids"][eid][coeff_key]
    
    # print(f"y_values shape: {y_values.shape}")
    
    # Perform linear fit
    linear_fits = np.zeros((n_coefs, n_zernikes, 2))  # [m, b] for each coefficient and Zernike
    fit_errors = np.zeros((n_coefs, n_zernikes, 2))   # Error estimates for [m, b]
    r_squared = np.zeros((n_coefs, n_zernikes))
    
    # print("Performing linear fits for each combination...")
    for k in range(n_coefs):
        for j in range(n_zernikes):
            # Fit y = mx + b using np.polyfit with cov=True to get error estimates
            coef, cov = np.polyfit(x_values, y_values[:, k, j], 1, cov=True)

            # print(f"coef shape: {coef.shape}, cov shape: {cov.shape}")

            linear_fits[k, j] = coef  # [m, b]
            
            # Calculate standard errors from the covariance matrix
            errors = np.sqrt(np.diag(cov))
            fit_errors[k, j] = errors  # [m_err, b_err]
     
            # coef = np.polyfit(x_values, y_values[:, k, j], 1)
            # linear_fits[k, j] = coef
            
            # Calculate R²
            m, b = coef
            y_pred = m * x_values + b
            y_actual = y_values[:, k, j]
            ss_total = np.sum((y_actual - np.mean(y_actual))**2)
            ss_residual = np.sum((y_actual - y_pred)**2)
            r_squared[k, j] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

            # print(f"  Fit for F{k}/Z{j}: m={coef[0]:.6f}, b={coef[1]:.6f}, R²={r_squared[k, j]:.4f}")
    
    # print(f"linear_fits shape: {linear_fits.shape}")
    # print(f"fit_errors shape: {fit_errors.shape}")
    # print(f"r_squared shape: {r_squared.shape}")
    # print(f"Mean R² value: {np.mean(r_squared):.4f}")

    return linear_fits, fit_errors, r_squared, x_values, y_values

def process_sensitivity_file(filename, jmax, kmax):
    """Process a sensitivity analysis file and return the results."""
    print(f"\n=== Processing {filename} ===")
    start_time = time.time()
    results = {}

    # Extract data from file
    zkTable, state_key, unit_alpha = extract_data_from_file(filename)
    
    results["state_key"] = state_key
    results["unit_alpha"] = unit_alpha
    
    # Get list of unique exposures
    unique_expids = np.unique(zkTable["expid"])
    print(f"Found {len(unique_expids)} unique exposures: {unique_expids}")
    results["expids"] = {}
    
    # Process each exposure
    for i, expid in enumerate(unique_expids):
        print(f"\nProcessing exposure {i+1}/{len(unique_expids)}: {expid}")
        # Get rows for this exposure
        mask = (zkTable["expid"] == expid)
        exp_data = {key: zkTable[key][mask] for key in zkTable.keys()}
        # print(f"  Found {sum(mask)} rows for this exposure")
        
        # Process this exposure
        results["expids"][expid] = process_exposure(exp_data, jmax, kmax)
        # print(f"  Results stored for exposure {expid}")
    
    # # Perform linear fits
    # linear_fits, r_squared, x_values, y_values = perform_linear_fits(
    #     results, unique_expids, unit_alpha
    # )
    # Perform linear fits
    # linear_fits, fit_errors, r_squared, x_values, y_values = perform_linear_fits(
    #     results, unique_expids, unit_alpha
    # )
    # Perform linear fits for real data
    data_linear_fits, data_fit_errors, data_r_squared, x_values, data_y_values = perform_linear_fits(
        results, unique_expids, unit_alpha, coeff_key="focal_zernike_data_coeffs"
    )
    
    # Perform linear fits for simulation data
    sim_linear_fits, sim_fit_errors, sim_r_squared, _, sim_y_values = perform_linear_fits(
        results, unique_expids, unit_alpha, coeff_key="focal_zernike_sim_coeffs"
    )
    
    # Store the results
    results["x_values"] = x_values

    results["data_linear_fits"] = data_linear_fits
    results["data_fit_errors"] = data_fit_errors
    results["data_r_squared"] = data_r_squared
    results["data_y_values"] = data_y_values
    
    results["sim_linear_fits"] = sim_linear_fits
    results["sim_fit_errors"] = sim_fit_errors
    results["sim_r_squared"] = sim_r_squared
    results["sim_y_values"] = sim_y_values
    
    elapsed_time = time.time() - start_time
    print(f"\nFile processing completed in {elapsed_time:.2f} seconds")
    
    return results

def process_giant_donuts(gd_files=None, state_key_str=None, kmax=None, jmax=None):
    """Process all giant donut sensitivity analysis files for the given state key."""
    # get DOF names to match to the ds array
    dof_order = []
    for i, prefix in enumerate(["M2_", "Cam_"]):
        for j, suffix in enumerate(["z", "x", "y", "rx", "ry"]):
            dof_order.append(f'{prefix}{suffix}')
    for i in range(20):
        dof_order.append(f"M1M3_B{i+1}")
    for i in range(20):
        dof_order.append(f"M2_B{i+1}")
    dof_order = np.array(dof_order)

    if gd_files is None:
        print(f"Retrieving giant donut data for {state_key_str}.")
        
        # convert to GD file name convention
        if state_key_str == 'M2_z':
            state_key_str = 'm2_dz'
        elif state_key_str == 'Cam_z':
            state_key_str = 'cam_dz'
        # glob all matching files and sort
        gd_dir = "/sdf/data/rubin/u/jmeyers3/projects/aos/Hartmann/dz_matrix/"
        gd_files = glob.glob(gd_dir + f"hartmann_zernike_sensitivity*{state_key_str.lower()}.asdf")

        if len(gd_files) == 0:
            print(f"No giant donut data found.")
            return None
        else:
            print(f"Processing giant donut data ({len(gd_files)} datasets).")

    gd_files.sort()
    gd_results_list = []
    for gd_file in gd_files:
        ff = asdf.open(gd_file)
        # get ds
        mask = ~np.isclose(ff['ds'], np.zeros_like(ff['ds']), atol=1e-2)
        sweep = ff['ds'][mask]
        dof_names = dof_order[mask]
        print(sweep)
        if len(sweep) > 1:
            multiplier = 1. # multiple things moving - don't divide out by the sweep
        else:
            multiplier = sweep
        # sweep = ff['ds'][np.nonzero(ff['ds'])[0]]
        # get sensitivities
        ms = ff['measured_sensitivity'] / 1000 / multiplier # nm to um/um (or just um for v/z modes)
        ps = ff['predicted_sensitivity'] / 1000 / multiplier # nm to um/um

        # Limit to kmax (focal Zernikes, axis 0) and jmax (pupil Zernikes, axis 1)
        if kmax is not None:
            ms = ms[:kmax+1, :]
            ps = ps[:kmax+1, :]
        if jmax is not None:
            ms = ms[:, :jmax+1]
            ps = ps[:, :jmax+1]

        gd_results_dict = {}
        gd_results_dict['reason'] = ff['reason']
        gd_results_dict['ds'] = sweep
        gd_results_dict['dof_names'] = dof_names
        gd_results_dict['measured_sensitivity'] = ms
        gd_results_dict['predicted_sensitivity'] = ps
        gd_results_list.append(gd_results_dict)

    return gd_results_list

def combine_results_by_dof(all_results):
    """
    Combine results that have the same state_key (DOF).
    Returns a dictionary with state_key as keys, and lists of results as values.
    """
    combined = {}
    
    for file_key, result in all_results.items():
        state_key = result["state_key"]
        state_key_str = str(state_key)
        if state_key_str not in combined:
            combined[state_key_str] = []
        
        # Add file_key to the result for reference
        result["file_key"] = file_key
        combined[state_key_str].append(result)
    
    return combined

def load_cached_results(pkl_path):
    """Load a per-DOF results pkl. Returns None if the file does not exist."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return None
    print(f"Loading cached results from {pkl_path}")
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    n_fam = len(loaded.get("fam_results", {}))
    n_gd = len(loaded.get("gd_results", []))
    print(f"Loaded {n_fam} FAM result sets and {n_gd} GD result sets")
    return loaded

def process_and_save(fam_files, gd_files, include_gd, state_key_str,
                     jmax, kmax, output_dir, ver_str):
    """
    Process FAM and GD files, combine FAM results by DOF, save per-DOF pkls.

    Returns all_results: dict mapping dof_key -> dof_results dict (keys:
    'fam_results', 'gd_results', 'state_key_str').
    """
    print(f"\n=== Sensitivity Analysis Data Extraction ===")
    print(f"Files to process: {len(fam_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Maximum focal Zernike coefficient (kmax): {kmax}")
    print(f"Maximum pupil Zernike coefficient (jmax): {jmax}")

    # --- FAM processing ---
    fam_results = {}  # keyed by file_key
    for i, filename in enumerate(fam_files):
        print(f"\nProcessing file {i+1}/{len(fam_files)}: {filename}")
        file_start_time = time.time()

        with asdf.open(filename) as af:
            state_key = af["state_key"]
            state_key_str = state_key[0] if len(state_key) == 1 else str(list(state_key))

        results = process_sensitivity_file(filename, jmax=jmax, kmax=kmax)
        file_key = Path(filename).stem[21:34]
        fam_results[file_key] = results

        print(f"File {i+1}/{len(fam_files)} completed in {time.time() - file_start_time:.2f} seconds")

    combined_fam_results = combine_results_by_dof(fam_results)

    # --- GD processing ---
    gd_results_list = []
    if include_gd or gd_files:
        gd_results_list = process_giant_donuts(
            gd_files if gd_files else None, state_key_str, kmax=kmax, jmax=jmax
        ) or []

    # --- Build and save per-DOF dicts ---
    comb_out_dir = output_dir / "DOF_combined" / "all_results"
    comb_out_dir.mkdir(exist_ok=True, parents=True)
    gd_fn_str = "+gd" if gd_results_list else ""
    all_results = {}

    if combined_fam_results:
        for dof, results_list in combined_fam_results.items():
            state_key_padded = _pad_state_key(dof) if len(results_list[0]["state_key"]) == 1 else dof
            pkl_file = comb_out_dir / f"{state_key_padded}_all_results{gd_fn_str}{ver_str}.pkl"
            dof_results = {
                "fam_results": {r["file_key"]: r for r in results_list},
                "state_key_str": dof,
            }
            if gd_results_list:
                dof_results["gd_results"] = gd_results_list
            print(f"  Saving per-DOF results to {pkl_file}")
            with open(pkl_file, "wb") as f:
                pickle.dump(dof_results, f)
            all_results[dof] = dof_results

    elif gd_results_list:
        # GD-only mode
        gd_dof_names = gd_results_list[0].get("dof_names", [state_key_str])
        gd_state_key_padded = _pad_state_key(gd_dof_names[0]) if len(gd_dof_names) == 1 else str(gd_dof_names)
        pkl_file = comb_out_dir / f"{gd_state_key_padded}_GD_results{ver_str}.pkl"
        dof_results = {
            "gd_results": gd_results_list,
            "state_key_str": state_key_str,
        }
        print(f"  Saving GD-only results to {pkl_file}")
        with open(pkl_file, "wb") as f:
            pickle.dump(dof_results, f)
        all_results[gd_state_key_padded] = dof_results

    return all_results

def create_combined_summary_plot(dof_results, output_dir, version=""):
    """
    Create a summary plot of sensitivity coefficients for a single DOF.
    Plots FAM data (with error bars), GD data (triangles), and simulation (x markers).

    Parameters:
    -----------
    dof_results : dict
        Dict with optional keys 'fam_results' (maps file_key -> result dict),
        'gd_results' (list of GD result dicts), and 'state_key_str' (fallback
        state key string for GD-only mode).
    output_dir : Path
        Directory to save the output plot
    version : str, optional
        Version string appended to output file name
    """
    fam_data = dof_results.get("fam_results", {})
    results_list = list(fam_data.values()) if fam_data else []
    gd_results_list = dof_results.get("gd_results") or []

    if not results_list and not gd_results_list:
        return

    # Get DOF, dimensions, and file keys depending on available data
    if results_list:
        state_key = results_list[0]["state_key"]
        unit_alpha = results_list[0]["unit_alpha"]
        n_coefs = results_list[0]["data_linear_fits"].shape[0]
        n_zernikes = results_list[0]["data_linear_fits"].shape[1]
    else:
        # GD-only mode — derive dimensions from giant donut data
        sk = dof_results.get("state_key_str") or gd_results_list[0]["reason"][:8]
        state_key = [sk] if isinstance(sk, str) else sk
        unit_alpha = gd_results_list[0]["ds"]
        n_coefs = gd_results_list[0]["measured_sensitivity"].shape[0]
        n_zernikes = gd_results_list[0]["measured_sensitivity"].shape[1]

    # Build complete file_keys list upfront for both FAM and GD
    file_keys = (["FAM " + r.get("file_key", f"Dataset {i}") for i, r in enumerate(results_list)]
                 if results_list else [])
    if gd_results_list:
        file_keys += [f"GD {gd['reason'][:8]}" for gd in gd_results_list]
        dof_names = gd_results_list[0]['dof_names']

    print(f"\nCreating combined summary plot for DOF: {state_key}")
    print(f"  Combining data from {len(results_list) if results_list else 0} FAM files"
          f" + {len(gd_results_list) if gd_results_list else 0} GD files")

    n_datasets = (len(results_list) if results_list else 0) + (len(gd_results_list) if gd_results_list else 0)
    n_active_zernikes = n_zernikes - 4
    n_subplots = n_coefs - 1  # one subplot per focal Zernike k=1..n_coefs-1

    # Get a color palette for the different datasets
    dataset_colors = plt.cm.tab10.colors[:n_datasets]
    errorbar_marksize = 6
    scatter_marksize = 25

    # Dynamic figure width based on number of j points and datasets
    fig_width = max(10, n_active_zernikes * 0.4 + n_datasets * 0.3) + 4
    fig_height = max(4, n_subplots * 2.2)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_width, fig_height),
                             sharex=True, constrained_layout=True)
    if n_subplots == 1:
        axes = [axes]

    # Stagger datasets within each j position: 0.8 total width, 0.2 gap
    stagger_total = 0.95
    dataset_width = stagger_total / n_datasets

    # Add alternating shading for even j positions (same on all subplots)
    for j in range(4, n_zernikes):
        if j % 2 == 0:
            for ax in axes:
                ax.axvspan(j - 0.5, j + 0.5, color='k', alpha=0.07, lw=0)

    # Plot FAM results
    for file_idx, results in enumerate(results_list if results_list else []):
        color = dataset_colors[file_idx]
        x_offset = (file_idx - (n_datasets - 1) / 2) * dataset_width
        x_positions = np.arange(4, n_zernikes) + x_offset

        for ii in range(n_subplots):
            k = ii + 1
            ax = axes[ii]

            data_slopes = results["data_linear_fits"][k, 4:n_zernikes, 0]
            data_slope_errors = results["data_fit_errors"][k, 4:n_zernikes, 0]
            sim_slopes = results["sim_linear_fits"][k, 4:n_zernikes, 0]

            ax.errorbar(
                x_positions, data_slopes,
                yerr=data_slope_errors,
                marker='o', color=color,
                capsize=3, elinewidth=1, markersize=errorbar_marksize,
                ls='', alpha=0.8
            )
            ax.scatter(x_positions, sim_slopes, marker='x', color=color, s=scatter_marksize, alpha=0.8)

    # Plot GD results
    n_fam = len(results_list) if results_list else 0
    for file_idx, gd_results in enumerate(gd_results_list if gd_results_list else []):
        adjusted_file_idx = file_idx + n_fam
        color = dataset_colors[adjusted_file_idx]
        x_offset = (adjusted_file_idx - (n_datasets - 1) / 2) * dataset_width
        x_positions = np.arange(4, n_zernikes) + x_offset

        for ii in range(n_subplots):
            k = ii + 1
            ax = axes[ii]

            data_gd = gd_results["measured_sensitivity"][k, 4:n_zernikes]
            sim_gd = gd_results["predicted_sensitivity"][k, 4:n_zernikes]

            ax.scatter(x_positions, data_gd, marker='^', color=color, s=20, alpha=0.8)
            ax.scatter(x_positions, sim_gd, marker='x', color=color, s=20, alpha=0.8)

    # Per-subplot formatting
    for ii, ax in enumerate(axes):
        k = ii + 1
        ax.set_ylabel(f"k={k}", fontsize=11)
        ax.axhline(0, color='gray', lw=0.5, ls='-')
        ax.grid(True, axis='y', alpha=0.4)

    # X-axis on bottom subplot only (sharex handles the rest)
    axes[-1].set_xticks(range(4, n_zernikes))
    axes[-1].set_xticklabels([str(j) for j in range(4, n_zernikes)], fontsize=11)
    axes[-1].set_xlim(4-0.5, n_zernikes-0.5)
    axes[-1].set_xlabel("Pupil Zernike Index (j)", fontsize=11)

    # Unit alpha annotation on top subplot
    ax_top = axes[0]
    ax_top.text(1.01, 0.99, "Step size [um or deg]:", transform=ax_top.transAxes,
                fontsize=11, va='top')
    key_ct = 0
    for idx, sk in enumerate(state_key):
        ua_list = [r["unit_alpha"][idx] for r in (results_list if results_list else [])]
        if ua_list:
            ax_top.text(1.02, 0.99 - (idx + 1) * 0.12,
                        f"{sk}: " + str([f"{ua:.5f}" for ua in ua_list]),
                        transform=ax_top.transAxes, fontsize=11, va='top')
        key_ct += 1
    if gd_results_list:
        ax_top.text(1.02, 0.99 - (key_ct + 1) * 0.12, 'Giant donuts:',
                    transform=ax_top.transAxes, fontsize=11, va='top')
        for idx, dof_name in enumerate(dof_names):
            ua_list = [gd["ds"][idx] for gd in gd_results_list]
            ax_top.text(1.02, 0.99 - (key_ct + idx + 2) * 0.12,
                        f"{dof_name}: " + str([f"{ua:.5f}" for ua in ua_list]),
                        transform=ax_top.transAxes, fontsize=11, va='top')

    if len(state_key) > 1 or len(unit_alpha) > 1:
        units = "dimless"
    elif "r" in state_key:
        units = "deg"
    else:
        units = "um"
    fig.suptitle(_wrap_text(f"{state_key} [wavefront um / DOF {units}]", 80), fontsize=12)

    # Legend on bottom subplot
    handles, labels = [], []
    for file_idx, file_key in enumerate(file_keys):
        color = dataset_colors[file_idx]
        marker = '^' if 'GD' in file_key else 'o'
        handles.append(plt.Line2D([0], [0], color=color, marker=marker, linestyle='none', markersize=errorbar_marksize))
        labels.append(file_key)
    handles.append(plt.Line2D([0], [0], color='k', marker='x', linestyle='none', markersize=errorbar_marksize))
    labels.append('Simulation')
    axes[-1].legend(handles, labels, ncols=1, loc='lower left',
                    bbox_to_anchor=(1.01, 0.), fontsize=11)

    # Save the plot
    state_key_str = state_key
    if len(state_key) == 1:
        state_key_str = _pad_state_key(state_key[0])
    gd_fn_str = "+gd" if gd_results_list else ""
    ver_str = f"_{version}" if version else ""
    summary_file = output_dir / f"{state_key_str}_combined_sensitivity_summary_+sim{gd_fn_str}{ver_str}.pdf"
    print(f"  Saving combined summary plot to {summary_file}")
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_combined_concatenated_plot(results_list, output_dir):
    """
    Create a comprehensive multi-panel figure showing all pupil-focal Zernike combinations
    for multiple data sets with the same DOF.
    
    Each COLUMN represents one pupil Zernike, each ROW represents one focal Zernike.
    Different data sets are shown with different colors within each subplot.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries, each from a different data set but same DOF
    output_dir : Path
        Directory to save the output plot
    """
    if not results_list:
        return
    
    # Get DOF and file keys
    state_key = results_list[0]["state_key"]
    file_keys = [r.get("file_key", f"Dataset {i}") for i, r in enumerate(results_list)]
    
    print(f"\nCreating combined concatenated plot for DOF: {state_key}")
    print(f"  Combining data from {len(file_keys)} files: {file_keys}")
    
    # Determine dimensions from the first result
    n_coefs = results_list[0]["data_linear_fits"].shape[0] # Focal Zernike terms (rows)
    n_zernikes = results_list[0]["data_linear_fits"].shape[1] # Pupil Zernike terms (columns)
 
    
    # Create subplots with better default spacing
    fig, axes = plt.subplots(n_coefs-1, n_zernikes-4, 
                             figsize=(n_zernikes*1.2, n_coefs*0.8),
                             constrained_layout=True)
    
    # Get a color palette for the different files
    dataset_colors = plt.cm.tab10.colors[:len(results_list)]
    
    # Ensure axes is always 2D
    if n_coefs == 1 and n_zernikes == 1:
        axes = np.array([[axes]])
    elif n_coefs == 1:
        axes = axes.reshape(1, n_zernikes)
    elif n_zernikes == 1:
        axes = axes.reshape(n_coefs, 1)
    
    # Get global min/max for y-axes
    vmin = np.inf
    vmax = -np.inf
    for results in results_list:
        vmin = min(vmin, np.nanmin(results["data_y_values"]), np.nanmin(results["sim_y_values"]))
        vmax = max(vmax, np.nanmax(results["data_y_values"]), np.nanmax(results["sim_y_values"]))
    y_range = vmax - vmin
    vmin -= 0.1 * y_range
    vmax += 0.1 * y_range
    
    # Create plots for each combination
    for i in range(1, n_coefs):  # Rows (focal Zernikes)
        for j in range(4, n_zernikes):  # Columns (pupil Zernikes)
            ii = i - 1
            jj = j - 4
            ax = axes[ii, jj]
            
            # Plot data from each file
            for file_idx, results in enumerate(results_list):
                color = dataset_colors[file_idx]
                file_key = results.get("file_key", f"Dataset {file_idx}")
                x_values = results["x_values"]
                data_y_values = results["data_y_values"]
                sim_y_values = results["sim_y_values"]

                # Plot data points (small and transparent)
                ax.scatter(x_values, data_y_values[:, i, j], color=color, 
                           s=10, alpha=0.5, marker='o')
                
                # Plot simulation points (small and transparent)
                ax.scatter(x_values, sim_y_values[:, i, j], color=color, 
                           s=10, alpha=0.5, marker='x')
                
                # Plot data linear fit
                m, b = results["data_linear_fits"][i, j]
                r2 = results["data_r_squared"][i, j]
                
                # Plot simulation linear fit
                sim_m, sim_b = results["sim_linear_fits"][i, j]
                sim_r2 = results["sim_r_squared"][i, j]
                
                x_fit = np.array([min(x_values), max(x_values)])
                
                # Data fit
                y_fit = m * x_fit + b
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=1.5)
                
                # Sim fit (dashed line)
                sim_y_fit = sim_m * x_fit + sim_b
                ax.plot(x_fit, sim_y_fit, '--', color=color, linewidth=1.5)
                
                # Add legend only to the first subplot
                if ii == 0 and jj == 0 and file_idx == 0:
                    handles = [
                        plt.Line2D([0], [0], marker='o', color='gray', label='Data',
                                   markersize=4, linestyle='none'),
                        plt.Line2D([0], [0], marker='x', color='gray', label='Sim',
                                   markersize=4, linestyle='none'),
                        plt.Line2D([0], [0], linestyle='-', color='gray', label='Data fit', markersize=0),
                        plt.Line2D([0], [0], linestyle='--', color='gray', label='Sim fit', markersize=0)
                    ]
                    ax.legend(handles=handles, fontsize=6, loc='lower right')
            
            # Add text showing slopes
            slope_text = ""
            for file_idx, results in enumerate(results_list):
                file_key = results.get("file_key", f"Dataset {file_idx}")
                m = results["data_linear_fits"][i, j][0]
                sim_m = results["sim_linear_fits"][i, j][0]
                slope_text += f"{file_key}:\nData m={m:.1e}\nSim m={sim_m:.1e}\n\n"
            
            ax.text(0.05, 0.95, slope_text, 
                    transform=ax.transAxes, fontsize=6,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                                                      facecolor='white', 
                                                      alpha=0.7))
     
            # Set y-limits to be consistent
            ax.set_ylim(vmin, vmax)
            
            # Set tick label size
            ax.tick_params(labelsize=6)
            
            # Add titles
            if ii == 0:  # Top row
                ax.set_title(f"j = {j}", fontsize=8)
            
            if jj == 0:  # Leftmost column
                ax.set_ylabel(f"k = {i}", fontsize=8)
            
            # Only show tick labels at the edges
            if jj != 0:  # Not the leftmost column
                ax.set_yticklabels([])
            if i != n_coefs - 1:  # Not the bottom row
                ax.set_xticklabels([])
            
            # Add a light grid
            ax.grid(True, alpha=0.2)
    
    # Create a legend for the datasets
    handles, labels = [], []
    for file_idx, results in enumerate(results_list):
        color = dataset_colors[file_idx]
        file_key = results.get("file_key", f"Dataset {file_idx}")
        handles.append(plt.Line2D([0], [0], color=color, lw=2))
        labels.append(file_key)
    
    # Add legend in a good position
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(5, len(results_list)), fontsize='small')
    
    # Add overall title and axis labels
    fig.suptitle(f"Combined Sensitivity Analysis for {state_key}, ({file_keys})",
                 fontsize=10)
    if "r" in state_key:
        units = "deg"
    else:
        units = "um"
    fig.supxlabel(f"{state_key} perturbation (unit_alpha * alpha) [{units}]",
                  fontsize=8)
    fig.supylabel("Zernike coefficient [um]", fontsize=8)
    
    # Save the plot
    plot_file = output_dir / f"{state_key}_combined_concatenated_sensitivity_+sim.png"
    print(f"  Saving combined concatenated plot to {plot_file}")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_matrix_visualization(results, output_dir, date=None):
    """
    Create a condensed matrix visualization of sensitivity data for a single dataset.
    Displays four matrices: data slopes, simulation slopes, data errors, and normalized residuals.
    
    Parameters:
    -----------
    results : dict
        Results dictionary for a single dataset
    output_dir : Path
        Directory to save the output plot
    date : str, optional
        Date string to include in the title and filename
    """
    state_key = results["state_key"]
    print(f"\nCreating matrix visualization for: {state_key}")
    
    # Extract matrices
    data_slopes = results["data_linear_fits"][1:, 4:, 0]  # m values (first index in the last dimension)
    sim_slopes = results["sim_linear_fits"][1:, 4:, 0]    # m values
    data_errors = results["data_fit_errors"][1:, 4:, 0]   # m error values
    
    # Calculate normalized residuals
    # Use np.divide with 'where' to avoid division by zero
    norm_residuals = np.divide(
        data_slopes - sim_slopes, 
        data_errors, 
        out=np.zeros_like(data_slopes), 
        where=data_errors!=0
    )

    n_coefs, n_zernikes = data_slopes.shape
    
    # Create figure
    fig, axes = plt.subplots(4,1, figsize=(5, 8))
    
    # Common params for imshow
    common_params = {
        'aspect': 'auto',
        'origin': 'upper',
        'interpolation': 'nearest'
    }
    
    # Set titles and specific parameters for each matrix
    matrices = [
        (data_slopes, "Data sensitivity"),
        (sim_slopes, "Design sensitivity"),
        (data_errors, "Data errors (standard deviation)"),
        (norm_residuals, "(Data-Design)/Error")
    ]

    # Create colorbars with appropriate scales for each matrix
    for i, (matrix, title) in enumerate(matrices):
        ax = axes[i]
        
        # For errors matrix, use a different colormap and scale
        if i == 2:  # Data errors
            vmin = 0
            vmax = np.max(matrix)
            cmap = 'viridis'
        else:  # Other matrices
            vmax = np.max(np.abs(matrix))
            vmin = -vmax
            cmap = 'RdBu_r'
        
        # Plot the matrix
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, **common_params)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        
        # Set title
        ax.set_title(title)
        
        # Set labels
        ax.set_ylabel("k")
        ax.set_xlabel("j")
        
        # Set ticks
        ax.set_yticks(np.arange(n_coefs))
        ax.set_xticks(np.arange(n_zernikes))
        ax.set_yticklabels(np.arange(1, n_coefs+1))
        ax.set_xticklabels(np.arange(4, n_zernikes+4))
        
        # Set grid lines between cells (at 0.5, 1.5, 2.5, etc.)
        ax.set_yticks(np.arange(-0.5, n_coefs), minor=True)
        ax.set_xticks(np.arange(-0.5, n_zernikes), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1.0, alpha=0.7)
        ax.grid(which="major", visible=False)
    
    title_text = f"{state_key}"
    if date:
        title_text += f" ({date})"
    fig.suptitle(title_text, fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    filename = f"{state_key}_matrix_visualization"
    if date:
        filename += f"_{date}"
    
    matrix_file = output_dir / f"{filename}.png"
    pdf_file = output_dir / f"{filename}.pdf"
    plt.savefig(pdf_file, dpi=150)
    
    plt.close(fig)

def plot_results(results, output_dir, date):
    """Generate all plots for the results."""
    print(f"\n=== Generating plots in {output_dir} ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    state_key = results["state_key"]
    unit_alpha = results["unit_alpha"]
    
    print(f"State key: {state_key}")
    print(f"Unit alpha: {unit_alpha}")
    print(f"Number of exposures: {len(results['x_values'])}")
    
    # Create the individual sensitivity plots
    # create_sensitivity_plots(results, output_dir, date)
    
    # Create the summary plot
    create_summary_plot(results, output_dir, date)
    
    # Create the concatenated sensitivity plot
    create_concatenated_sensitivity_plot(results, output_dir, date)

    # Create matrix visualization plot
    create_matrix_visualization(results, output_dir, date)
    
    print("Plotting completed")

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze sensitivity data")
    parser.add_argument("files", nargs="*", default=[], help="FAM sensitivity analysis files to process (optional if --gd_files provided)")
    parser.add_argument("--include_giant_donuts", action="store_true", help="Include giant donut sensitivity analysis results if available")
    parser.add_argument("--gd_files", nargs="*", default=[], help="Giant donut sensitivity analysis files to process (optional if using FAM files)")
    parser.add_argument("--state_key", type=str, default=None, help="State key string (required for GD-only mode without FAM files)")
    parser.add_argument("--output", "-o", default="sens_results", help="Output directory for plots")
    parser.add_argument("--kmax", type=int, default=3, help="Maximum focal Zernike coefficient to fit")
    parser.add_argument("--jmax", type=int, default=28, help="Maximum pupil Zernike coefficient to fit")
    parser.add_argument("--use_cached", action="store_true",
                        help="Load per-DOF results pkl from the output dir instead of reprocessing files")
    parser.add_argument("--version", type=str, default="", help="Version string appended to output file names")
    args = parser.parse_args()

    if not args.files and not args.gd_files and not args.include_giant_donuts and not args.use_cached:
        parser.error("Must provide FAM files, --gd_files, --include_giant_donuts, or --use_cached")
    if not args.files and not args.state_key:
        parser.error("--state_key is required when no FAM files are provided")

    output_dir = Path(args.output + "+sim" + f"_max-k{args.kmax}-j{args.jmax}")
    output_dir.mkdir(exist_ok=True, parents=True)
    combined_dir = output_dir / "DOF_combined"
    combined_dir.mkdir(exist_ok=True)
    ver_str = f"_{args.version}" if args.version else ""

    # --- Try to load cached results ---
    all_results = {}
    if args.use_cached:
        state_key_str = args.state_key
        if not state_key_str and args.files:
            with asdf.open(args.files[0]) as af:
                sk = af["state_key"]
                state_key_str = sk[0] if len(sk) == 1 else str(list(sk))
        if state_key_str:
            gd_fn_str = "+gd" if (args.gd_files or args.include_giant_donuts) else ""
            cached_pkl = (output_dir / "DOF_combined" / "all_results" /
                          f"{_pad_state_key(state_key_str)}_all_results{gd_fn_str}{ver_str}.pkl")
            loaded = load_cached_results(cached_pkl)
            if loaded is not None:
                all_results[state_key_str] = loaded

    # --- Process if nothing was loaded ---
    if not all_results:
        if args.use_cached:
            print("Cached results not found, processing files...")
        all_results = process_and_save(
            args.files, args.gd_files, args.include_giant_donuts,
            args.state_key, args.jmax, args.kmax, output_dir, ver_str
        )

    # --- Plot ---
    print("\n=== Creating plots ===")
    for dof_results in all_results.values():
        create_combined_summary_plot(dof_results, combined_dir, version=args.version)

    print(f"All processing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    overall_start = time.time()
    main()
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal execution time: {overall_elapsed:.2f} seconds")