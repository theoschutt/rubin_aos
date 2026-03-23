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
import matplotlib.transforms as transforms

def _wrap_text(text, max_line_length):
    """Wraps text into multiple lines at word boundaries."""
    return '\n'.join(textwrap.wrap(text, width=max_line_length))

def _pad_state_key(state_key_str):
    """Adds a leading zero to single digit bending modes to help with sorting."""
    padded_str = re.sub(r'B(\d+)', lambda match: f'B{int(match.group(1)):02d}', state_key_str)
    print(state_key_str, padded_str)
    return padded_str
    
def fit_focal_zernikes(x, y, zk_all, kmax, R_outer=np.deg2rad(1.75)):
    """Fit focal Zernike coefficients to pupil Zernike values."""
    print(f"  Fitting focal Zernikes with x shape: {x.shape}, y shape: {y.shape}, zk_all shape: {zk_all.shape}")
    
    # Assuming x/y are in radians
    basis = galsim.zernike.zernikeBasis(kmax, x, y, R_outer=R_outer, R_inner=0.0)
    print(f"  Basis shape: {basis.shape}")
    
    start = time.time()
    coefs_all, residuals, rank, s = np.linalg.lstsq(basis.T, zk_all, rcond=None)
    elapsed = time.time() - start
    print(f"  Least squares fit completed in {elapsed:.2f} seconds")
    print(f"  Coefficients shape: {coefs_all.shape}, rank: {rank}")
    
    resids_all = basis.T @ coefs_all - zk_all
    print(f"  Residuals shape: {resids_all.shape}")
    print(f"  RMS residual: {np.sqrt(np.mean(resids_all**2)):.6f}")
    
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
        for key in zkTable_asdf.dtype.names:
            # Convert each column to a NumPy array
            zkTable[key] = np.array(zkTable_asdf[key])
        
    print(f"File opened and data converted to native types")
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
    
    print(f"  thx shape: {thx.shape}, range: [{thx.min():.5f}, {thx.max():.5f}] rad")
    print(f"  thy shape: {thy.shape}, range: [{thy.min():.5f}, {thy.max():.5f}] rad")
    print(f"  alpha_values shape: {alpha_values.shape}, value: {alpha_values[0]}")
    print(f"  zk_values shape after limiting to jmax={jmax}: {zk_values.shape}")
    
    # Check that alpha is the same for all rows
    alpha_min, alpha_max = np.min(alpha_values), np.max(alpha_values)
    if not np.allclose(alpha_values, alpha_values[0]):
        print(f"  WARNING: Alpha values vary within exposure")
        print(f"  Alpha range: [{alpha_min}, {alpha_max}], std: {np.std(alpha_values)}")
        
    alpha = float(np.mean(alpha_values))  # Ensure it's a plain float
    print(f"  Using mean alpha value: {alpha}")
    
    # Fit focal Zernikes
    print(f"  Fitting focal Zernikes...")
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
    
    print(f"\nPerforming linear fits across {len(unique_expids)} exposures for {coeff_key}")
    print(f"Number of focal Zernike coefficients: {n_coefs}")
    print(f"Number of pupil Zernikes: {n_zernikes}")
    
    # Prepare data for linear fit
    # if multiple DOFs are moving, we need to just use the dimensionless alpha
    if len(unit_alpha) == 1:
        unit_alpha_multiplier = np.abs(unit_alpha[0])
    else:
        unit_alpha_multiplier = 1.

    print(f"ALPHA INFO: u_a: {unit_alpha}, u_a.dtype: {unit_alpha.dtype}, type(u_a): {type(unit_alpha)}, u_a multiplier: {unit_alpha_multiplier}")

    x_values = np.array([unit_alpha_multiplier * results["expids"][eid]["alpha"]
                        for eid in unique_expids])
    y_values = np.zeros((len(unique_expids), n_coefs, n_zernikes))
    
    print(f"x_values shape: {x_values.shape}, range: [{min(x_values)}, {max(x_values)}]")
    
    for i, eid in enumerate(unique_expids):
        y_values[i] = results["expids"][eid][coeff_key]
    
    print(f"y_values shape: {y_values.shape}")
    
    # Perform linear fit
    linear_fits = np.zeros((n_coefs, n_zernikes, 2))  # [m, b] for each coefficient and Zernike
    fit_errors = np.zeros((n_coefs, n_zernikes, 2))   # Error estimates for [m, b]
    r_squared = np.zeros((n_coefs, n_zernikes))
    
    print("Performing linear fits for each combination...")
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
    
    print(f"linear_fits shape: {linear_fits.shape}")
    print(f"fit_errors shape: {fit_errors.shape}")
    print(f"r_squared shape: {r_squared.shape}")
    print(f"Mean R² value: {np.mean(r_squared):.4f}")

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
        print(f"  Found {sum(mask)} rows for this exposure")
        
        # Process this exposure
        results["expids"][expid] = process_exposure(exp_data, jmax, kmax)
        print(f"  Results stored for exposure {expid}")
    
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
    """Process all giant donut sensitivity analysis files for the given state key. Doesn't work for multi-DOF runs yet."""
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

def create_sensitivity_plots(results, output_dir, date):
    """Create plots showing the sensitivity relationships."""
    state_key = results["state_key"]
    n_coefs = results["linear_fits"].shape[0]
    n_zernikes = results["linear_fits"].shape[1]
    x_values = results["x_values"]
    y_values = results["y_values"]
    
    # Create plots
    for j in range(n_zernikes):
        pupil_zk_name = f"Z{j}"  
        print(f"Creating plots for pupil Zernike {pupil_zk_name} ({j}/{n_zernikes})")
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 15), sharex=True)
        axes = axes.flatten()
        fig.suptitle(f"{state_key}, {pupil_zk_name}, ({date})", fontsize=18)
        
        for k in range(min(n_coefs, len(axes))):
            ax = axes[k]
            focal_zk_name = f"F{k}"  # Focal Zernike index (0-indexed)
            
            # Plot data points
            ax.scatter(x_values, y_values[:, k, j], color='blue', label='Data')
            
            # Plot linear fit
            m, b = results["linear_fits"][k, j]
            r2 = results["r_squared"][k, j]
            x_fit = np.array([min(x_values), max(x_values)])
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, 'r-', 
                    label=f'y = {m:.4f}x + {b:.4f}\nR² = {r2:.4f}')

            if "r" in state_key:
                units = "deg"
            else:
                units = "um"
            ax.set_title(f"{focal_zk_name}")
            ax.set_xlabel(f"{state_key} [unit_alpha * alpha] ({units})")
            # ax.set_ylabel(f"{focal_zk_name} coefficient")
            ax.grid(True)
            ax.legend()
            
        # Hide any unused subplots
        for k in range(n_coefs, len(axes)):
            axes[k].set_visible(False)
            
        plt.tight_layout()
        plot_file = output_dir / f"{state_key}_{pupil_zk_name}_sensitivity_{date}.png"
        print(f"  Saving plot to {plot_file}")
        plt.savefig(plot_file, dpi=150)
        plt.close(fig)

def create_combined_sensitivity_plots(results_list, output_dir):
    """Create plots showing the sensitivity relationships from multiple files with same DOF."""
    if not results_list:
        return
    
    state_key = results_list[0]["state_key"]
    print(f"\nCreating combined plots for DOF: {state_key}")
    
    # Get information about files
    file_keys = [r["file_key"] for r in results_list]
    print(f"  Combining data from {len(file_keys)} files: {file_keys}")
    
    # Determine dimensions from the first result
    n_coefs = results_list[0]["linear_fits"].shape[0]
    n_zernikes = results_list[0]["linear_fits"].shape[1]
    
    # Create plots for each Zernike
    for j in range(n_zernikes):
        pupil_zk_name = f"Z{j}"
        print(f"  Creating combined plots for pupil Zernike {pupil_zk_name} ({j+1}/{n_zernikes})")
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Get a color palette for the different files
        colors = plt.cm.tab10.colors[:len(results_list)]
        
        for i in range(min(n_coefs, len(axes))):
            ax = axes[i]
            focal_zk_name = f"F{i}"
            
            # Plot data from each file
            for file_idx, results in enumerate(results_list):
                color = colors[file_idx]
                x_values = results["x_values"]
                y_values = results["y_values"]
                file_key = results["file_key"]
                
                # Plot data points
                ax.scatter(x_values, y_values[:, i, j], color=color, 
                           s=30, alpha=0.7, label=f"{file_key} data")
                
                # Plot linear fit
                m, b = results["linear_fits"][i, j]
                r2 = results["r_squared"][i, j]
                x_fit = np.array([min(x_values), max(x_values)])
                y_fit = m * x_fit + b
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{file_key}: y = {m:.4f}x + {b:.4f}, R² = {r2:.4f}")
            
            ax.set_title(f"{state_key} {pupil_zk_name} → {focal_zk_name}")
            ax.set_xlabel(f"{state_key} [unit_alpha * alpha]")
            ax.set_ylabel(f"{focal_zk_name} coefficient")
            ax.grid(True)
            
            # Create a more compact legend
            if i == 0:  # Only full legend on first plot
                ax.legend(fontsize='small', loc='best')
            else:
                # Simplified legend for other plots
                handles, labels = ax.get_legend_handles_labels()
                simplified_handles = handles[::2]  # Only take data points
                simplified_labels = [label.split(' ')[0] for label in labels[::2]]
                ax.legend(simplified_handles, simplified_labels, fontsize='small', loc='best')
            
        # Hide any unused subplots
        for i in range(n_coefs, len(axes)):
            axes[i].set_visible(False)
            
        # Create title with all file keys
        combined_title = f"Combined Sensitivity Analysis for {state_key}\nFiles: {', '.join(file_keys)}"
        fig.suptitle(combined_title, fontsize=14, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plot_file = output_dir / f"{state_key}_combined_{pupil_zk_name}_sensitivity.png"
        print(f"    Saving combined plot to {plot_file}")
        plt.savefig(plot_file, dpi=150)
        plt.close(fig)

def create_concatenated_sensitivity_plot(results, output_dir, date):
    """
    Create a comprehensive multi-panel figure with all pupil-focal Zernike combinations.
    Each COLUMN represents one pupil Zernike, each ROW represents one focal Zernike.
    Includes both real data and simulation results.
    """
    state_key = results["state_key"]
    n_coefs = results["data_linear_fits"].shape[0]  # Focal Zernike terms (rows)
    n_zernikes = results["data_linear_fits"].shape[1]  # Pupil Zernike terms (columns)
    x_values = results["x_values"]
    data_y_values = results["data_y_values"]
    sim_y_values = results["sim_y_values"]
    
    print(f"Creating concatenated sensitivity plot with {n_coefs-1} rows and {n_zernikes-4} columns")
    
    # Create subplots with better default spacing
    fig, axes = plt.subplots(n_coefs-1, n_zernikes-4, 
                             figsize=(n_zernikes*1.4, n_coefs*0.8),
                             # sharex=True,  # Share x-axis across columns
                             # sharey=True,  # Share y-axis across rows
                             constrained_layout=True)  # Better than tight_layout for gridded plots
    
    # Get global min/max for y-axes
    vmin = min(np.nanmin(data_y_values), np.nanmin(sim_y_values))
    vmax = max(np.nanmax(data_y_values), np.nanmax(sim_y_values))
    y_range = vmax - vmin
    vmin -= 0.1 * y_range
    vmax += 0.1 * y_range
    
    # Ensure axes is always 2D
    if n_coefs == 1 and n_zernikes == 1:
        axes = np.array([[axes]])
    elif n_coefs == 1:
        axes = axes.reshape(1, n_zernikes)
    elif n_zernikes == 1:
        axes = axes.reshape(n_coefs, 1)
    
    # Create plots for each combination
    for i in range(1, n_coefs):  # Rows (focal Zernikes)
        for j in range(4, n_zernikes):  # Columns (pupil Zernikes)
            # for plotting purposes
            ii = i-1
            jj = j-4
            ax = axes[ii, jj]
            
            # Plot real data points
            ax.scatter(x_values, data_y_values[:, i, j],
                       color='blue', s=15, alpha=0.7, label='Data')
            
            # Plot simulation data points
            ax.scatter(x_values, sim_y_values[:, i, j],
                       color='green', s=15, alpha=0.7, marker='x', label='Sim')
            
            # Plot real data linear fit
            m, b = results["data_linear_fits"][i, j]
            m_err, b_err = results["data_fit_errors"][i, j]
            r2 = results["data_r_squared"][i, j]
            x_fit = np.array([min(x_values), max(x_values)])
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, 'b-', linewidth=1.5, label=f'Data fit')
            
            # Plot simulation linear fit
            sim_m, sim_b = results["sim_linear_fits"][i, j]
            sim_r2 = results["sim_r_squared"][i, j]
            sim_y_fit = sim_m * x_fit + sim_b
            ax.plot(x_fit, sim_y_fit, 'g-', linewidth=1.5, label=f'Sim fit')
            
            # Add fit parameters as text (simple format)
            ax.text(0.05, 0.95, 
                    f"Data: m={m:.3f}±{m_err:.3f}, R²={r2:.2f}\nSim: m={sim_m:.3f}, R²={sim_r2:.2f}", 
                    transform=ax.transAxes, fontsize=6,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                                                       facecolor='white', 
                                                       alpha=0.7))
            
            # Set y-limits to be consistent
            ax.set_ylim(vmin, vmax)
            
            # Remove most tick labels for a cleaner look
            ax.tick_params(labelsize=6)
            
            # Only add titles and labels where needed
            if ii == 0:  # Top row
                ax.set_title(f"j = {j}", fontsize=8)
            
            if jj == 0:  # Leftmost column
                ax.set_ylabel(f"k = {i}", fontsize=8)
            
            # Only show tick labels at the edges
            if jj != 0:  # Not the leftmost column
                ax.set_yticklabels([])
            if i != n_coefs - 1:  # Not the bottom row
                ax.set_xticklabels([])

    # Add a legend to the first subplot only
    if n_coefs > 1 and n_zernikes > 4:
        handles = [
            plt.Line2D([0], [0], marker='o', color='blue', label='Data', markersize=4, linestyle='none'),
            plt.Line2D([0], [0], marker='x', color='green', label='Sim', markersize=4, linestyle='none'),
        ]
        axes[0, 0].legend(handles=handles, fontsize=6, loc='lower right')
    
    # Add overall title and axis labels
    fig.suptitle(f"Sensitivity Analysis for {state_key} ({date})", fontsize=12)
    if "r" in state_key:
        units = "deg"
    else:
        units = "um"
    fig.supxlabel(f"{state_key} perturbation (unit_alpha * alpha) [{units}]", fontsize=10)
    fig.supylabel("Zernike coefficient [um]", fontsize=10)
    
    # Save the plot
    plot_file = output_dir / f"{state_key}_concatenated_sensitivity_+sim_{date}.png"
    print(f"Saving concatenated sensitivity plot to {plot_file}")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_summary_plot(results, output_dir, date):
    """Create a summary plot of sensitivity coefficients."""
    state_key = results["state_key"]
    n_coefs = results["data_linear_fits"].shape[0]
    n_zernikes = results["data_linear_fits"].shape[1]

    plasma_cmap = plt.cm.plasma
    colors = [plasma_cmap(j/n_zernikes) for j in range(n_zernikes)]
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h', '*']
    
    print("Creating summary plot of sensitivity coefficients")
    fig, ax = plt.subplots(figsize=(6,6))
    for j in range(4, n_zernikes):
        pupil_zk_name = f"Z{j}"
        data_slopes = results["data_linear_fits"][:, j, 0]  # m values
        data_slope_errors = results["data_fit_errors"][:, j, 0]  # m error values
        sim_slopes = results["sim_linear_fits"][:, j, 0]  # m values
        
        # Plot real data with error bars
        ax.errorbar(
            np.arange(1, n_coefs) + (j/2 - 4 + 0.5)*0.2,
            data_slopes[1:],
            yerr=data_slope_errors[1:],
            marker=markers[j%len(markers)],
            # color=colors[j],
            color='k',
            label=f"{pupil_zk_name} data",
            capsize=3,
            elinewidth=1.5,
            ls='',
            alpha=0.8
        )
        
        # Plot simulation data as points
        ax.scatter(
            np.arange(1, n_coefs) + (j/2 - 4 + 0.5)*0.2,
            sim_slopes[1:],
            marker='x',  # Use 'x' to distinguish from data points
            # color=colors[j],
            color='b',
            s=30,
            # label=f"{pupil_zk_name} sim"
        )

    # Set integer ticks on x-axis
    x_ticks = np.arange(1, n_coefs)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])
    
    # Add grid lines aligned with integer ticks
    ax.set_xticks(x_ticks, minor=False)
    ax.grid(True, which='major', axis='y')

    ax.set_xlabel("Focal Zernike Index")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{state_key} ({date})")

    # Create a more readable legend
    unique_labels = []
    handles = []
    labels = []
    
    h, l = ax.get_legend_handles_labels()

    # Keep only one instance of each Zernike (for both data and sim)
    for i, label in enumerate(l):
        zk = label.split(' ')[0]  # Extract Zernike name
        if zk not in unique_labels:
            unique_labels.append(zk)
            handles.append(h[i])
            labels.append(zk)
    
    # Add two more entries to explain the markers
    handles.append(plt.Line2D([0], [0], color='gray', marker='o', linestyle='none'))
    handles.append(plt.Line2D([0], [0], color='gray', marker='x', linestyle='none'))
    labels.append('Data')
    labels.append('Sim')

    
    ax.legend(handles, labels, ncol=1, loc='center left', bbox_to_anchor=(1., 0.5))
    plt.tight_layout()  # Adjust to make room for the legend
    
    
    summary_file = output_dir / f"{state_key}_sensitivity_summary_+sim_{date}.pdf"
    print(f"Saving summary plot to {summary_file}")
    plt.savefig(summary_file, dpi=150)
    plt.close(fig)

def create_combined_summary_plot(results_list, output_dir, gd_results_list=None, state_key_override=None):
    """
    Create a summary plot of sensitivity coefficients for multiple datasets.
    Plots both real data (with error bars) and simulation data (as points).

    Parameters:
    -----------
    results_list : list
        List of result dictionaries, each from a different data set but same DOF.
        Can be empty if gd_results_list is provided (GD-only mode).
    output_dir : Path
        Directory to save the output plot
    gd_results_list : list, optional
        List of giant donut result dictionaries
    state_key_override : str, optional
        State key to use when results_list is empty (GD-only mode)
    """
    if not results_list and not gd_results_list:
        return

    # Get DOF, dimensions, and file keys depending on available data
    if results_list:
        state_key = results_list[0]["state_key"]
        unit_alpha = results_list[0]["unit_alpha"]
        file_keys = ["FAM " + r.get("file_key", f"Dataset {i}") for i, r in enumerate(results_list)]
        n_coefs = results_list[0]["data_linear_fits"].shape[0]
        n_zernikes = results_list[0]["data_linear_fits"].shape[1]
    else:
        # GD-only mode — derive dimensions from giant donut data
        sk = state_key_override if state_key_override else gd_results_list[0]["reason"][:8]
        state_key = [sk] if isinstance(sk, str) else sk
        unit_alpha = gd_results_list[0]["ds"]
        dof_names = gd_results_list[0]["dof_names"]
        file_keys = []
        n_coefs = gd_results_list[0]["measured_sensitivity"].shape[0]
        n_zernikes = gd_results_list[0]["measured_sensitivity"].shape[1]

    print(f"\nCreating combined summary plot for DOF: {state_key}")
    print(f"  Combining data from {len(file_keys)} FAM files + {len(gd_results_list) if gd_results_list else 0} GD files")

    n_datasets = len(results_list) + (len(gd_results_list) if gd_results_list is not None else 0)
    
    # Define markers for different Zernikes
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h', '*']
    
    # Get a color palette for the different datasets
    dataset_colors = plt.cm.tab10.colors[:n_datasets]
    
    print("Creating combined summary plot of sensitivity coefficients")
    fig, ax = plt.subplots(figsize=(15,6))
    
    # Calculate stagger parameters
    n_active_zernikes = n_zernikes - 4  # We're using Zernikes from Noll index 4 onward
    
    total_width = 1. # Total width for all points at focal Zernike
    # gap_width = 0.05 # gap between datasets for one focal Zernike
    dataset_width = total_width / n_datasets
    zk_width = dataset_width / n_active_zernikes # space b/w data points

    # set up for plotting labels etc
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for file_idx, results in enumerate(results_list):
        # where does the first point in the dataset get plotted
        dataset_base_offset = (zk_width-total_width)/2  + file_idx * zk_width

        # Get the color for this dataset
        color = dataset_colors[file_idx]

        for j in range(4, n_zernikes):
            jj = j - 4
            pupil_zk_name = f"Z{j}"

            # Calculate offset for this pupil Zernike within the dataset
            zernike_offset = jj * n_datasets * zk_width # interleaves datasets
            total_offset = dataset_base_offset + zernike_offset
            # Calculate x positions with offset
            x_positions = np.arange(1, n_coefs) + total_offset
 
            # Get slopes for data and simulation
            data_slopes = results["data_linear_fits"][:, j, 0]  # m values
            data_slope_errors = results["data_fit_errors"][:, j, 0]  # m error values
            sim_slopes = results["sim_linear_fits"][:, j, 0]  # m values
            
            # Create label for legend
            label = f"{pupil_zk_name}" if file_idx == 0 else None
            
            # Plot real data with error bars
            ax.errorbar(
                x_positions,
                data_slopes[1:],
                yerr=data_slope_errors[1:],
                # marker=markers[j % len(markers)],
                marker='o',
                color=color,
                # label=label,
                capsize=3,
                elinewidth=1,
                markersize=6,
                ls='',
                alpha=0.8
            )
            
            # Plot simulation data as 'x' markers
            ax.scatter(
                x_positions,
                sim_slopes[1:],
                marker='x',
                color=color,
                s=25,
                alpha=0.8
            )

            # add visual clarity
            if (file_idx == 0):
                for xpos in x_positions:
                    # add j labeling
                    ax.text(xpos + zk_width * (n_datasets - 1)/2, 0., j,
                    transform=trans, horizontalalignment='center',
                    verticalalignment='bottom'
                    )
                    if j%2 == 0:
                        # add shaded boxes grouping j Zernikes
                        ax.axvspan(
                            xpos - zk_width/2,
                            xpos + n_datasets * zk_width - zk_width/2,
                            color='k',
                            alpha=0.1
                        )

    # add giant donut sensitivities
    if gd_results_list is not None:
        for file_idx, gd_results in enumerate(gd_results_list):
            
            adjusted_file_idx = file_idx + len(results_list)
            file_keys.append(f"GD {gd_results['reason'][:8]}")
            
            # Get the color for this dataset
            color = dataset_colors[adjusted_file_idx]
            # where does the first point in the dataset get plotted
            dataset_base_offset = (zk_width-total_width)/2  + adjusted_file_idx * zk_width

            for j in range(4, n_zernikes):
                jj = j - 4
                pupil_zk_name = f"Z{j}"

                # Calculate offset for this pupil Zernike within the dataset
                zernike_offset = jj * n_datasets * zk_width # interleaves datasets
                total_offset = dataset_base_offset + zernike_offset
                # Calculate x positions with offset
                x_positions = np.arange(1, n_coefs) + total_offset

                # Get slopes for data and simulation
                data_gd = gd_results["measured_sensitivity"][:, j]
                sim_gd = gd_results["predicted_sensitivity"][:, j]

                # # Create label for legend
                # label = f"{pupil_zk_name}" if file_idx == 0 else None

                # Plot real data with error bars
                ax.scatter(
                    x_positions,
                    data_gd[1:n_coefs],
                    marker='^',
                    color=color,
                    s=25,
                    alpha=0.8
                )

                # Plot real data with error bars
                ax.scatter(
                    x_positions,
                    sim_gd[1:n_coefs],
                    marker='x',
                    color=color,
                    s=25,
                    alpha=0.8
                )

                # add visual clarity
                if (file_idx == 0):
                    for xpos in x_positions:
                        # add j labeling
                        ax.text(xpos + zk_width * (n_datasets - 1)/2, 0., j,
                        transform=trans, horizontalalignment='center',
                        verticalalignment='bottom'
                        )
                        if j%2 == 0:
                            # add shaded boxes grouping j Zernikes
                            ax.axvspan(
                                xpos - zk_width/2,
                                xpos + n_datasets * zk_width - zk_width/2,
                                color='k',
                                alpha=0.1
                            )
    # Add unit alpha info
    ax.text(1.01, 0.98, "Step size [um or deg]:", transform=ax.transAxes)
    for idx, sk in enumerate(state_key):
        ua_list = []
        for results in results_list:
            ua_list.append(results["unit_alpha"][idx])
        if ua_list:
            ua_text = str([f"{ua:.5f}" for ua in ua_list])
            ax.text(1.02, 0.98 - (idx+1)*0.04,
                f"{sk}: " + ua_text,
                transform=ax.transAxes)
    if gd_results_list is not None:
        # unit alpha info for giant donuts
        for idx, (ua, dof_name) in enumerate(zip(unit_alpha, dof_names)):
            ax.text(1.02, 0.98 - (idx+1)*0.04,
                    f"{dof_name}: " + f"{ua:.5f}",
                    transform=ax.transAxes)
        # divide focal zernikes
        for dz in range(1, n_coefs):
            ax.axvline(dz+0.5, lw=1., c='gray')

    # Set integer ticks on x-axis
    ax.set_xlim(0.5, 3.5)
    x_ticks = np.arange(1, n_coefs)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])

    # Add grid lines
    ax.grid(True, axis='y', alpha=0.5)

    if len(state_key) > 1 or len(unit_alpha) > 1:
        units = "dimless"
    elif "r" in state_key:
        units = "deg"
    else:
        units = "um"
    ax.set_xlabel("Focal Zernike Index", fontsize=12)
    ax.set_ylabel(f"Sensitivity Coefficient [wavefront um / DOF {units}]", fontsize=12)
    ax.set_title(_wrap_text(f"{state_key}", 80), fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    for file_idx, file_key in enumerate(file_keys):
        color = dataset_colors[file_idx]
        marker = 'o'
        if 'GD' in file_key:
            marker = '^'
        handles.append(plt.Line2D([0], [0], color=color, marker=marker, linestyle='none', markersize=5))
        labels.append(file_key)
    
    # Sim marker meaning
    handles.append(plt.Line2D([0], [0], color='k', marker='x', linestyle='none', markersize=5))
    labels.append('Simulation')

    leg = ax.legend(handles, labels, ncols=1, loc='lower left',
                    bbox_to_anchor=(1.01, 0.), fontsize=10)    
    plt.tight_layout()
    
    # Save the plot
    state_key_str = state_key
    if len(state_key) == 1:
        state_key_str = _pad_state_key(state_key[0])
    gd_fn_str = "+gd" if gd_results_list is not None else ""
    summary_file = output_dir / f"{state_key_str}_combined_sensitivity_summary_+sim{gd_fn_str}.pdf"
    print(f"  Saving combined summary plot to {summary_file}")
    plt.savefig(summary_file, dpi=150)
    
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
    args = parser.parse_args()

    if not args.files and not args.gd_files and not args.include_giant_donuts:
        parser.error("Must provide either FAM files, --gd_files, or --include_giant_donuts (or a combination)")
    if not args.files and not args.state_key:
        parser.error("--state_key is required when no FAM files are provided")

    print(f"\n=== Sensitivity Analysis Data Extraction ===")
    print(f"Files to process: {len(args.files)}")
    print(f"Output directory: {args.output}")
    print(f"Maximum focal Zernike coefficient (kmax): {args.kmax}")
    print(f"Maximum pupil Zernike coefficient (jmax): {args.jmax}")
    
    output_dir = Path(args.output + "+sim" + f"_max-k{args.kmax}-j{args.jmax}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory created: {output_dir}")
    
    all_results = {}
    state_key_str = args.state_key  # May be None if FAM files will set it

    for i, filename in enumerate(args.files):
        print(f"\nProcessing file {i+1}/{len(args.files)}: {filename}")
        file_start_time = time.time()

        with asdf.open(filename) as af:
            state_key = af["state_key"]
            if len(state_key) == 1:
                state_key_str = f"{state_key[0]}"
            else:
                state_key_str = f"{state_key}"

        # Process the file
        results = process_sensitivity_file(filename, jmax=args.jmax, kmax=args.kmax)

        # Save results
        file_key = Path(filename).stem[21:34]
        date = file_key[:8]
        all_results[file_key] = results

        # Generate plots - skipping
        subdir = state_key_str + "_" + file_key
        plot_dir = output_dir / subdir
        # plot_results(results, plot_dir, date)

        file_elapsed = time.time() - file_start_time
        print(f"File {i+1}/{len(args.files)} completed in {file_elapsed:.2f} seconds")

    # Create combined plots by DOF (still handles one dataset)
    print("\n=== Creating combined plots for same DOFs ===")
    combined_results = combine_results_by_dof(all_results)

    combined_dir = output_dir / f"DOF_combined"
    combined_dir.mkdir(exist_ok=True)

    # Process giant donut data
    gd_results_list = None
    if args.include_giant_donuts or args.gd_files:
        gd_results_list = process_giant_donuts(args.gd_files if args.gd_files else None, state_key_str, kmax=args.kmax, jmax=args.jmax)

    if combined_results:
        # FAM data exists — plot per-DOF (with optional GD overlay)
        for dof, results_list in combined_results.items():
            print(f"\nCombining {len(results_list)} files for DOF: {dof}")
            create_combined_summary_plot(results_list, combined_dir, gd_results_list)
            # create_combined_concatenated_plot(results_list, combined_dir)
    elif gd_results_list:
        # GD-only mode — no FAM data
        print("\nGD-only mode: creating summary plot from giant donut data")
        create_combined_summary_plot([], combined_dir, gd_results_list, state_key_override=state_key_str)

    # Save all results to a pickle file
    results_to_save = {"fam_results": all_results}
    if gd_results_list is not None:
        results_to_save["gd_results"] = gd_results_list
    results_file = output_dir / f"all_results.pkl"
    print(f"\nSaving all results to {results_file}")
    with open(results_file, "wb") as f:
        pickle.dump(results_to_save, f)

    print(f"All processing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    overall_start = time.time()
    main()
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal execution time: {overall_elapsed:.2f} seconds")