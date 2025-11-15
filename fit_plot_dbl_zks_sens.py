import numpy as np
import asdf
import matplotlib.pyplot as plt
import galsim
from astropy import units as u
from pathlib import Path
import argparse
import pickle
import time

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
        state_key = str(af["state_key"][0])  # Convert to string
        unit_alpha = float(af["unit_alpha"])  # Convert to float
        
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
    coefs, resids = fit_focal_zernikes(thx, thy, zk_values, kmax)
    
    return {
        "alpha": alpha,
        "focal_zernike_coeffs": coefs,
        "residuals": resids,
    }

def perform_linear_fits(results, unique_expids, unit_alpha):
    """Perform linear fits across exposures for each Zernike term."""
    # Get the first exposure's data to determine dimensions
    first_exp = results["expids"][unique_expids[0]]
    n_coefs = first_exp["focal_zernike_coeffs"].shape[0]
    n_zernikes = first_exp["focal_zernike_coeffs"].shape[1]
    
    print(f"\nPerforming linear fits across {len(unique_expids)} exposures")
    print(f"Number of focal Zernike coefficients: {n_coefs}")
    print(f"Number of pupil Zernikes: {n_zernikes}")
    
    # Prepare data for linear fit
    x_values = np.array([unit_alpha * results["expids"][eid]["alpha"] 
                        for eid in unique_expids])
    y_values = np.zeros((len(unique_expids), n_coefs, n_zernikes))
    
    print(f"x_values shape: {x_values.shape}, range: [{min(x_values)}, {max(x_values)}]")
    
    for i, eid in enumerate(unique_expids):
        y_values[i] = results["expids"][eid]["focal_zernike_coeffs"]
    
    print(f"y_values shape: {y_values.shape}")
    
    # Perform linear fit
    linear_fits = np.zeros((n_coefs, n_zernikes, 2))  # [m, b] for each coefficient and Zernike
    fit_errors = np.zeros((n_coefs, n_zernikes, 2))   # Error estimates for [m, b]
    r_squared = np.zeros((n_coefs, n_zernikes))
    
    print("Performing linear fits for each combination...")
    for k in range(n_coefs):
        for j in range(n_zernikes):
            # Fit y = mx + b
            # Fit y = mx + b using np.polyfit with cov=True to get error estimates
            coef, cov = np.polyfit(x_values, y_values[:, k, j], 1, cov=True)

            print(f"coef shape: {coef.shape}, cov shape: {cov.shape}")

            # Store the coefficients
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

            print(f"  Fit for F{k}/Z{j}: m={coef[0]:.6f}, b={coef[1]:.6f}, R²={r_squared[k, j]:.4f}")
    
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
    
    # Store state_key and unit_alpha
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
    linear_fits, fit_errors, r_squared, x_values, y_values = perform_linear_fits(
        results, unique_expids, unit_alpha
    )
    
    results["linear_fits"] = linear_fits
    results["fit_errors"] = fit_errors
    results["r_squared"] = r_squared
    results["x_values"] = x_values
    results["y_values"] = y_values
    
    elapsed_time = time.time() - start_time
    print(f"\nFile processing completed in {elapsed_time:.2f} seconds")
    
    return results

def combine_results_by_dof(all_results):
    """
    Combine results that have the same state_key (DOF).
    Returns a dictionary with state_key as keys, and lists of results as values.
    """
    combined = {}
    
    for file_key, result in all_results.items():
        state_key = result["state_key"]
        if state_key not in combined:
            combined[state_key] = []
        
        # Add file_key to the result for reference
        result["file_key"] = file_key
        combined[state_key].append(result)
    
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
    """
    state_key = results["state_key"]
    n_coefs = results["linear_fits"].shape[0]  # Focal Zernike terms (rows)
    n_zernikes = results["linear_fits"].shape[1]  # Pupil Zernike terms (columns)
    x_values = results["x_values"]
    y_values = results["y_values"]
    
    print(f"Creating concatenated sensitivity plot with {n_coefs-1} rows and {n_zernikes-4} columns")
    
    # Create subplots with better default spacing
    fig, axes = plt.subplots(n_coefs-1, n_zernikes-4, 
                             figsize=(n_zernikes*1.4, n_coefs*0.8),
                             # sharex=True,  # Share x-axis across columns
                             # sharey=True,  # Share y-axis across rows
                             constrained_layout=True)  # Better than tight_layout for gridded plots
    
    # Get global min/max for y-axes
    vmin = np.nanmin(y_values)
    vmax = np.nanmax(y_values)
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
            
            # Plot data points
            ax.scatter(x_values, y_values[:, i, j], color='blue', s=15)
            
            # Plot linear fit
            m, b = results["linear_fits"][i, j]
            m_err, b_err = results["fit_errors"][i, j]
            r2 = results["r_squared"][i, j]
            x_fit = np.array([min(x_values), max(x_values)])
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, 'r-', linewidth=1.5)
            
            # Add fit parameters as text (simple format)
            ax.text(0.05, 0.95, f"m={m:.3f}±{m_err:.3f}, R²={r2:.2f}", 
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
    
    # Add overall title and axis labels
    fig.suptitle(f"Sensitivity Analysis for {state_key} ({date})", fontsize=12)
    if "r" in state_key:
        units = "deg"
    else:
        units = "um"
    fig.supxlabel(f"{state_key} perturbation (unit_alpha * alpha) [{units}]", fontsize=10)
    fig.supylabel("Zernike coefficient [um]", fontsize=10)
    
    # Save the plot
    plot_file = output_dir / f"{state_key}_concatenated_sensitivity_{date}.png"
    print(f"Saving concatenated sensitivity plot to {plot_file}")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_summary_plot(results, output_dir, date):
    """Create a summary plot of sensitivity coefficients."""
    state_key = results["state_key"]
    n_coefs = results["linear_fits"].shape[0]
    n_zernikes = results["linear_fits"].shape[1]

    plasma_cmap = plt.cm.plasma
    colors = [plasma_cmap(j/n_zernikes) for j in range(n_zernikes)]
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h', '*']
    
    print("Creating summary plot of sensitivity coefficients")
    fig, ax = plt.subplots(figsize=(12, 8))
    for j in range(4, n_zernikes):
        pupil_zk_name = f"Z{j}"
        slopes = results["linear_fits"][:, j, 0]  # m values
        slope_errors = results["fit_errors"][:, j, 0]  # m error values
        
        # Plot with error bars
        ax.errorbar(
            range(0, n_coefs),
            slopes,
            yerr=slope_errors,
            marker=markers[j%len(markers)],
            color=colors[j],
            label=pupil_zk_name,
            capsize=3,
            elinewidth=1.5,
        )
        # ax.plot(
        #     range(n_coefs),
        #     slopes,
        #     marker=markers[j%len(markers)],
        #     color=colors[j],
        #     label=pupil_zk_name
        # )

    # Set integer ticks on x-axis
    x_ticks = np.arange(0, n_coefs)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])
    
    # Add grid lines aligned with integer ticks
    ax.set_xticks(x_ticks, minor=False)
    ax.grid(True, which='major', axis='y')

    ax.set_xlabel("Focal Zernike Index")
    ax.set_ylabel("Sensitivity (Slope)")
    ax.set_title(f"Sensitivity of Focal Zernikes to {state_key} ({date})")
    ax.legend(ncol=5)
    plt.tight_layout()

    # Create a more readable legend
    # if n_zernikes > 15:
    #     # For many Zernikes, use a multi-column legend outside the plot
    #     ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small')
    #     plt.tight_layout(rect=[0, 0.1, 1, 0.98])  # Adjust to make room for the legend
    # else:
    #     # For fewer Zernikes, legend can fit on the side
    #     ax.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust to make room for the legend
    
    
    summary_file = output_dir / f"{state_key}_sensitivity_summary_{date}.png"
    print(f"Saving summary plot to {summary_file}")
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
    n_coefs = results_list[0]["linear_fits"].shape[0]  # Focal Zernike terms (rows)
    n_zernikes = results_list[0]["linear_fits"].shape[1]  # Pupil Zernike terms (columns)
    
    # Create subplots with better default spacing
    fig, axes = plt.subplots(n_coefs-1, n_zernikes-4, 
                             figsize=(n_zernikes*1.2, n_coefs*0.8),  # Transposed dimensions
                             constrained_layout=True)  # Better than tight_layout for gridded plots
    
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
        vmin = min(vmin, np.nanmin(results["y_values"]))
        vmax = max(vmax, np.nanmax(results["y_values"]))
    
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
                y_values = results["y_values"]

                # Only add label to the first subplot (will be used for legend)
                # label = file_key if ii == 0 and jj == 0 else None
                # Plot data points (small and transparent)
                ax.scatter(x_values, y_values[:, i, j], color=color, 
                           s=10, alpha=0.5, label=None)
                
                # Plot linear fit
                m, b = results["linear_fits"][i, j]
                m_err, b_err = results["fit_errors"][i, j]
                r2 = results["r_squared"][i, j]
                
                x_fit = np.array([min(x_values), max(x_values)])
                y_fit = m * x_fit + b
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=1.5,
                        label=f"m={m:.1e}±{m_err:.1e}, R²={r2:.2f}")
                ax.legend(fontsize=4)
                
                # # Add fit parameters as text on first row
                # if ii == 0:
                #     text_y = 0.95 - file_idx * 0.15  # Stagger text vertically
                #     text_color = color
                # else:
                #     continue  # Only add text on first row to avoid clutter
                
                # ax.text(0.05, text_y, f"m={m:.3f}", 
                #         transform=ax.transAxes, fontsize=6, color=text_color,
                #         verticalalignment='top', bbox=dict(boxstyle='round', 
                #                                           facecolor='white', 
                #                                           alpha=0.5))
            
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
    plot_file = output_dir / f"{state_key}_combined_concatenated_sensitivity.png"
    print(f"  Saving combined concatenated plot to {plot_file}")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
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
    
    print("Plotting completed")

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze sensitivity data")
    parser.add_argument("files", nargs="+", help="Sensitivity analysis files to process")
    parser.add_argument("--output", "-o", default="sens_results", help="Output directory for plots")
    parser.add_argument("--kmax", type=int, default=3, help="Maximum focal Zernike coefficient to fit")
    parser.add_argument("--jmax", type=int, default=28, help="Maximum pupil Zernike coefficient to fit")
    parser.add_argument("--combine", action="store_true", help="Combine plots for the same DOF")
    args = parser.parse_args()
    
    print(f"\n=== Sensitivity Analysis Data Extraction ===")
    print(f"Files to process: {len(args.files)}")
    print(f"Output directory: {args.output}")
    print(f"Maximum focal Zernike coefficient (kmax): {args.kmax}")
    print(f"Maximum pupil Zernike coefficient (jmax): {args.jmax}")
    print(f"Combine same DOF: {args.combine}")
    
    output_dir = Path(args.output + f"_max-k{args.kmax}-j{args.jmax}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory created: {output_dir}")
    
    all_results = {}
    
    for i, filename in enumerate(args.files):
        print(f"\nProcessing file {i+1}/{len(args.files)}: {filename}")
        file_start_time = time.time()

        with asdf.open(filename) as af:
            state_key = af["state_key"]
            state_key_str = f"{state_key[0]}"
            if len(state_key) > 1:
                state_key_str += "++"
        
        # Process the file
        results = process_sensitivity_file(filename, jmax=args.jmax, kmax=args.kmax)
        
        # Save results
        file_key = Path(filename).stem[21:34]
        date = file_key[:8]
        all_results[file_key] = results
        
        # Generate plots
        subdir = state_key_str + "_" + file_key
        plot_dir = output_dir / subdir
        plot_results(results, plot_dir, date)
        
        file_elapsed = time.time() - file_start_time
        print(f"File {i+1}/{len(args.files)} completed in {file_elapsed:.2f} seconds")

    # Create combined plots by DOF if requested
    if args.combine and len(args.files) > 1:
        print("\n=== Creating combined plots for same DOFs ===")
        combined_results = combine_results_by_dof(all_results)
        
        combined_dir = output_dir / f"DOF_combined"
        combined_dir.mkdir(exist_ok=True)
        
        for dof, results_list in combined_results.items():
            if len(results_list) > 1:
                print(f"\nCombining {len(results_list)} files for DOF: {dof}")
                # create_combined_sensitivity_plots(results_list, combined_dir)
                create_combined_concatenated_plot(results_list, combined_dir)
                # create_combined_summary_plot(results_list, combined_dir)
            else:
                print(f"Only one file for DOF {dof}, skipping combination")
    
    # Save all results to a pickle file
    results_file = output_dir / f"all_results.pkl"
    print(f"\nSaving all results to {results_file}")
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    
    print(f"All processing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    overall_start = time.time()
    main()
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal execution time: {overall_elapsed:.2f} seconds")