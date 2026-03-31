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

