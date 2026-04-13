"""
Utilities for the DZ-to-DOF inversion problem.

Loads/slices a sensitivity matrix, reshapes DZ coefficients from table
columns, solves the least-squares problem A @ x_hat ≈ dz, and provides
plotting and printing helpers.

Index conventions
-----------------
- Sensitivity matrix: ``sens[k, j, dof]`` — k = focal Zernike, j = pupil
  Zernike, dof = AOS degree of freedom.
- Table column names: ``z1toz6_z{j}_c{k}`` — **j (pupil) outer, k (focal)
  inner**.  So iterating columns gives (n_pupil, n_focal) order.
- Design matrix A: C-order flatten of (n_focal, n_pupil, n_dof) →
  ``A[k * n_pupil + j_idx, dof]``.  k varies slowly, j fast.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import galsim
from lsst.ts.ofc import OFCData # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOF_LABELS = (
    ["M2_hex_z", "M2_hex_x", "M2_hex_y", "M2_hex_rx", "M2_hex_ry",
     "Cam_hex_z", "Cam_hex_x", "Cam_hex_y", "Cam_hex_rx", "Cam_hex_ry"]
    + [f"M1M3_B{i}" for i in range(1, 21)]
    + [f"M2_B{i}" for i in range(1, 21)]
)
"""50-element list of DOF names in OFC ordering."""

N_DOF = len(DOF_LABELS)

DOF_GROUPS = {
    "M2 hexapod":  slice(0, 5),
    "Cam hexapod": slice(5, 10),
    "M1M3 bends":  slice(10, 30),
    "M2 bends":    slice(30, 50),
}


# =========================================================================
# Section 1: Sensitivity Matrix
# =========================================================================
def load_ofc_data():
    ofc_config_dir = ('/sdf/home/s/schutt20/repos/lsst-ts/'
                      'ts_config_mttcs/MTAOS/v13/ofc')
    ofc_data = OFCData('lsst', config_dir=ofc_config_dir)

    return ofc_data

def load_sensitivity_matrix(ofc_data, focal_indices, pupil_indices, renorm=False):
    """Load, slice and optionally renormalize the OFC sensitivity matrix.

    Returns
    -------
    sliced : ndarray, shape (n_focal, n_pupil, n_dof)
    full_coef : ndarray
        Full DoubleZernike coefficient array (for heatmap plots).
    """
    ideal_sens_dz = galsim.zernike.DoubleZernike(
        ofc_data.sensitivity_matrix[..., :],
        uv_inner=ofc_data.config["field"]["radius_inner"],
        uv_outer=ofc_data.config["field"]["radius_outer"],
        xy_inner=ofc_data.config["pupil"]["radius_inner"],
        xy_outer=ofc_data.config["pupil"]["radius_outer"],
    )
    full_coef = ideal_sens_dz.coef
    print(f"Full sensitivity matrix shape: {full_coef.shape}")

    renorm_full_coef = renormalize_sensitivity_matrix(
        ofc_data, full_coef) if renorm else None

    sliced = slice_sensitivity_matrix(renorm_full_coef if renorm else full_coef,
                                      focal_indices, pupil_indices)
    print(f"Sliced sensitivity matrix shape: {sliced.shape}")
    return sliced, full_coef, renorm_full_coef

def renormalize_sensitivity_matrix(ofc_data, smatrix, dof_idx=range(50)):

    # The normalization is defined in Eqs 9-11 of
    # https://ui.adsabs.harvard.edu/abs/2024ApJ...974..108M

    norm_matrix = np.diag(ofc_data.normalization_weights[dof_idx])
    renorm_smatrix = smatrix @ norm_matrix

    return renorm_smatrix

def reverse_normalization(ofc_data, dof_vector, dof_idx=range(50)):
    """Reverses normalization of DOF vector."""
    norm_vector = ofc_data.normalization_weights[dof_idx]
    print('Renorm weights:', norm_vector)
    return dof_vector * norm_vector

def slice_sensitivity_matrix(sens_coef, focal_indices, pupil_indices):
    """Select focal and pupil Zernike indices from a full sensitivity tensor.

    Parameters
    ----------
    sens_coef : ndarray, shape (K_full, J_full, n_dof)
        Full double-Zernike sensitivity coefficient array, e.g.
        ``ideal_sens_dz.coef`` from OFCData.
    focal_indices : array_like of int
        Focal (field) Zernike indices to keep (k dimension).
    pupil_indices : array_like of int
        Pupil Zernike indices to keep (j dimension).

    Returns
    -------
    ndarray, shape (n_focal, n_pupil, n_dof)
    """
    return sens_coef[np.ix_(focal_indices, pupil_indices)][..., :]


def build_design_matrix(sliced_smatrix):
    """Reshape a 3-D sensitivity tensor into the 2-D design matrix A.

    C-order reshape: focal (k) varies *slowly*, pupil (j) varies *fast*.

        A[k * n_pupil + j, dof] == sliced_smatrix[k, j, dof]

    Parameters
    ----------
    sliced_smatrix : ndarray, shape (n_focal, n_pupil, n_dof)

    Returns
    -------
    A : ndarray, shape (n_focal * n_pupil, n_dof)
    """
    n_focal, n_pupil, n_dof = sliced_smatrix.shape
    return sliced_smatrix.reshape(-1, n_dof)


# =========================================================================
# Section 2: DZ Data Wrangling
# =========================================================================

def make_dz_column_names(pupil_indices, focal_indices):
    """Generate DZ column names in the order they appear in the data tables.

    The naming convention is ``z1toz6_z{j}_c{k}``.  The natural iteration
    order is **pupil outer, focal inner** — j varies slowly, k varies fast.

    Parameters
    ----------
    pupil_indices : array_like of int
    focal_indices : array_like of int

    Returns
    -------
    list of str
    """
    return [f"z1toz6_z{j}_c{k}" for j in pupil_indices for k in focal_indices]


def columns_to_dz_matrix(column_values, n_focal, n_pupil):
    """Reshape a 1-D vector of DZ values (column order) into a 2-D matrix.

    Column order is (pupil-outer, focal-inner): the natural order from
    ``make_dz_column_names``.  The returned matrix has shape
    (n_focal, n_pupil) to match the sensitivity tensor convention.

    Steps: reshape(n_pupil, n_focal) → .T → (n_focal, n_pupil).

    Parameters
    ----------
    column_values : array_like, shape (n_pupil * n_focal,)
    n_focal, n_pupil : int

    Returns
    -------
    ndarray, shape (n_focal, n_pupil)
    """
    column_values = np.asarray(column_values, dtype=float)
    if column_values.size != n_focal * n_pupil:
        raise ValueError(
            f"Expected {n_focal * n_pupil} values, got {column_values.size}"
        )
    return column_values.reshape(n_pupil, n_focal).T


def dz_matrix_to_flat(dz_matrix):
    """Flatten a (n_focal, n_pupil) DZ matrix to a 1-D vector compatible with A.

    C-order: k varies slowly, j varies fast — matches ``build_design_matrix``.

    Parameters
    ----------
    dz_matrix : ndarray, shape (n_focal, n_pupil)

    Returns
    -------
    ndarray, shape (n_focal * n_pupil,)
    """
    return np.asarray(dz_matrix).reshape(-1)


def flat_to_dz_matrix(flat, n_focal, n_pupil):
    """Reshape a flat vector back to (n_focal, n_pupil).

    Inverse of ``dz_matrix_to_flat``.

    Parameters
    ----------
    flat : array_like, shape (n_focal * n_pupil,)
    n_focal, n_pupil : int

    Returns
    -------
    ndarray, shape (n_focal, n_pupil)
    """
    flat = np.asarray(flat)
    if flat.size != n_focal * n_pupil:
        raise ValueError(
            f"Expected {n_focal * n_pupil} values, got {flat.size}"
        )
    return flat.reshape(n_focal, n_pupil)


def group_by_tolerance(values, tolerance=1.0):
    """Group values that fall within *tolerance* of each other.

    Parameters
    ----------
    values : array_like
    tolerance : float

    Returns
    -------
    list of list of int
        Each inner list contains indices of values in one group.
    """
    values = np.asarray(values)
    used = np.zeros(len(values), dtype=bool)
    groups = []
    for i in range(len(values)):
        if used[i]:
            continue
        close_mask = np.isclose(values, values[i], atol=tolerance, rtol=0)
        groups.append(np.where(close_mask)[0].tolist())
        used[close_mask] = True
    return groups


def assign_groups(values, tolerance=1.0):
    """Return an integer group label for each value.

    Parameters
    ----------
    values : array_like
    tolerance : float

    Returns
    -------
    ndarray of int
    """
    groups = group_by_tolerance(values, tolerance)
    labels = np.zeros(len(values), dtype=int)
    for group_id, indices in enumerate(groups):
        labels[indices] = group_id
    return labels


def median_per_group(table, column_names, group_idx_list, n_focal, n_pupil):
    """Compute median DZ coefficients per group as (n_focal, n_pupil) matrices.

    Parameters
    ----------
    table : table-like
        Must support ``table[col_name][indices]`` indexing.
    column_names : list of str
        DZ column names in (pupil-outer, focal-inner) order.
        Use ``make_dz_column_names()`` to generate these.
    group_idx_list : list of array_like
        Each entry is a list/array of row indices forming one group.
    n_focal, n_pupil : int

    Returns
    -------
    list of ndarray
        One (n_focal, n_pupil) matrix per group.
    """
    result = []
    for group_idxs in group_idx_list:
        medians = [np.median(np.asarray(table[col])[group_idxs])
                   for col in column_names]
        result.append(columns_to_dz_matrix(medians, n_focal, n_pupil))
    return result


# =========================================================================
# Section 3: Least-Squares Solve
# =========================================================================

def solve_dof(A, dz_matrix, rcond=1e-3):
    """Solve A @ x_hat ≈ dz_flat via least-squares.

    Parameters
    ----------
    A : ndarray, shape (n_obs, n_dof)
        Design matrix from ``build_design_matrix``.
    dz_matrix : ndarray, shape (n_focal, n_pupil)
        DZ coefficient matrix (one dataset).
    rcond : float
        Cutoff for small singular values.

    Returns
    -------
    x_hat : ndarray, shape (n_dof,)
    residuals : ndarray  (may be empty if underdetermined)
    rank : int
    singular_values : ndarray
    """
    y = dz_matrix_to_flat(dz_matrix)
    if y.shape[0] != A.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but dz vector has "
            f"{y.shape[0]} elements. Check that focal/pupil indices match."
        )
    return np.linalg.lstsq(A, y, rcond=rcond)


# =========================================================================
# Section 4: Display / Printing
# =========================================================================

def print_dofs(x_hat):
    """Print DOF solution vector in a two-column table layout."""
    x_hat = np.asarray(x_hat)
    label_to_value = dict(zip(DOF_LABELS, x_hat))

    left_groups = [
        ("Camera hexapod", DOF_LABELS[5:10]),
        ("M1M3 bends", DOF_LABELS[10:30]),
    ]
    right_groups = [
        ("M2 hexapod", DOF_LABELS[:5]),
        ("M2 bends", DOF_LABELS[30:50]),
    ]

    left_data = _build_rows(left_groups, label_to_value)
    right_data = _build_rows(right_groups, label_to_value)

    all_rows = [row for _, rows in (left_data + right_data) for row in rows]
    label_w = max(len("DOF"), max(len(r[0]) for r in all_rows))
    value_w = max(len("Value"), 14)
    unit_w = max(len("Unit"), max(len(r[2]) for r in all_rows))

    left_lines = _flatten_blocks(left_data, label_w, value_w, unit_w)
    right_lines = _flatten_blocks(right_data, label_w, value_w, unit_w)

    left_width = max(len(line) for line in left_lines)
    n_lines = max(len(left_lines), len(right_lines))
    left_lines += [""] * (n_lines - len(left_lines))
    right_lines += [""] * (n_lines - len(right_lines))

    gap = "   "
    for l_line, r_line in zip(left_lines, right_lines):
        print(f"{l_line:<{left_width}}{gap}{r_line}")


def _build_rows(group_specs, label_to_value):
    out = []
    for group_name, labels in group_specs:
        rows = []
        for label in labels:
            val = label_to_value[label]
            if label.endswith("_rx") or label.endswith("_ry"):
                display_val = val * 3600.0
                unit = "arcsec"
            else:
                display_val = val
                unit = "um"
            rows.append((label, display_val, unit))
        out.append((group_name, rows))
    return out


def _make_block(group_name, rows, label_w, value_w, unit_w):
    line = f"+-{'-' * label_w}-+-{'-' * value_w}-+-{'-' * unit_w}-+"
    group_line_w = len(line) - 4

    block = []
    block.append(line)
    block.append(f"| {'DOF':<{label_w}} | {'Value':>{value_w}} | {'Unit':<{unit_w}} |")
    block.append(line)
    block.append(f"| {(' ' + group_name + ' '):<{group_line_w}} |")
    block.append(line)
    for label, value, unit in rows:
        block.append(f"| {label:<{label_w}} | {value:>{value_w}.6f} | {unit:<{unit_w}} |")
    block.append(line)
    return block


def _flatten_blocks(grouped_blocks, label_w, value_w, unit_w):
    lines = []
    for i, (group_name, rows) in enumerate(grouped_blocks):
        lines.extend(_make_block(group_name, rows, label_w, value_w, unit_w))
        if i != len(grouped_blocks) - 1:
            lines.append("")
    return lines


def print_residuals(residual_vector, focal_indices, pupil_indices, tolerance=0.01):
    """Print residuals as a 2D table (pupil rows × focal columns).

    Parameters
    ----------
    residual_vector : array_like
        Flattened residual vector, length n_focal * n_pupil.
    focal_indices : list of int
    pupil_indices : list of int
    tolerance : float
        Absolute value threshold for asterisk marking.
    """
    residuals_2d = flat_to_dz_matrix(
        residual_vector, len(focal_indices), len(pupil_indices)
    ).T  # transpose to (pupil, focal) for display

    pupil_label_w = max(len(str(idx)) for idx in pupil_indices)
    pupil_label_w = max(pupil_label_w, len("Pupil"))
    focal_label_w = max(len(str(idx)) for idx in focal_indices)
    focal_label_w = max(focal_label_w, 12)

    header = f"{'j \\\\ k':<{pupil_label_w}}"
    for focal_idx in focal_indices:
        header += f" | {str(focal_idx):>{focal_label_w}}"
    print(header)
    print("-" * len(header))

    for i, pupil_idx in enumerate(pupil_indices):
        row = f"{str(pupil_idx):<{pupil_label_w}}"
        for j in range(len(focal_indices)):
            val = residuals_2d[i, j]
            marker = "*" if abs(val) > tolerance else " "
            row += f" | {val:>{focal_label_w - 2}.6f} {marker}"
        print(row)


# =========================================================================
# Section 5: Plotting — DZ figures
# =========================================================================

def setup_dz_figure(n_focal_zernikes, pupil_indices, n_datasets, fixed_y=False):
    """Create a figure with one subplot per focal Zernike k, pupil j on x-axis.

    Parameters
    ----------
    n_focal_zernikes : int
        Number of focal Zernike subplots.
    pupil_indices : list of int
        Actual pupil Zernike indices (e.g. [4,5,...,19,22,...,26]).
    n_datasets : int

    Returns
    -------
    fig : Figure
    axes : list of Axes
    dataset_width : float
    pupil_positions : dict
        Maps pupil index j → x-axis position.
    """
    n_subplots = n_focal_zernikes
    n_pupil = len(pupil_indices)

    fig_width = max(10, n_pupil * 0.4 + n_datasets * 0.3) + 4
    fig_height = max(4, n_subplots * 2.2)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_width, fig_height),
                             sharex=True, constrained_layout=True)
    if n_subplots == 1:
        axes = [axes]

    stagger_total = 0.95
    dataset_width = stagger_total / n_datasets

    pupil_positions = {j: i for i, j in enumerate(pupil_indices)}

    for i, j in enumerate(pupil_indices):
        if j % 2 == 0:
            for ax in axes:
                ax.axvspan(i - 0.5, i + 0.5, color='k', alpha=0.07, lw=0)

    for k, ax in enumerate(axes):
        ax.set_ylabel(f"k={k + 1}\n[wavefront μm]", fontsize=11)
        if fixed_y:
            ax.set_ylim(-0.18, 0.18)
        ax.axhline(0, color='gray', lw=0.5, ls='-')
        ax.grid(True, axis='y', alpha=0.4)

    axes[-1].set_xticks(range(n_pupil))
    axes[-1].set_xticklabels([str(j) for j in pupil_indices], fontsize=9)
    axes[-1].set_xlim(-0.5, n_pupil - 0.5)
    axes[-1].set_xlabel("Pupil Zernike Index (j)", fontsize=11)

    return fig, axes, dataset_width, pupil_positions


def plot_dz_matrix(axes, dz_matrix, pupil_positions, dataset_idx,
                   dataset_width, color, marker='o'):
    """Plot DZ matrix coefficients on k-j subplots.

    Parameters
    ----------
    axes : list of Axes
    dz_matrix : ndarray, shape (n_focal, n_pupil)
    pupil_positions : dict
        Maps pupil index j → x position.
    dataset_idx : int
    dataset_width : float
    color : color
    marker : str
    """
    dz_matrix = np.asarray(dz_matrix)
    n_focal, n_pupil = dz_matrix.shape

    x_offset = (dataset_idx - 0.5) * dataset_width - 0.25

    for k in range(n_focal):
        x_positions = np.arange(n_pupil) + x_offset
        axes[k].plot(x_positions, dz_matrix[k, :], marker=marker, color=color,
                     linestyle='none', markersize=6)


def finalize_dz_figure(fig, axes, file_keys, dataset_colors,
                       title, output_path, marker_size=6):
    """Add legend, title, and save a DZ k-j subplot figure.

    Parameters
    ----------
    fig : Figure
    axes : list of Axes
    file_keys : list of str
    dataset_colors : sequence of colors
    title : str
    output_path : Path
    marker_size : float
    """
    handles, labels = [], []
    for file_idx, file_key in enumerate(file_keys):
        color = dataset_colors[file_idx]
        handles.append(Line2D([0], [0], color=color, marker='o',
                              linestyle='none', markersize=marker_size))
        labels.append(file_key)
    axes[-1].legend(handles, labels, ncols=1, loc='lower left',
                    bbox_to_anchor=(1.01, 0.), fontsize=11)

    fig.suptitle(title, fontsize=12)

    print(f"  Saving DZ plot to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dz_datasets(dz_matrix_list, pupil_indices, file_keys,
                     dataset_colors, title, output_path, fixed_y=False):
    """Plot multiple DZ coefficient matrices.

    Parameters
    ----------
    dz_matrix_list : list of array_like
        Each array is shape (n_focal, n_pupil).
    pupil_indices : list of int
    file_keys : list of str
    dataset_colors : list of colors
    title : str
    output_path : Path
    """
    n_datasets = len(dz_matrix_list)
    n_focal = dz_matrix_list[0].shape[0]

    fig, axes, dataset_width, pupil_positions = setup_dz_figure(
        n_focal, pupil_indices, n_datasets, fixed_y)

    for dataset_idx, (dz_matrix, color) in enumerate(
            zip(dz_matrix_list, dataset_colors)):
        plot_dz_matrix(axes, dz_matrix, pupil_positions, dataset_idx,
                       dataset_width, color)

    finalize_dz_figure(fig, axes, file_keys, dataset_colors, title, output_path)


# =========================================================================
# Section 6: Plotting — DOF figures
# =========================================================================

def setup_dof_figure(n_datasets):
    """Create a figure with one subplot per DOF group.

    Returns
    -------
    fig : Figure
    axes : dict  (keys: 'xyz', 'rxry', 'm1m3_bends', 'm2_bends')
    dataset_width : float
    """
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.93)

    axes = {
        'xyz': fig.add_subplot(gs[0, 0]),
        'rxry': fig.add_subplot(gs[0, 1]),
        'm1m3_bends': fig.add_subplot(gs[1, :]),
        'm2_bends': fig.add_subplot(gs[2, :]),
    }

    stagger_total = 0.8
    dataset_width = stagger_total / n_datasets

    # XYZ
    xyz_labels = ['Cam_z', 'Cam_x', 'Cam_y', 'M2_z', 'M2_x', 'M2_y']
    axes['xyz'].set_xlim(-0.5, len(xyz_labels)-0.5)
    axes['xyz'].set_xticks(range(len(xyz_labels)))
    axes['xyz'].set_xticklabels(xyz_labels, rotation=45, ha='right')
    axes['xyz'].set_ylabel('DOF Value (μm)', fontsize=10)
    axes['xyz'].set_title('Hexapod Translations', fontsize=11)
    axes['xyz'].axhline(0, color='gray', lw=0.5, ls='-')
    axes['xyz'].grid(True, axis='y', alpha=0.4)
    for i in range(0, len(xyz_labels), 2):
        axes['xyz'].axvspan(i - 0.5, i + 0.5, color='k', alpha=0.07, lw=0)

    # RxRy
    rxry_labels = ['Cam_rx', 'Cam_ry', 'M2_rx', 'M2_ry']
    axes['rxry'].set_xlim(-0.5, len(rxry_labels)-0.5)
    axes['rxry'].set_xticks(range(len(rxry_labels)))
    axes['rxry'].set_xticklabels(rxry_labels, rotation=45, ha='right')
    axes['rxry'].set_ylabel('DOF Value (arcsec)', fontsize=10)
    axes['rxry'].set_title('Hexapod Rotations', fontsize=11)
    axes['rxry'].axhline(0, color='gray', lw=0.5, ls='-')
    axes['rxry'].grid(True, axis='y', alpha=0.4)
    for i in range(0, len(rxry_labels), 2):
        axes['rxry'].axvspan(i - 0.5, i + 0.5, color='k', alpha=0.07, lw=0)

    # M1M3 bending modes
    axes['m1m3_bends'].set_xticks(range(20))
    axes['m1m3_bends'].set_xlim(-0.5, 20.-0.5)
    axes['m1m3_bends'].set_xticklabels([str(i + 1) for i in range(20)])
    axes['m1m3_bends'].set_ylabel('DOF Value (μm)', fontsize=10)
    axes['m1m3_bends'].set_title('M1M3 Bending Modes', fontsize=11)
    axes['m1m3_bends'].axhline(0, color='gray', lw=0.5, ls='-')
    axes['m1m3_bends'].grid(True, axis='y', alpha=0.4)
    for i in range(0, 20, 2):
        axes['m1m3_bends'].axvspan(i - 0.5, i + 0.5, color='k', alpha=0.07, lw=0)

    # M2 bending modes
    axes['m2_bends'].set_xticks(range(20))
    axes['m2_bends'].set_xlim(-0.5, 20.-0.5)
    axes['m2_bends'].set_xticklabels([str(i + 1) for i in range(20)])
    axes['m2_bends'].set_xlabel('Bending Mode', fontsize=10)
    axes['m2_bends'].set_ylabel('DOF Value (μm)', fontsize=10)
    axes['m2_bends'].set_title('M2 Bending Modes', fontsize=11)
    axes['m2_bends'].axhline(0, color='gray', lw=0.5, ls='-')
    axes['m2_bends'].grid(True, axis='y', alpha=0.4)
    for i in range(0, 20, 2):
        axes['m2_bends'].axvspan(i - 0.5, i + 0.5, color='k', alpha=0.07, lw=0)

    return fig, axes, dataset_width


def plot_dof_vector(ax, x_positions, values, dataset_idx, dataset_width,
                    color, marker='o'):
    """Plot DOF values at staggered x positions.

    Parameters
    ----------
    ax : Axes
    x_positions : array_like
    values : array_like
    dataset_idx : int
    dataset_width : float
    color : color
    marker : str
    """
    x_offset = (dataset_idx - 0.5) * dataset_width - 0.25
    x_plot = np.array(x_positions) + x_offset
    ax.plot(x_plot, values, marker=marker, color=color,
            linestyle='none', markersize=6)


def plot_dof_datasets(x_hat_list, file_keys, dataset_colors, title, output_path):
    """Plot multiple DOF solution vectors.

    Parameters
    ----------
    x_hat_list : list of array_like
        Each array is a 50-element DOF vector.
        Order: [M2_hex(5), Cam_hex(5), M1M3_bends(20), M2_bends(20)].
    file_keys : list of str
    dataset_colors : list of colors
    title : str
    output_path : Path
    """
    n_datasets = len(x_hat_list)
    fig, axes, dataset_width = setup_dof_figure(n_datasets)

    for dataset_idx, (x_hat, color) in enumerate(zip(x_hat_list, dataset_colors)):
        m2_hex = x_hat[0:5]
        cam_hex = x_hat[5:10]
        m1m3_bends = x_hat[10:30]
        m2_bends = x_hat[30:50]

        xyz_values = np.concatenate([cam_hex[0:3], m2_hex[0:3]])
        plot_dof_vector(axes['xyz'], range(6), xyz_values,
                        dataset_idx, dataset_width, color)

        rxry_values = np.concatenate([cam_hex[3:5], m2_hex[3:5]]) * 3600
        plot_dof_vector(axes['rxry'], range(4), rxry_values,
                        dataset_idx, dataset_width, color)

        plot_dof_vector(axes['m1m3_bends'], range(20), m1m3_bends,
                        dataset_idx, dataset_width, color)
        plot_dof_vector(axes['m2_bends'], range(20), m2_bends,
                        dataset_idx, dataset_width, color)

    finalize_dof_figure(fig, axes, file_keys, dataset_colors, title, output_path)


def finalize_dof_figure(fig, axes, file_keys, dataset_colors,
                        title, output_path, marker_size=6):
    """Add legend, title, and save DOF figure.

    Parameters
    ----------
    fig : Figure
    axes : dict of Axes
    file_keys : list of str
    dataset_colors : sequence of colors
    title : str
    output_path : Path
    marker_size : float
    """
    handles, labels = [], []
    for file_idx, file_key in enumerate(file_keys):
        color = dataset_colors[file_idx]
        handles.append(Line2D([0], [0], color=color, marker='o',
                              linestyle='none', markersize=marker_size))
        labels.append(file_key)
    axes['m2_bends'].legend(handles, labels, ncols=1, loc='upper right',
                            fontsize=9)

    fig.suptitle(title, fontsize=12)

    print(f"  Saving DOF plot to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Section 7: Plotting — Sensitivity matrix heatmaps
# =========================================================================

def plot_sensitivity_matrix_layer(sensitivity_layer, pupil_indices, k_index,
                                  output_path):
    """Plot one focal Zernike layer of the sensitivity matrix as a heatmap.

    Plots the full (unsliced) layer with all pupil rows.  Rows that are
    in ``pupil_indices`` are labelled normally; rows that are excluded
    are greyed out to show what gets cut.

    Parameters
    ----------
    sensitivity_layer : ndarray, shape (J_full, n_dof)
        Full (unsliced) sensitivity layer for one focal Zernike.
    pupil_indices : list of int
        Pupil Zernike indices that are used in the analysis.
    k_index : int
    output_path : Path
    """
    sensitivity_layer = np.asarray(sensitivity_layer)
    n_pupil_full, n_dof = sensitivity_layer.shape
    selected = set(pupil_indices)

    m2_xyz = sensitivity_layer[:, 0:3]
    m2_rxy = sensitivity_layer[:, 3:5] / 3600.0
    cam_xyz = sensitivity_layer[:, 5:8]
    cam_rxy = sensitivity_layer[:, 8:10] / 3600.0
    m1m3_bends = sensitivity_layer[:, 10:30]
    m2_bends = sensitivity_layer[:, 30:50]

    fig, (ax_hex, ax_bends) = plt.subplots(
        1, 2, figsize=(16, max(6, n_pupil_full * 0.3)),
        gridspec_kw={'width_ratios': [1, 3]}
    )

    # Hexapods
    hexapod_data = np.concatenate([m2_xyz, m2_rxy, cam_xyz, cam_rxy], axis=1)
    abs_max = np.max(np.abs(hexapod_data))
    im_hex = ax_hex.imshow(hexapod_data, aspect='auto', cmap='RdBu_r',
                           vmin=-abs_max, vmax=abs_max, interpolation='nearest')
    hex_labels = ['z', 'x', 'y', r'$r_x$', r'$r_y$',
                  'z', 'x', 'y', r'$r_x$', r'$r_y$']
    ax_hex.set_xticks(range(10))
    ax_hex.set_xticklabels(hex_labels, fontsize=10)
    ax_hex.set_xlabel('M2          Camera', fontsize=11)
    ax_hex.axvline(4.5, color='gray', lw=1.5, ls='--')

    # Bending modes
    bends_data = np.concatenate([m1m3_bends, m2_bends], axis=1)
    abs_max = np.max(np.abs(bends_data))
    im_bends = ax_bends.imshow(bends_data, aspect='auto', cmap='RdBu_r',
                               vmin=-abs_max, vmax=abs_max, interpolation='nearest')
    bend_labels = ([f'$B_{{1,{i}}}$' for i in range(1, 21)]
                   + [f'$B_{{2,{i}}}$' for i in range(1, 21)])
    tick_positions = list(range(0, 40, 5))
    ax_bends.set_xticks(tick_positions)
    ax_bends.set_xticklabels([bend_labels[i] for i in tick_positions], fontsize=10)
    ax_bends.set_xlabel('M1M3                                        M2', fontsize=11)
    ax_bends.axvline(19.5, color='gray', lw=1.5, ls='--')

    # Y-axis: label every pupil row, grey out excluded ones
    y_labels = []
    y_colors = []
    for j in range(n_pupil_full):
        y_labels.append(f'$Z_{{{j}}}$')
        y_colors.append('black' if j in selected else 'lightgray')

    for ax in (ax_hex, ax_bends):
        ax.set_yticks(range(n_pupil_full))
        ax.set_yticklabels(y_labels, fontsize=8)
        for tick_label, color in zip(ax.get_yticklabels(), y_colors):
            tick_label.set_color(color)

    plt.colorbar(im_hex, ax=ax_hex, fraction=0.046, pad=0.04).set_label(
        r'$\mu m$ or $\mu m$/arcsec', fontsize=10)
    plt.colorbar(im_bends, ax=ax_bends, fraction=0.046, pad=0.04).set_label(
        r'$\mu m$ or $\mu m$/arcsec', fontsize=10)

    fig.suptitle(f'Sensitivity Matrix for Focal Zernike $k={{k_index}}$',
                 fontsize=13, y=0.98)
    plt.tight_layout()
    print(f"  Saving sensitivity matrix plot to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_sensitivity_layers(sensitivity_matrix, pupil_indices, n_focal,
                                output_dir, version):
    """Plot all focal Zernike layers of the sensitivity matrix.

    Parameters
    ----------
    sensitivity_matrix : ndarray, shape (n_focal, n_pupil, n_dof)
    pupil_indices : list of int
    n_focal : int
    output_dir : Path
    """
    output_dir = Path(output_dir)
    for k in range(1, n_focal):
        output_path = output_dir / f'sensitivity_k{k}{version}.png'
        plot_sensitivity_matrix_layer(sensitivity_matrix[k], pupil_indices,
                                     k, output_path)
