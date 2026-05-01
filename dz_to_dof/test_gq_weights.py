"""Most of this code is from
ts/ofc/scripts/generate_normalization_weights.npy
"""

import numpy as np
from lsst.ts.ofc import OFCData, SensitivityMatrix
from lsst.ts.wep.utils import convertZernikesToPsfWidth
import yaml

from dz_to_dof import (
    DOF_LABELS,
    DZtoDOFSolver,
    load_ofc_data,
    get_rf_weights
)

def make_quadrature_grid(
    rings: int,
    spokes: int,
    field_radius: float,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Create a quadrature grid on a circular field.

    Parameters
    ----------
    rings : `int`
        Number of radial quadrature rings.
    spokes : `int`
        Number of azimuthal spokes.
    field_radius : `float`
        Field radius in degrees.

    Returns
    -------
    field_angles : `list` [`tuple` [`float`, `float`]]
        List of (x, y) field angles in degrees.
    field_weights : `np.ndarray`
        Quadrature weights for each field point, normalized to sum to 1.
    """
    li, w_ring = np.polynomial.legendre.leggauss(rings)
    radii = np.sqrt((1.0 + li) / 2.0) * field_radius
    w_ring = w_ring * np.pi / (2.0 * spokes)

    azs = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)
    radii, azs = np.meshgrid(radii, azs, indexing="ij")

    x = (radii * np.cos(azs)).ravel()
    y = (radii * np.sin(azs)).ravel()
    field_angles = [(float(xx), float(yy)) for xx, yy in zip(x, y)]

    field_weights = np.broadcast_to(w_ring[:, np.newaxis], radii.shape).ravel()
    field_weights = field_weights / np.sum(field_weights)

    return field_angles, field_weights


def compute_fwhm_matrix_per_field(
    ofc_data: OFCData,
    dz_sensitivity_matrix: SensitivityMatrix,
    field_angles: list,
) -> np.ndarray:
    """Compute the FWHM response aggregated per field point.
    Parameters
    ----------
    fwhm_matrix : np.ndarray
        FWHM matrix.
    ofc_data : `OFCData`
        OFC data object.
    dz_sensitivity_matrix : `SensitivityMatrix`
        Double-Zernike sensitivity matrix object.
    field_angles : `list`
        Field angles at which to evaluate the matrix.

    Returns
    -------
       fwhm_per_field : `np.ndarray`
        FWHM response per field point, with shape (n_field, n_dof_used).
    """
    sensitivity_matrix = dz_sensitivity_matrix.evaluate(field_angles, rotation_angle=0.0)

    sensitivity_matrix = sensitivity_matrix[:, dz_sensitivity_matrix.ofc_data.zn_idx, :]

    fwhm_matrix = np.zeros(sensitivity_matrix.shape)
    for idy in range(sensitivity_matrix.shape[0]):
        fwhm_matrix[idy, ...] = convertZernikesToPsfWidth(sensitivity_matrix[idy, ...].T).T

    fwhm_matrix = fwhm_matrix[..., ofc_data.dof_idx]
    fwhm_per_field = np.sqrt(np.sum(fwhm_matrix**2, axis=1))

    return fwhm_per_field


def compute_fwhm_weights_quadrature(
    fwhm_per_field: np.ndarray,
    field_weights: np.ndarray,
) -> np.ndarray:
    """Compute quadrature-weighted FWHM weights.

    Parameters
    ----------
    fwhm_per_field : `np.ndarray`
        FWHM response per field point, shape (n_field, n_dof_used).
    field_weights : `np.ndarray`
        Normalized field weights, shape (n_field,).

    Returns
    -------
    fwhm_weights : `np.ndarray`
        Weighted FWHM weights.
    """
    return np.sqrt(np.sum(field_weights[:, None] * fwhm_per_field**2, axis=0))


def generate_fwhm_weights() -> None:
    """Generate normalization weights for the sensitivity matrix."""

    args = {
        "instrument": "lsst",
        "method": "quadrature",
        "rings": 5,
        "spokes": 6,
        "field_radius": 1.75,
    }

    ofc_data = OFCData(args["instrument"])
    dz_sensitivity_matrix = SensitivityMatrix(ofc_data)

    field_angles, field_weights = make_quadrature_grid(
        rings=args["rings"],
        spokes=args["spokes"],
        field_radius=args["field_radius"],
    )

    fwhm_per_field = compute_fwhm_matrix_per_field(
        ofc_data, dz_sensitivity_matrix, field_angles
    )
    fwhm_weights = compute_fwhm_weights_quadrature(
        fwhm_per_field, field_weights
    )

    np.save("gq_fwhm_weights.npy", fwhm_weights)


def test_weights():
    """Reproduce/check official ``geom_gq``-style normalization weights
    (``range0.5_fwhm-0.15.yaml``)."""
    DEFAULT_PUPIL_INDICES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
    DEFAULT_FOCAL_INDICES = [1, 2, 3, 4, 5, 6]
    dof_indices = range(50)
    ofc_data = load_ofc_data()
    solver = DZtoDOFSolver(
        ofc_data,
        DEFAULT_PUPIL_INDICES,
        DEFAULT_FOCAL_INDICES,
        dof_indices=dof_indices
    )

    r_i, f_i, _ = get_rf_weights(
        ofc_data, solver.full_coef, dof_indices
    )

    my_weights = np.sqrt(r_i / f_i)

    with open("input_data/range0.5_fwhm-0.15.yaml") as f:
        yy = yaml.load(f, yaml.CLoader)
    gmh_weights = np.array(yy)

    gq_fwhm_weights = np.load("gq_fwhm_weights.npy")

    my_weights_w_gq = my_weights * np.sqrt(f_i / gq_fwhm_weights)
    print_arr(my_weights_w_gq / gmh_weights)

def print_arr(arr):
    for dof, w in zip(DOF_LABELS, arr):
        print(f"{dof:>10}", w)

if __name__ == "__main__":
    generate_fwhm_weights()
    test_weights()