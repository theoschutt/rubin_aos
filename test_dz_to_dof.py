"""Tests for dz_to_dof module."""

import numpy as np
import pytest

from dz_to_dof import (
    build_design_matrix,
    columns_to_dz_matrix,
    dz_matrix_to_flat,
    flat_to_dz_matrix,
    group_by_tolerance,
    make_dz_column_names,
    solve_dof,
    DOF_LABELS,
    N_DOF,
)


def test_design_matrix_indexing():
    """A[k * n_pupil + j, dof] == smatrix[k, j, dof] for all k, j, dof."""
    n_focal, n_pupil, n_dof = 4, 7, 5
    smatrix = np.arange(n_focal * n_pupil * n_dof).reshape(n_focal, n_pupil, n_dof)
    A = build_design_matrix(smatrix)
    assert A.shape == (n_focal * n_pupil, n_dof)
    for k in range(n_focal):
        for j in range(n_pupil):
            np.testing.assert_array_equal(A[k * n_pupil + j], smatrix[k, j])


def test_columns_to_dz_matrix():
    """Column order (j outer, k inner) maps to matrix[k_idx, j_idx]."""
    n_focal, n_pupil = 3, 4
    pupil_indices = [4, 5, 6, 7]
    focal_indices = [1, 2, 3]
    col_names = make_dz_column_names(pupil_indices, focal_indices)

    # Verify column name ordering is j-outer, k-inner
    assert col_names[0] == "z1toz6_z4_c1"   # j=4, k=1
    assert col_names[1] == "z1toz6_z4_c2"   # j=4, k=2
    assert col_names[2] == "z1toz6_z4_c3"   # j=4, k=3
    assert col_names[3] == "z1toz6_z5_c1"   # j=5, k=1

    values = np.arange(n_focal * n_pupil, dtype=float)
    matrix = columns_to_dz_matrix(values, n_focal, n_pupil)
    assert matrix.shape == (n_focal, n_pupil)

    # value at column index (j_idx * n_focal + k_idx) should appear at matrix[k_idx, j_idx]
    for j_idx in range(n_pupil):
        for k_idx in range(n_focal):
            col_index = j_idx * n_focal + k_idx
            assert matrix[k_idx, j_idx] == values[col_index], (
                f"matrix[{k_idx}, {j_idx}] = {matrix[k_idx, j_idx]} "
                f"!= values[{col_index}] = {values[col_index]}"
            )


def test_round_trip_solve():
    """Known x_true -> A @ x_true = y -> lstsq(A, y) recovers x_true."""
    n_focal, n_pupil, n_dof = 6, 21, 50
    rng = np.random.default_rng(42)
    smatrix = rng.standard_normal((n_focal, n_pupil, n_dof))
    A = build_design_matrix(smatrix)
    x_true = rng.standard_normal(n_dof)
    y_flat = A @ x_true
    dz_matrix = flat_to_dz_matrix(y_flat, n_focal, n_pupil)
    x_hat, _, _, _ = solve_dof(A, dz_matrix, rcond=None)
    np.testing.assert_allclose(x_hat, x_true, atol=1e-10)


def test_flat_matrix_roundtrip():
    """flat_to_dz_matrix(dz_matrix_to_flat(m)) == m."""
    n_focal, n_pupil = 6, 21
    mat = np.arange(n_focal * n_pupil, dtype=float).reshape(n_focal, n_pupil)
    flat = dz_matrix_to_flat(mat)
    recovered = flat_to_dz_matrix(flat, n_focal, n_pupil)
    np.testing.assert_array_equal(mat, recovered)


def test_solve_dof_shape_mismatch():
    """solve_dof raises ValueError when A rows != dz_matrix elements."""
    A = np.zeros((126, 50))
    dz_wrong_shape = np.zeros((5, 21))  # 105 != 126
    with pytest.raises(ValueError, match="Shape mismatch"):
        solve_dof(A, dz_wrong_shape)


def test_columns_to_dz_matrix_wrong_size():
    """columns_to_dz_matrix raises ValueError on wrong input size."""
    with pytest.raises(ValueError, match="Expected 12 values"):
        columns_to_dz_matrix(np.zeros(10), n_focal=3, n_pupil=4)


def test_group_by_tolerance():
    """Values within tolerance are grouped together."""
    values = [0.0, 0.5, 10.0, 10.3, 20.0]
    groups = group_by_tolerance(values, tolerance=1.0)
    assert len(groups) == 3
    assert set(groups[0]) == {0, 1}
    assert set(groups[1]) == {2, 3}
    assert set(groups[2]) == {4}


def test_dof_labels():
    """DOF_LABELS has the right length and expected entries."""
    assert len(DOF_LABELS) == N_DOF == 50
    assert DOF_LABELS[0] == "M2_hex_z"
    assert DOF_LABELS[9] == "Cam_hex_ry"
    assert DOF_LABELS[10] == "M1M3_B1"
    assert DOF_LABELS[30] == "M2_B1"
    assert DOF_LABELS[49] == "M2_B20"
