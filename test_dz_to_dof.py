"""Tests for dz_to_dof module."""

import numpy as np
import pytest

from dz_to_dof import (
    DZtoDOFSolver,
    build_design_matrix,
    columns_to_dz_matrix,
    dz_matrix_to_flat,
    flat_to_dz_matrix,
    group_by_tolerance,
    make_dz_column_names,
    renormalize_sensitivity_matrix,
    reverse_normalization,
    pad_ofc_array,
    load_weights_yaml,
    solve_dof,
    DOF_LABELS,
    N_DOF,
    N_HEX,
    N_M1M3_BEND,
    N_M2_BEND,
    IDX_M1M3_START,
    IDX_M2_START,
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
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
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
    A = np.zeros((126, N_DOF))
    dz_wrong_shape = np.zeros((5, 21))  # 105 != 126
    with pytest.raises(ValueError, match="Shape mismatch"):
        solve_dof(A, dz_wrong_shape, rcond=1e-4)


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


def test_group_by_tolerance_chain():
    """Single-linkage: a chain of values each
    within tolerance of neighbours ends up as
    one group, regardless of input order.
    """
    # -45.3, -44.9, -44.5: adjacent gaps 0.4 each
    values = [-44.5, -45.3, -44.9]
    groups = group_by_tolerance(values, tolerance=1.0)
    assert len(groups) == 1
    assert set(groups[0]) == {0, 1, 2}


def test_group_by_tolerance_order_invariant():
    """Grouping is invariant to input order."""
    values = [-45.2, -44.8, -45.6, 14.8, 15.2]
    g1 = group_by_tolerance(values, tolerance=1.0)
    values_shuffled = [values[i]
                       for i in [3, 0, 4, 2, 1]]
    g2 = group_by_tolerance(
        values_shuffled, tolerance=1.0)
    # Same set of value-groups (via values at
    # the grouped indices) regardless of input
    # order
    by_val = lambda vs, gs: sorted(
        tuple(sorted(vs[i] for i in g))
        for g in gs)
    assert (by_val(values, g1)
            == by_val(values_shuffled, g2))


def test_dof_labels():
    """DOF_LABELS has the right length and expected entries."""
    assert len(DOF_LABELS) == N_DOF == 51
    assert DOF_LABELS[0] == "M2_hex_z"
    assert DOF_LABELS[9] == "Cam_hex_ry"
    assert DOF_LABELS[10] == "M1M3_B1"
    assert DOF_LABELS[30] == "M1M3_B52"
    assert DOF_LABELS[31] == "M2_B1"
    assert DOF_LABELS[50] == "M2_B20"


def test_dof_structure():
    """DOF boundary constants match DOF_LABELS."""
    assert N_HEX + N_M1M3_BEND + N_M2_BEND == N_DOF
    assert IDX_M1M3_START == N_HEX
    assert IDX_M2_START == N_HEX + N_M1M3_BEND
    assert DOF_LABELS[IDX_M1M3_START].startswith(
        "M1M3_B")
    assert DOF_LABELS[IDX_M2_START].startswith(
        "M2_B")


@pytest.fixture(scope="module")
def ofc_data():
    """Load real OFCData once for all renorm tests."""
    from dz_to_dof import load_ofc_data
    return load_ofc_data()


def test_renorm_orig_roundtrip(ofc_data):
    """renormalize then reverse with 'orig' recovers
    the same physical DOFs as solving unnormalized."""
    n_focal, n_pupil = 6, 21
    focal_indices = list(range(1, 7))
    pupil_indices = ([4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                      14, 15, 16, 17, 18, 19,
                      22, 23, 24, 25, 26])

    from dz_to_dof import (
        load_sensitivity_matrix,
    )
    # Load with and without normalization
    sliced_orig, full, _ = load_sensitivity_matrix(
        ofc_data, focal_indices, pupil_indices,
        norm_type=None)
    sliced_renorm, _, _ = load_sensitivity_matrix(
        ofc_data, focal_indices, pupil_indices,
        norm_type="orig")

    A_orig = build_design_matrix(sliced_orig)
    A_renorm = build_design_matrix(sliced_renorm)

    # Synthesize DZ data from known physical DOFs
    # (B52 column is zero in OFC smatrix, so
    # x_true[B52] is not recoverable; set it to 0)
    rng = np.random.default_rng(42)
    x_true = rng.standard_normal(N_DOF)
    x_true[IDX_M1M3_START + N_M1M3_BEND - 1] = 0.0
    y = A_orig @ x_true
    dz_mat = flat_to_dz_matrix(y, n_focal, n_pupil)

    # Solve unnormalized
    x_direct, _, _, _ = solve_dof(A_orig, dz_mat, rcond=None)

    # Solve normalized then reverse
    x_renorm, _, _, _ = solve_dof(A_renorm, dz_mat, rcond=None)
    x_recovered = reverse_normalization(
        ofc_data, x_renorm, "orig")

    # 'orig' weights span ~5 decades (68 to 0.001),
    # so the two solve paths diverge at ~1e-7.
    np.testing.assert_allclose(
        x_recovered, x_direct, atol=1e-6)
    np.testing.assert_allclose(
        x_recovered, x_true, atol=1e-6)


def test_renorm_geom_roundtrip(ofc_data):
    """renormalize then reverse with 'geom' recovers
    the same physical DOFs as solving unnormalized."""
    n_focal, n_pupil = 6, 21
    focal_indices = list(range(1, 7))
    pupil_indices = ([4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                      14, 15, 16, 17, 18, 19,
                      22, 23, 24, 25, 26])

    from dz_to_dof import (
        load_sensitivity_matrix,
    )
    # Load with and without normalization
    sliced_orig, full, _ = load_sensitivity_matrix(
        ofc_data, focal_indices, pupil_indices,
        norm_type=None)
    sliced_renorm, _, _ = load_sensitivity_matrix(
        ofc_data, focal_indices, pupil_indices,
        norm_type="geom")

    A_orig = build_design_matrix(sliced_orig)
    A_renorm = build_design_matrix(sliced_renorm)

    # Synthesize DZ data from known physical DOFs
    # (B52 column is zero in OFC smatrix, so
    # x_true[B52] is not recoverable; set it to 0)
    rng = np.random.default_rng(42)
    x_true = rng.standard_normal(N_DOF)
    x_true[IDX_M1M3_START + N_M1M3_BEND - 1] = 0.0
    y = A_orig @ x_true
    dz_mat = flat_to_dz_matrix(y, n_focal, n_pupil)

    # Solve unnormalized
    x_direct, _, _, _ = solve_dof(A_orig, dz_mat, rcond=None)

    # Solve normalized then reverse
    x_renorm, _, _, _ = solve_dof(A_renorm, dz_mat, rcond=None)
    x_recovered = reverse_normalization(
        ofc_data, x_renorm, "geom", full)

    np.testing.assert_allclose(
        x_recovered, x_direct, atol=1e-10)
    np.testing.assert_allclose(
        x_recovered, x_true, atol=1e-10)


def test_renorm_none_is_identity(ofc_data):
    """norm_type=None leaves smatrix unchanged."""
    focal_indices = list(range(1, 7))
    pupil_indices = [4, 5, 6, 7, 8, 9]

    from dz_to_dof import load_sensitivity_matrix
    sliced_none, full, _ = load_sensitivity_matrix(
        ofc_data, focal_indices, pupil_indices,
        norm_type=None)

    renormed = renormalize_sensitivity_matrix(
        ofc_data, full, None)
    np.testing.assert_array_equal(full, renormed)

    x = np.ones(N_DOF)
    x_back = reverse_normalization(
        ofc_data, x, None)
    np.testing.assert_array_equal(x, x_back)


# ---- DZtoDOFSolver tests (synthetic data) ----

def test_solver_roundtrip():
    """Solver recovers known DOFs with all 50."""
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
    rng = np.random.default_rng(99)
    smatrix = rng.standard_normal(
        (n_focal, n_pupil, n_dof))
    A = build_design_matrix(smatrix)
    solver = DZtoDOFSolver._from_components(
        A, n_focal, n_pupil)

    x_true = rng.standard_normal(n_dof)
    y = A @ x_true
    dz_mat = flat_to_dz_matrix(
        y, n_focal, n_pupil)

    result = solver.solve(dz_mat)
    np.testing.assert_allclose(
        result["x_hat"], x_true, atol=1e-10)


def test_solver_dof_subset():
    """Subset solver recovers signal that lives
    entirely in the subset."""
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
    rng = np.random.default_rng(77)
    smatrix = rng.standard_normal(
        (n_focal, n_pupil, n_dof))
    A_full = build_design_matrix(smatrix)

    # Signal only in DOFs 0-9 (hexapods)
    subset = list(range(10))
    x_true = np.zeros(n_dof)
    x_true[subset] = rng.standard_normal(
        len(subset))

    y = A_full @ x_true
    dz_mat = flat_to_dz_matrix(
        y, n_focal, n_pupil)

    A_sub = build_design_matrix(
        smatrix[:, :, subset])
    solver = DZtoDOFSolver._from_components(
        A_sub, n_focal, n_pupil,
        dof_indices=subset)

    result = solver.solve(dz_mat)
    # Subset DOFs recovered exactly
    np.testing.assert_allclose(
        result["x_hat"][subset],
        x_true[subset], atol=1e-10)
    # Excluded DOFs are zero
    excluded = list(range(10, N_DOF))
    np.testing.assert_array_equal(
        result["x_hat"][excluded], 0.0)


def test_solver_residual_identity():
    """reconstructed + residual == input."""
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
    rng = np.random.default_rng(55)
    smatrix = rng.standard_normal(
        (n_focal, n_pupil, n_dof))
    A = build_design_matrix(smatrix)
    solver = DZtoDOFSolver._from_components(
        A, n_focal, n_pupil)

    dz_mat = rng.standard_normal(
        (n_focal, n_pupil))
    result = solver.solve(dz_mat)

    recovered = (result["dz_reconstructed"]
                 + result["dz_residual"])
    np.testing.assert_allclose(
        recovered, dz_mat, atol=1e-14)


def test_svd_caching():
    """svd() returns the same cached objects."""
    rng = np.random.default_rng(88)
    A = rng.standard_normal((20, 5))
    solver = DZtoDOFSolver._from_components(
        A, 4, 5)
    r1 = solver.svd()
    r2 = solver.svd()
    assert r1[0] is r2[0]
    assert r1[1] is r2[1]
    assert r1[2] is r2[2]


def test_effective_rank():
    """effective_rank matches expected count."""
    # Build a rank-3 matrix (5 cols, only 3
    # have signal)
    rng = np.random.default_rng(99)
    base = rng.standard_normal((20, 3))
    A = np.zeros((20, 5))
    A[:, :3] = base
    solver = DZtoDOFSolver._from_components(
        A, 4, 5, rcond=1e-10)
    assert solver.effective_rank == 3


def test_svd_reconstruction():
    """U @ diag(s) @ Vt reconstructs A."""
    rng = np.random.default_rng(101)
    A = rng.standard_normal((20, 8))
    solver = DZtoDOFSolver._from_components(
        A, 4, 5)
    U, s, Vt = solver.svd()
    reconstructed = U @ np.diag(s) @ Vt
    np.testing.assert_allclose(
        reconstructed, A, atol=1e-12)


def test_pad_ofc_array():
    """pad_ofc_array inserts a placeholder at
    the B52 slot and is a no-op if already N_DOF
    long."""
    b52_idx = IDX_M1M3_START + N_M1M3_BEND - 1
    # 50-element OFC-style array
    src = np.arange(N_DOF - 1, dtype=float)
    padded = pad_ofc_array(src, fill_value=1.0)
    assert len(padded) == N_DOF
    assert padded[b52_idx] == 1.0
    # Values before and after are unchanged
    np.testing.assert_array_equal(
        padded[:b52_idx], src[:b52_idx])
    np.testing.assert_array_equal(
        padded[b52_idx + 1:], src[b52_idx:])
    # Already-N_DOF arrays pass through
    already = np.arange(N_DOF, dtype=float)
    np.testing.assert_array_equal(
        pad_ofc_array(already), already)


def test_solver_rank_truncation():
    """Rank-k solver solution matches the explicit
    truncated pseudoinverse of A applied to y."""
    rng = np.random.default_rng(42)
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
    smatrix = rng.standard_normal(
        (n_focal, n_pupil, n_dof))
    A = build_design_matrix(smatrix)
    k = 20
    solver = DZtoDOFSolver._from_components(
        A, n_focal, n_pupil, rank=k)

    dz_mat = rng.standard_normal(
        (n_focal, n_pupil))
    result = solver.solve(dz_mat)

    # Expected: x = V[:,:k] @ diag(1/s[:k])
    #              @ U[:,:k]^T @ y
    U, s, Vt = np.linalg.svd(
        A, full_matrices=False)
    y = dz_matrix_to_flat(dz_mat)
    expected = (
        Vt[:k].T
        @ ((U[:, :k].T @ y) / s[:k]))

    np.testing.assert_allclose(
        result["x_hat"], expected, atol=1e-10)
    assert result["rank"] == k


def test_solver_rank_equals_full():
    """rank = min(m, n) gives the same solution as
    the rcond-based solve at tight rcond."""
    rng = np.random.default_rng(7)
    n_focal, n_pupil, n_dof = 6, 21, N_DOF
    smatrix = rng.standard_normal(
        (n_focal, n_pupil, n_dof))
    A = build_design_matrix(smatrix)
    k_full = min(A.shape)

    dz_mat = rng.standard_normal(
        (n_focal, n_pupil))

    # rcond-based solve with very tight rcond
    s_rcond = DZtoDOFSolver._from_components(
        A, n_focal, n_pupil, rcond=1e-12)
    r_rcond = s_rcond.solve(dz_mat)

    # rank-based solve with full rank
    s_rank = DZtoDOFSolver._from_components(
        A, n_focal, n_pupil, rank=k_full)
    r_rank = s_rank.solve(dz_mat)

    np.testing.assert_allclose(
        r_rank["x_hat"], r_rcond["x_hat"],
        atol=1e-8)


def test_solver_rank_caps_at_min_mn():
    """Requesting rank > min(m, n) is silently
    capped to min(m, n)."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 10))
    k_max = min(A.shape)
    solver = DZtoDOFSolver._from_components(
        A, 6, 5, rank=k_max + 50)
    dz_mat = rng.standard_normal((6, 5))
    result = solver.solve(dz_mat)
    assert result["rank"] == k_max


def test_load_weights_yaml_flat_list(tmp_path):
    """Flat-list YAML is loaded and padded."""
    import yaml
    # 50 weights → auto-padded to N_DOF
    weights = [float(i + 1) for i in range(50)]
    p = tmp_path / "w.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(weights, f)

    loaded = load_weights_yaml(p)
    assert len(loaded) == N_DOF
    b52_idx = IDX_M1M3_START + N_M1M3_BEND - 1
    assert loaded[b52_idx] == 1.0


def test_load_weights_yaml_dict(tmp_path):
    """Dict-with-metadata YAML also works."""
    import yaml
    weights = [float(i + 1) for i in range(N_DOF)]
    p = tmp_path / "w.yaml"
    with open(p, "w") as f:
        yaml.safe_dump({
            "metadata": {"method": "orig"},
            "normalization_weights": weights,
        }, f)

    loaded = load_weights_yaml(p)
    assert len(loaded) == N_DOF
    np.testing.assert_array_equal(
        loaded, weights)
