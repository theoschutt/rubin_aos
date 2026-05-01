#!/usr/bin/env python
"""Generate cached OFC artifacts (sensitivity matrix
+ normalization weights) as YAML/.npy files.

Run this once with the LSST stack set up; afterwards
``run_dz_to_dof.py`` can be invoked with
``--smatrix_file`` and ``--weights_file`` pointing at
these artifacts, bypassing the LSST stack at runtime.

Usage
-----
    python build_ofc_cache.py [-o output_dir]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from dz_to_dof import (
    DOF_LABELS, N_DOF,
    IDX_M1M3_START, N_M1M3_BEND,
    load_ofc_data,
    load_sensitivity_matrix,
    get_rf_weights,
    pad_ofc_array,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_ofc_cache")


def main():
    """CLI entry point: load OFC data via the LSST stack and dump the
    sensitivity matrix (YAML + .npy) and normalization weights (YAML)
    into the output dir so downstream scripts can run stack-free."""
    parser = argparse.ArgumentParser(
        description="Build OFC cache files.")
    parser.add_argument(
        "-o", "--output", default="ofc_cache",
        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    log.info("Loading OFCData")
    ofc_data = load_ofc_data()

    # --- Sensitivity matrix ---
    log.info("Building full sensitivity matrix")
    # Use a dummy focal/pupil selection; we only
    # want the full_coef output, which is
    # independent of the slicing.
    _, full_coef, _ = load_sensitivity_matrix(
        ofc_data,
        focal_indices=[1],
        pupil_indices=[4],
        norm_type=None,
    )
    smatrix_npy = out_dir / "smatrix_cache.npy"
    np.save(smatrix_npy, full_coef)
    log.info(
        "Saved smatrix to %s (shape %s)",
        smatrix_npy, full_coef.shape)

    smatrix_yaml = out_dir / "smatrix_cache.yaml"
    with open(smatrix_yaml, "w") as f:
        yaml.safe_dump({
            "dof_labels": list(DOF_LABELS),
            "smatrix_npy": smatrix_npy.name,
        }, f, sort_keys=False)
    log.info("Saved smatrix spec to %s", smatrix_yaml)

    # --- 'orig' normalization weights ---
    orig_weights = pad_ofc_array(
        np.asarray(ofc_data.normalization_weights))
    orig_yaml = out_dir / "weights_orig.yaml"
    with open(orig_yaml, "w") as f:
        yaml.safe_dump({
            "metadata": {
                "instrument": "lsst",
                "method": "orig",
            },
            "normalization_weights": (
                orig_weights.tolist()),
        }, f, sort_keys=False)
    log.info("Saved orig weights to %s", orig_yaml)

    # --- 'geom' normalization weights ---
    # Compute on non-B52 DOFs to avoid f_i = 0
    # at the (zero) B52 column; pad with 1.0.
    b52_idx = IDX_M1M3_START + N_M1M3_BEND - 1
    non_b52 = [
        i for i in range(N_DOF) if i != b52_idx]
    r_i, f_i, _ = get_rf_weights(
        ofc_data, full_coef, non_b52)
    geom_weights_50 = np.sqrt(r_i / f_i)
    geom_weights = pad_ofc_array(
        geom_weights_50, fill_value=1.0)
    geom_yaml = out_dir / "weights_geom.yaml"
    with open(geom_yaml, "w") as f:
        yaml.safe_dump({
            "metadata": {
                "instrument": "lsst",
                "method": "geom",
            },
            "normalization_weights": (
                geom_weights.tolist()),
        }, f, sort_keys=False)
    log.info("Saved geom weights to %s", geom_yaml)

    log.info("Cache built in %s", out_dir)


if __name__ == "__main__":
    main()
