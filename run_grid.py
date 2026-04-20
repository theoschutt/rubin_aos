#!/usr/bin/env python
"""Run DZ-to-DOF inversion across a grid of
DOF sets, truncation values (rcond and/or rank),
and normalization schemes.

Runs iterations in-process so the OFC data load
(or module imports) is paid at most once.

Usage
-----
    python run_grid.py <parquet_file> \
        [--config grid_config.json] \
        [-o output_dir] [--dry-run]

The config JSON may have ``rcond_values``,
``rank_values``, or both.  Each listed value
becomes one run per (dof_set, norm) combo.
"""
import argparse
import copy
import json
import logging

import run_dz_to_dof
from dz_to_dof import load_ofc_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_grid")


def load_config(path):
    """Load grid config from JSON file.

    Returns
    -------
    dof_sets : dict
    mode_specs : list of (str, value)
    norm_schemes : list
    """
    with open(path) as f:
        cfg = json.load(f)
    has_rcond = "rcond_values" in cfg
    has_rank = "rank_values" in cfg
    if not (has_rcond or has_rank):
        raise ValueError(
            "grid_config must have at least one "
            "of rcond_values or rank_values")
    mode_specs = []
    for r in cfg.get("rcond_values", []):
        mode_specs.append(("rcond", r))
    for k in cfg.get("rank_values", []):
        mode_specs.append(("rank", k))
    return (
        cfg["dof_sets"],
        mode_specs,
        cfg["norm_schemes"],
    )


def _base_defaults():
    """Build an argparse.Namespace with all the
    fields run_dz_to_dof.run_single expects,
    populated from the main parser's defaults.
    """
    # Pre-parse with no extra args to get defaults
    # (requires a placeholder for the positional
    # parquet_file; we overwrite it per run).
    import sys
    saved = sys.argv
    try:
        sys.argv = ["run_dz_to_dof.py", "PLACEHOLDER"]
        # Build the parser the same way main() does
        # by calling main() up to parse_args is
        # awkward; instead, construct a Namespace
        # manually with all defaults.
    finally:
        sys.argv = saved
    # Build Namespace with the same defaults as
    # run_dz_to_dof.main()'s argparse.
    from run_dz_to_dof import (
        DEFAULT_PUPIL_INDICES,
        DEFAULT_FOCAL_INDICES,
        DEFAULT_DOF_INDICES,
    )
    return argparse.Namespace(
        parquet_file=None,
        pupil_indices=DEFAULT_PUPIL_INDICES,
        focal_indices=DEFAULT_FOCAL_INDICES,
        dof_indices=DEFAULT_DOF_INDICES,
        renorm=None,
        rot_tolerance=1.0,
        rcond=1e-4,
        rank=None,
        smatrix_file=None,
        weights_file=None,
        output="dz_to_dof_results",
        dof_name=None,
        version=None,
        skip_sensitivity=False,
        skip_dz=False,
        skip_vmodes=False,
    )


def build_args(
    parquet_file, dof_name, dof_indices,
    norm, mode, mode_value,
    skip_flags, output_dir,
    smatrix_file=None, weights_file=None,
):
    """Build a Namespace for one grid iteration."""
    args = _base_defaults()
    args.parquet_file = parquet_file
    args.dof_indices = list(dof_indices)
    args.dof_name = dof_name
    args.renorm = norm
    args.output = output_dir
    args.smatrix_file = smatrix_file
    args.weights_file = weights_file
    if mode == "rank":
        args.rank = mode_value
    else:
        args.rcond = mode_value
    args.skip_sensitivity = (
        "--skip-sensitivity" in skip_flags)
    args.skip_dz = (
        "--skip-dz" in skip_flags)
    args.skip_vmodes = (
        "--skip-vmodes" in skip_flags)
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run DZ-to-DOF grid sweep.")
    parser.add_argument("parquet_file",
                        help="Path to parquet file")
    parser.add_argument(
        "--config", default="grid_config.json",
        help="Path to grid config JSON")
    parser.add_argument(
        "-o", "--output",
        default="dz_to_dof_results",
        help="Output directory")
    parser.add_argument(
        "--smatrix_file", type=str, default=None,
        help="YAML spec for a cached smatrix "
        "(bypasses OFC load if combined with "
        "--weights_file)")
    parser.add_argument(
        "--weights_file", type=str, default=None,
        help="YAML with precomputed norm weights "
        "(per-run; applied to all grid points)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without running")
    args = parser.parse_args()

    (dof_sets, mode_specs, norm_schemes) = (
        load_config(args.config))

    # Decide whether we need OFC up front.  If
    # smatrix_file is given but weights_file is
    # not, OFC is only needed when a renorm is
    # used; we load lazily in that case.
    ofc_data = None

    # Track what's been plotted for skip flags
    seen_sens = set()      # (norm,)
    seen_dz = False
    seen_vmodes = set()    # (dof_name, norm)

    n_total = 0
    n_skipped = 0

    for norm in norm_schemes:
        norm_str = norm if norm else "None"
        for dof_name, dof_indices in (
            dof_sets.items()
        ):
            # Skip renorm + M1M3_B52 combos
            if norm is not None and (
                30 in dof_indices
            ):
                n_skipped += 1
                continue

            for mode, mode_value in mode_specs:
                n_total += 1
                skip_flags = []

                sens_key = (norm_str,)
                if sens_key in seen_sens:
                    skip_flags.append(
                        "--skip-sensitivity")
                else:
                    seen_sens.add(sens_key)

                if seen_dz:
                    skip_flags.append("--skip-dz")
                else:
                    seen_dz = True

                vm_key = (dof_name, norm_str)
                if vm_key in seen_vmodes:
                    skip_flags.append(
                        "--skip-vmodes")
                else:
                    seen_vmodes.add(vm_key)

                run_args = build_args(
                    args.parquet_file,
                    dof_name, dof_indices,
                    norm, mode, mode_value,
                    skip_flags, args.output,
                    smatrix_file=args.smatrix_file,
                    weights_file=args.weights_file,
                )

                log.info(
                    "=" * 60 + "\n"
                    "[%d] %s, norm=%s, %s=%s\n"
                    + "=" * 60,
                    n_total, dof_name, norm_str,
                    mode, mode_value,
                )

                if args.dry_run:
                    log.info(
                        "(dry-run) would call "
                        "run_single(args)")
                    continue

                # Lazy-load OFC the first time a
                # run actually needs it.
                need_ofc = (
                    args.smatrix_file is None
                    or (norm is not None
                        and args.weights_file
                        is None)
                )
                if (need_ofc
                        and ofc_data is None):
                    log.info(
                        "Loading OFCData "
                        "(shared across grid)")
                    ofc_data = load_ofc_data()

                try:
                    run_dz_to_dof.run_single(
                        run_args,
                        ofc_data=ofc_data,
                    )
                except Exception:
                    log.exception(
                        "Run failed; continuing")

    log.info(
        "Done. %d runs, %d skipped (renorm+B52).",
        n_total, n_skipped)


if __name__ == "__main__":
    main()
