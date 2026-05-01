#!/usr/bin/env python
"""Run DZ-to-DOF inversion across a grid of
DOF sets, truncation values (rcond and/or rank),
and normalization schemes.

Runs iterations in-process so the OFC data load
(or module imports) is paid at most once.

All CLI arguments accepted by ``run_dz_to_dof.py``
are also accepted here (and applied to every
iteration).  Grid-specific extras: ``--config``
and ``--dry-run``.

The config JSON may include any of:
  * ``dof_sets``:     {name: [indices, ...]}
  * ``rcond_values``: [floats]
  * ``rank_values``:  [ints]
  * ``norm_schemes``: [null | "orig" | "geom", ...]
  * ``run_args``:     dict of default arg values
                      applied to every run (CLI
                      overrides if both given)

Each config entry in ``rcond_values`` /
``rank_values`` contributes one run per
(dof_set, norm) combo.

Usage
-----
    python run_grid.py <parquet_file> \
        [--config grid_config.json] \
        [-o output_dir] [--dry-run]
        [<any run_dz_to_dof option>]
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
    run_defaults : dict
        Values to apply as parser defaults
        (overridable by CLI).
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
        cfg.get("run_args", {}),
    )


def _peek_config_path():
    """Peek at --config before full parsing so we
    can apply the config's run_args as parser
    defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config", default="grid_config.json")
    pre_args, _ = pre.parse_known_args()
    return pre_args.config


def main():
    """CLI entry point: load a JSON grid config, then invoke
    :func:`run_dz_to_dof.main` once per (DOF set, normalization, rcond/
    rank) combination, sharing a single OFC data load across runs."""
    # Peek at --config so we can apply its
    # run_args as parser defaults BEFORE parsing.
    # Tolerate missing file (e.g. --help) by
    # falling back to empty defaults.
    config_path = _peek_config_path()
    try:
        (dof_sets, mode_specs, norm_schemes,
         run_defaults) = load_config(config_path)
        config_loaded = True
    except FileNotFoundError:
        dof_sets = mode_specs = norm_schemes = None
        run_defaults = {}
        config_loaded = False

    # Reuse run_dz_to_dof's parser so every CLI
    # arg is accepted automatically.
    parser = run_dz_to_dof.build_parser()
    parser.description = (
        "Run DZ-to-DOF grid sweep.")
    parser.add_argument(
        "--config", default="grid_config.json",
        help="Path to grid config JSON")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without running")
    # Apply run_args from config as defaults
    # (CLI-specified values still win).
    if run_defaults:
        parser.set_defaults(**run_defaults)

    args = parser.parse_args()

    if not config_loaded:
        parser.error(
            f"config file not found: "
            f"{config_path}")

    if (len(args.parquet_file) > 1
            and args.dataset_name is None):
        parser.error(
            "--dataset_name is required when "
            "more than one parquet file is given")

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

                # Build this iteration's args by
                # copying the parsed Namespace
                # and overwriting the per-iter
                # fields.
                run_args = copy.copy(args)
                run_args.dof_indices = list(
                    dof_indices)
                run_args.dof_name = dof_name
                run_args.renorm = norm
                if mode == "rank":
                    run_args.rank = mode_value
                    run_args.rcond = (
                        parser.get_default(
                            "rcond"))
                else:
                    run_args.rcond = mode_value
                    run_args.rank = None

                # Skip-flag bookkeeping
                sens_key = (norm_str,)
                run_args.skip_sensitivity = (
                    sens_key in seen_sens)
                if not run_args.skip_sensitivity:
                    seen_sens.add(sens_key)

                run_args.skip_dz = seen_dz
                if not seen_dz:
                    seen_dz = True

                vm_key = (dof_name, norm_str)
                run_args.skip_vmodes = (
                    vm_key in seen_vmodes)
                if not run_args.skip_vmodes:
                    seen_vmodes.add(vm_key)

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
