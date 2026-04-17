#!/usr/bin/env python
"""Run DZ-to-DOF inversion across a grid of
DOF sets, rcond values, and normalization schemes.

Usage
-----
    python run_grid.py <parquet_file> \
        [--config grid_config.json] \
        [-o output_dir] [--dry-run]
"""
import argparse
import json
import subprocess
import sys


def load_config(path):
    """Load grid config from JSON file."""
    with open(path) as f:
        cfg = json.load(f)
    return (
        cfg["dof_sets"],
        cfg["rcond_values"],
        cfg["norm_schemes"],
    )


def build_command(
    parquet_file, dof_name, dof_indices,
    norm, rcond, skip_flags, output_dir,
):
    """Build the run_dz_to_dof.py command."""
    cmd = [
        sys.executable,
        "run_dz_to_dof.py",
        parquet_file,
        "--dof_indices",
    ] + [str(d) for d in dof_indices]

    cmd += ["--dof_name", dof_name]
    cmd += ["--rcond", str(rcond)]

    if norm is not None:
        cmd += ["--renorm", norm]

    cmd += ["-o", output_dir]
    cmd += skip_flags

    return cmd


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
        "--dry-run", action="store_true",
        help="Print commands without running")
    args = parser.parse_args()

    dof_sets, rcond_values, norm_schemes = (
        load_config(args.config))

    # Track what's been plotted for skip flags
    seen_sens = set()      # (norm,)
    seen_dz = False        # once per dataset
    seen_vmodes = set()    # (dof_name, norm)

    n_total = 0
    n_skipped = 0

    # Iteration order: norm -> dof_set -> rcond
    # (optimal for skip-flag clustering)
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

            for rcond in rcond_values:
                n_total += 1
                skip_flags = []

                # Sensitivity: once per norm
                sens_key = (norm_str,)
                if sens_key in seen_sens:
                    skip_flags.append(
                        "--skip-sensitivity")
                else:
                    seen_sens.add(sens_key)

                # DZ coefficients: once total
                if seen_dz:
                    skip_flags.append(
                        "--skip-dz")
                else:
                    seen_dz = True

                # V-modes: once per (dof, norm)
                vm_key = (dof_name, norm_str)
                if vm_key in seen_vmodes:
                    skip_flags.append(
                        "--skip-vmodes")
                else:
                    seen_vmodes.add(vm_key)

                cmd = build_command(
                    args.parquet_file,
                    dof_name, dof_indices,
                    norm, rcond,
                    skip_flags, args.output,
                )

                print(
                    f"\n{'=' * 60}\n"
                    f"[{n_total}] "
                    f"{dof_name}, "
                    f"norm={norm_str}, "
                    f"rcond={rcond}"
                    f"\n{'=' * 60}")

                if args.dry_run:
                    print(" ".join(cmd))
                else:
                    subprocess.run(
                        cmd, check=True)

    print(
        f"\nDone. {n_total} runs, "
        f"{n_skipped} skipped "
        f"(renorm+B52).")


if __name__ == "__main__":
    main()
