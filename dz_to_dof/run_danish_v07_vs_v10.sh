#!/bin/bash
# Requires LSST env and TS packages to be set up.
# Recreate the v0.7-vs-v1.0 Danish DZ-to-DOF comparison analysis.
# Runs the 22-24dof and 50dof grids on both the v0.7 and v1.0 input
# parquets (filtered to rotator_angle=0, grouped by alt), combines each
# version's PDFs, and interleaves the two combined PDFs into a single
# diff-style comparison.

set -euo pipefail

cd "$(dirname "$0")"

V07_PARQUET=input_data/aos_fam_danish_v0.7_triplets_20260315_20260404_fits.parquet
V10_PARQUET=input_data/aos_fam_danish_v1_triplets_20260315_20260409_fits.parquet

V07_OUT=danish_v0.7_rot0
V10_OUT=danish_v1.0_rot0
INTERLEAVED=dz_to_dof_results/danish_0.7-vs-1.0_rot0_interleaved_vtest.pdf

CONFIGS=(grid_config_22-24dof grid_config_50dof)

# Run both grid configs for each parquet (same -o so the combine step
# picks up versions from both DOF sets).  --filter_val 0 on the CLI
# overrides the configs' default rotator_angle filter.
for cfg in "${CONFIGS[@]}"; do
    python run_grid.py "$V07_PARQUET" \
        --config "grid_configs/${cfg}.json" -o "$V07_OUT" \
        --filter_col_name rotator_angle --filter_val 0
    python run_grid.py "$V10_PARQUET" \
        --config "grid_configs/${cfg}.json" -o "$V10_OUT" \
        --filter_col_name rotator_angle --filter_val 0
done

V07_DATASET=$(basename "$V07_PARQUET" .parquet)
V10_DATASET=$(basename "$V10_PARQUET" .parquet)
V07_COMBINED="dz_to_dof_results/$V07_OUT/${V07_OUT}_combined.pdf"
V10_COMBINED="dz_to_dof_results/$V10_OUT/${V10_OUT}_combined.pdf"

python combine_grid_plots.py \
    "dz_to_dof_results/$V07_OUT" -o "$V07_COMBINED"
python combine_grid_plots.py \
    "dz_to_dof_results/$V10_OUT" -o "$V10_COMBINED"

python interleave.py \
    "$V07_COMBINED" "$V10_COMBINED" "$INTERLEAVED" \
    --label1 v0.7 --label2 v1.0

echo
echo "Done."
echo "  v0.7 combined: $V07_COMBINED"
echo "  v1.0 combined: $V10_COMBINED"
echo "  interleaved:   $INTERLEAVED"
