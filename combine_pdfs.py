#!/usr/bin/env python3
"""Merge per-DOF sensitivity PDFs in logical order using Ghostscript."""
import subprocess
from pathlib import Path

d = Path("/sdf/home/s/schutt20/u/rubin_aos/"
         "sens_results+sim_max-k6-j28/DOF_combined")
version = "v4"
plot_type = "sensitivity_summary"
# plot_type = "sensmat_impact"
out = d / f"{plot_type}_all_combined_{version}.pdf"

ordered = []

# Cam rigid body: rx, ry, x, y, z
for dof in ["rx", "ry", "x", "y", "z"]:
    ordered.extend(sorted(d.glob(f"Cam_{dof}_*{version}.pdf")))

# M2 rigid body: rx, ry, x, y, z
for dof in ["rx", "ry", "x", "y", "z"]:
    ordered.extend(sorted(d.glob(f"M2_{dof}_*{version}.pdf")))

# M1M3 bending modes (sorted numerically by padded name)
ordered.extend(sorted(d.glob(f"M1M3_B*{version}.pdf")))

# M2 bending modes
ordered.extend(sorted(d.glob(f"M2_B*{version}.pdf")))

# Multi-DOF modes (filenames starting with '[')
ordered.extend(sorted(d.glob(f"'[*{version}.pdf")) or sorted(d.glob(f"[*{version}.pdf")))

# nullmodes
ordered.extend(sorted(d.glob(f"nullmode_*{version}.pdf")))

# zmodes
ordered.extend(sorted(d.glob(f"zmode_*{version}.pdf")))

missing = [f for f in ordered if not f.exists()]
if missing:
    print("WARNING: missing files:", missing)
    ordered = [f for f in ordered if f.exists()]

print(f"Merging {len(ordered)} files:")
for f in ordered:
    print(f"  {f.name}")

cmd = (
    ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
     f"-sOutputFile={out}"]
    + [str(f) for f in ordered]
)
subprocess.run(cmd, check=True)
print(f"\nWritten to {out}")
