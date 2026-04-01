#!/usr/bin/env python3
"""Merge per-DOF sensitivity PDFs in logical order using Ghostscript."""
import subprocess
import sys
from pathlib import Path

d = Path("/sdf/home/s/schutt20/u/lsstcam_commish/sens_results+sim_max-k6-j28/DOF_combined")
out = d / "all_combined_v3.pdf"

ordered = []

# Cam rigid body: rx, ry, x, y, z
for dof in ["rx", "ry", "x", "y", "z"]:
    ordered.extend(sorted(d.glob(f"Cam_{dof}_*v3.pdf")))

# M2 rigid body: rx, ry, x, y, z
for dof in ["rx", "ry", "x", "y", "z"]:
    ordered.extend(sorted(d.glob(f"M2_{dof}_*v3.pdf")))

# M1M3 bending modes (sorted numerically by padded name)
ordered.extend(sorted(d.glob("M1M3_B*v3.pdf")))

# M2 bending modes
ordered.extend(sorted(d.glob("M2_B*v3.pdf")))

# Multi-DOF modes (filenames starting with '[')
ordered.extend(sorted(d.glob("'[*v3.pdf")) or sorted(d.glob("[*v3.pdf")))

# nullmodes
ordered.extend(sorted(d.glob("nullmode_*v3.pdf")))

# zmodes
ordered.extend(sorted(d.glob("zmode_*v3.pdf")))

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
