#!/usr/bin/env python
"""Collate DOF + DZ-residual plots from a grid run
into a single giant PDF, one page per grid version.

Walks ``<grid_dir>/<dataset>/<version>/`` looking for
``dof_solution_<ver>.pdf`` and ``dz_residuals_<ver>.pdf``
(or ``dz_residuals_fixed_ylims_<ver>.pdf`` for older
runs).  Each version contributes one page: DOF on
the left, DZ residual on the right.  Versions are
sorted alphabetically and each gets a PDF bookmark
for easy navigation.

Usage
-----
    python combine_grid_plots.py <grid_dir> \
        [-o combined.pdf]
"""
import argparse
import logging
from pathlib import Path

import pypdf
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("combine_grid_plots")


def find_pdf(version_dir, prefix, ver):
    """Return path to <prefix>_<ver>.pdf in
    version_dir, or None if not found.
    """
    p = version_dir / f"{prefix}_{ver}.pdf"
    return p if p.exists() else None


def combine_two_pages(dof_pdf, resid_pdf):
    """Return a pypdf PageObject with the two input
    single-page PDFs placed side-by-side (DOF left,
    residual right), both vertically top-aligned.
    """
    dof_page = PdfReader(str(dof_pdf)).pages[0]
    res_page = PdfReader(str(resid_pdf)).pages[0]

    w1 = float(dof_page.mediabox.width)
    h1 = float(dof_page.mediabox.height)
    w2 = float(res_page.mediabox.width)
    h2 = float(res_page.mediabox.height)

    new_w = w1 + w2
    new_h = max(h1, h2)

    writer = PdfWriter()
    blank = writer.add_blank_page(
        width=new_w, height=new_h)

    # Place DOF on the left, top-aligned
    blank.merge_transformed_page(
        dof_page,
        Transformation().translate(0, new_h - h1),
    )
    # Place residual on the right, top-aligned
    blank.merge_transformed_page(
        res_page,
        Transformation().translate(
            w1, new_h - h2),
    )
    blank.mediabox = RectangleObject(
        (0, 0, new_w, new_h))
    return blank


def main():
    parser = argparse.ArgumentParser(
        description="Collate grid plots.")
    parser.add_argument(
        "grid_dir",
        help="Top-level grid output directory")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDF path "
        "(default: <grid_dir>/<grid_dir>_combined.pdf)")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    if not grid_dir.is_dir():
        raise SystemExit(
            f"Not a directory: {grid_dir}")

    output = (
        Path(args.output) if args.output
        else grid_dir / f"{grid_dir}_combined.pdf")

    writer = PdfWriter()

    # Find all <dataset>/<version>/ subdirectories
    entries = []
    for dataset_dir in sorted(
        p for p in grid_dir.iterdir()
        if p.is_dir()
    ):
        for version_dir in sorted(
            p for p in dataset_dir.iterdir()
            if p.is_dir()
        ):
            ver = version_dir.name
            dof = find_pdf(
                version_dir, "dof_solution", ver)
            res = find_pdf(
                version_dir, "dz_residuals_fixed_ylims", ver)
            if res is None:
                # Older runs used a separate
                # fixed-ylim filename.
                res = find_pdf(
                    version_dir,
                    "dz_residuals",
                    ver)
            if dof is None or res is None:
                log.warning(
                    "Skipping %s/%s (missing "
                    "DOF or residual plot)",
                    dataset_dir.name, ver)
                continue
            entries.append(
                (dataset_dir.name, ver,
                 dof, res))

    log.info(
        "Found %d grid runs to combine",
        len(entries))

    for i, (dataset, ver, dof, res) in enumerate(
        entries
    ):
        log.info(
            "[%d/%d] %s / %s",
            i + 1, len(entries), dataset, ver)
        page = combine_two_pages(dof, res)
        writer.add_page(page)
        # Bookmark on the page just added
        writer.add_outline_item(
            f"{dataset} / {ver}",
            len(writer.pages) - 1,
        )

    output.parent.mkdir(
        parents=True, exist_ok=True)
    with open(output, "wb") as f:
        writer.write(f)
    log.info(
        "Wrote %d pages to %s",
        len(writer.pages), output)


if __name__ == "__main__":
    main()
