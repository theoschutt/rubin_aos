"""Interleave two combined-grid PDFs page-by-page with version labels."""
import argparse

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf1", help="First PDF (labelled with --label1)")
    p.add_argument("pdf2", help="Second PDF (labelled with --label2)")
    p.add_argument("output", help="Output interleaved PDF path")
    p.add_argument("--label1", default="v0.7")
    p.add_argument("--label2", default="v1.0")
    args = p.parse_args()

    writer = PdfWriter()
    pdf1 = PdfReader(args.pdf1)
    pdf2 = PdfReader(args.pdf2)

    for page1, page2 in zip(pdf1.pages, pdf2.pages):
        writer.add_page(page1)
        writer.add_annotation(
            page_number=-1,
            annotation=FreeText(
                text=args.label1,
                rect=(1970, 890, 2000, 920),
                font_size="30pt",
            ),
        )
        writer.add_page(page2)
        writer.add_annotation(
            page_number=-1,
            annotation=FreeText(
                text=args.label2,
                rect=(1970, 890, 2000, 920),
                font_size="30pt",
            ),
        )

    with open(args.output, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    main()
