"""Render report.md -> report.pdf via headless Chrome.

Usage: uv run python scripts/make_pdf.py
"""

import pathlib, subprocess, sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 0.6in; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 8.6pt; line-height: 1.26; color: #111; }
h1 { font-size: 12.5pt; margin: 0 0 2pt; }
h2 { font-size: 10pt; margin: 8pt 0 2pt; border-bottom: 1px solid #ccc; padding-bottom: 1pt; }
h3 { font-size: 9pt; margin: 6pt 0 1pt; }
p, li { margin: 2.2pt 0; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 7.8pt; background: #f4f4f4;
       padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; font-size: 7.6pt; }
th, td { border: 1px solid #bbb; padding: 1.8pt 3pt; text-align: left; }
th { background: #eee; }
img { max-width: 52%; display: block; margin: 4pt auto; }
blockquote { margin: 4pt 0 4pt 10pt; color: #444; border-left: 2px solid #ccc; padding-left: 8pt; }
"""


def main():
    md_path = ROOT / "report.md"
    html_path = ROOT / "report.html"
    pdf_path = ROOT / "report.pdf"

    html_body = markdown.markdown(
        md_path.read_text(),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )
    size = pdf_path.stat().st_size
    print(f"wrote {pdf_path} ({size/1024:.0f} KB)")
    if size < 5000:
        sys.exit("PDF suspiciously small -- check rendering")


if __name__ == "__main__":
    main()
