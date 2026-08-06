"""Render report.md -> report.pdf via headless Chrome.

Usage: uv run python scripts/make_pdf.py
"""

import pathlib, subprocess, sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 0.7in; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 9.4pt; line-height: 1.34; color: #111; }
h1 { font-size: 14pt; margin: 0 0 2pt; }
h2 { font-size: 11pt; margin: 11pt 0 3pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 9.8pt; margin: 8pt 0 2pt; }
p, li { margin: 3pt 0; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt; background: #f4f4f4;
       padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0; font-size: 8.2pt; }
th, td { border: 1px solid #bbb; padding: 2.5pt 4pt; text-align: left; }
th { background: #eee; }
img { max-width: 100%; margin: 5pt 0; }
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
