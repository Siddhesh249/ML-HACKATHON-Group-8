#!/usr/bin/env python3
"""Build the consolidated technical report PDF from ``report/report.md``.

Markdown → HTML (python-markdown) → PDF (headless Chrome/Chromium "print to
PDF"). Figures are referenced from ``results/`` so the PDF always shows the
committed evidence. Output: ``report/PathSnap_Group8_Report.pdf`` (and the
intermediate ``report/report.html``).

Requires: ``pip install markdown`` and Google Chrome or Chromium.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "report.md"
HTML = ROOT / "report" / "report.html"
PDF = ROOT / "report" / "PathSnap_Group8_Report.pdf"
CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
html { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; color: #1b1b1b; line-height: 1.42; }
body { margin: 0; }
h1 { font-size: 24pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; border-bottom: 1.5px solid #3b6ea8; padding-bottom: 3pt; margin: 18pt 0 8pt; break-before: page; }
h3 { font-size: 12pt; margin: 14pt 0 6pt; color: #234; }
h4 { font-size: 10.8pt; margin: 12pt 0 4pt; color: #234; break-after: avoid; }
p, li { text-align: left; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.4pt; break-inside: avoid; }
th, td { border: 0.6px solid #b8c2cc; padding: 3pt 4pt; vertical-align: top; }
th { background: #eef2f7; text-align: left; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #f3f4f6; padding: 0 2px; border-radius: 2px; }
pre { background: #f3f4f6; padding: 6pt 8pt; font-size: 7.8pt; line-height: 1.35; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { display: block; max-width: 100%; max-height: 118mm; margin: 8pt auto 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; margin-bottom: 0; }
p.caption { font-size: 8.6pt; color: #475569; font-style: italic; text-align: center; margin: 0 6mm 10pt; }
hr { display: none; }
.title-page { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.title-page h1 { font-size: 28pt; border: none; break-before: avoid; }
.title-page h2 { font-size: 16pt; border: none; color: #3b6ea8; break-before: avoid; margin-top: 4pt; }
.title-page table { width: 70%; font-size: 11pt; margin: 18pt 0; }
.title-page th, .title-page td { padding: 5pt 8pt; }
.title-page p { font-size: 11pt; }
.title-page + h2 { break-before: page; }
.todo { background: #fff3cd; padding: 1px 4px; border: 1px dashed #c69500; }
"""


def find_chrome() -> str | None:
    for c in CHROME:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def main() -> int:
    try:
        import markdown
    except ImportError:
        print("pip install markdown")
        return 1
    text = SRC.read_text(encoding="utf-8")
    # Prefer fresh screenshots from the live two-laptop demo when they exist.
    shots = ROOT / "results" / "screenshots"
    for old_name, new_name in (("api_terminal_log.png", "final_api_terminal.png"),
                               ("ui_dark_hmm_trip00000000.png", "final_ui.png")):
        if (shots / new_name).exists():
            text = text.replace(f"screenshots/{old_name}", f"screenshots/{new_name}")
            print(f"using live demo screenshot {new_name}")
    if (shots / "final_api_terminal.png").exists() and (shots / "final_ui.png").exists():
        import re as _re
        text = _re.sub(r" \(Captured during\s+integration testing with an earlier decoder\s+configuration;[^)]*\)", "", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    import re
    body = re.sub(r"<p><em>(Figure \d+[^<]*(?:<code>[^<]*</code>[^<]*)*)</em></p>", r'<p class="caption">\1</p>', body)
    HTML.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PathSnap — Technical Report (Group 8)</title><style>{CSS}</style></head>
<body>{body}</body></html>""", encoding="utf-8")
    chrome = find_chrome()
    if chrome is None:
        print(f"HTML written to {HTML.relative_to(ROOT)}; no Chrome found — open it in a browser and print to PDF.")
        return 0
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=10000", f"--print-to-pdf={PDF}", HTML.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    print(f"PDF written to {PDF.relative_to(ROOT)} ({PDF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
