"""Render CHECKPOINT_3_v2.md to a print-ready PDF via Edge headless.
Usage (from FinalProject/): uv run --with markdown python build_checkpoint_pdf.py"""
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC, HTML, PDF = ROOT / 'CHECKPOINT_3_v2.md', ROOT / 'build/checkpoint3.html', ROOT / 'Milestone3_Jefferson_Kline.pdf'
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

body = markdown.markdown(SRC.read_text(encoding='utf-8'), extensions=['tables'])
body = body.replace('src="figures/', f'src="{(ROOT / "figures").as_uri()}/')
css = """
@page { size: Letter; margin: 0.65in 0.75in; }
body { font: 10pt/1.42 "Segoe UI", Calibri, Arial, sans-serif; color: #111; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; break-after: avoid; }
p, li { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 6pt 0; break-inside: avoid; }
th, td { border-bottom: 1px solid #ddd; padding: 3pt 4pt; text-align: left; white-space: nowrap; }
td:not(:first-child):not(:nth-child(2)) { text-align: right; font-variant-numeric: tabular-nums; }
th { border-bottom: 1.5px solid #888; }
img { width: 100%; break-inside: avoid; }
code { font: 9pt Consolas, monospace; }
em { color: #444; }
a { color: #1a4f8b; text-decoration: none; }
"""
HTML.parent.mkdir(exist_ok=True)
HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Milestone 3</title><style>{css}</style></head><body>{body}</body></html>', encoding='utf-8')
subprocess.run([EDGE, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={PDF}', HTML.as_uri()], check=True, timeout=120)
print('wrote', PDF, PDF.stat().st_size, 'bytes')


