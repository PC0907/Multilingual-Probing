"""REPORT.md -> report/report.html: one self-contained page (figures embedded as data URIs)."""
import base64, re, sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "REPORT.md").read_text()


def embed(m):
    alt, src = m.group(1), m.group(2)
    data = base64.b64encode((ROOT / src).read_bytes()).decode()
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{data}"><figcaption>{alt}</figcaption></figure>'


# significance stars after a number ("-0.042*", "-0.68***" inside bold) are literal, not emphasis
md = re.sub(r"(?<=[\d\]])\*(?=\*\*|[^*]|$)", r"\\*", md, flags=re.M)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', embed, body)
body = re.sub(r"<table>", '<div class="tbl"><table>', body).replace("</table>", "</table></div>")
# the first h1 becomes the masthead
body = re.sub(r"<h1[^>]*>(.*?)</h1>", "", body, count=1)

STYLE = """
<title>Truth Directions Beyond Script</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@500;600;700&family=Literata:opsz,wght@7..72,400;7..72,600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column for prose, tables and figures break wider and scroll in their own boxes */
:root {
  --bg: #f4f6f8; --surface: #ffffff; --ink: #18202b; --muted: #556170; --rule: #d8dee6;
  --accent: #0e6a73; --mark: #b97a0c;
  --plate: #ffffff; --plate-ink: #556170;  /* figures are white matplotlib plates in both themes */
  --display: "Hanken Grotesk", "Helvetica Neue", Arial, sans-serif;
  --body: "Literata", Georgia, "Noto Serif", serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151b; --surface: #181d25; --ink: #e2e7ee; --muted: #9aa6b4; --rule: #2b333f;
  --accent: #55bcc6; --mark: #e0a43a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11151b; --surface: #181d25; --ink: #e2e7ee; --muted: #9aa6b4; --rule: #2b333f;
  --accent: #55bcc6; --mark: #e0a43a; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 400 1.0625rem/1.65 var(--body); }
.wrap { max-width: 60rem; margin: 0 auto; padding-inline: 1rem; padding-block: 2.5rem 4rem; }
header.mast { border-bottom: 2px solid var(--ink); padding-bottom: 1.25rem; margin-bottom: 2rem; }
.eyebrow { font: 600 .75rem/1.2 var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
header.mast h1 { font: 700 clamp(2rem, 5vw, 3.1rem)/1.05 var(--display); letter-spacing: -.02em; margin: .5rem 0 .75rem; text-wrap: balance; }
.scripts { font-family: var(--body); color: var(--mark); font-size: 1.05rem; letter-spacing: .02em; }
.meta { color: var(--muted); font: 400 .9rem/1.5 var(--display); margin-top: .5rem; }
main > * , main p, main ul, main ol, main blockquote { max-width: 42rem; }
h2 { font: 700 1.6rem/1.2 var(--display); margin: 3rem 0 1rem; padding-top: 1rem; border-top: 1px solid var(--rule); text-wrap: balance; }
h3 { font: 600 1.2rem/1.3 var(--display); margin: 2.2rem 0 .6rem; color: var(--accent); text-wrap: balance; }
p, li { margin: .6rem 0; }
strong { font-weight: 600; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; max-width: none; }
blockquote { margin: 1rem 0; padding: .5rem 1rem; border-left: 3px solid var(--mark); color: var(--muted); }
code { font: 400 .86em var(--mono); background: color-mix(in oklab, var(--rule) 45%, transparent); padding: .05em .3em; border-radius: 3px; }
pre { max-width: none !important; overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); padding: 1rem; border-radius: 4px; }
pre code { background: none; padding: 0; font-size: .8rem; line-height: 1.5; }
.tbl { max-width: none !important; overflow-x: auto; margin: 1rem 0 1.5rem; border: 1px solid var(--rule); background: var(--surface); border-radius: 4px; }
table { border-collapse: collapse; font: 400 .82rem/1.4 var(--mono); font-variant-numeric: tabular-nums; min-width: 100%; }
th { font: 600 .74rem/1.3 var(--display); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); text-align: left; background: color-mix(in oklab, var(--rule) 35%, var(--surface)); }
th, td { padding: .45rem .7rem; border-bottom: 1px solid var(--rule); white-space: nowrap; vertical-align: top; }
td:first-child { white-space: normal; min-width: 9rem; }
tr:last-child td { border-bottom: 0; }
figure { max-width: none !important; margin: 1.5rem 0; background: var(--plate); border: 1px solid var(--rule); border-radius: 4px; padding: .75rem; }
figure img { display: block; width: 100%; height: auto; max-width: 100%; }
figcaption { font: 500 .78rem/1.3 var(--display); color: var(--plate-ink); margin-top: .4rem; text-transform: uppercase; letter-spacing: .06em; }
main > ol:first-of-type li { margin: .8rem 0; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@page { size: A4; margin: 16mm 14mm 18mm; }
@media print {
  :root { --bg: #ffffff; --surface: #ffffff; --ink: #18202b; --muted: #4a5563; --rule: #cfd6de; --accent: #0e6a73; --mark: #9a6508; color-scheme: light; }
  body { font-size: 10pt; line-height: 1.45; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .wrap { max-width: none; padding: 0; }
  main > *, main p, main ul, main ol, main blockquote { max-width: none; }
  h2 { break-before: page; border-top: 0; margin-top: 0; font-size: 17pt; }
  h2, h3 { break-after: avoid; }
  .tbl { overflow: visible; break-inside: auto; }
  table { font-size: 7.2pt; min-width: 0; width: 100%; }
  th, td { white-space: normal; padding: .25rem .35rem; }
  td:first-child { min-width: 0; }
  tr, figure, pre { break-inside: avoid; }
  figure { padding: .3rem; }
  pre code { font-size: 7.5pt; white-space: pre-wrap; }
  header.mast h1 { font-size: 26pt; }
}
</style>
"""

HEAD = """
<div class="wrap">
<header class="mast">
  <div class="eyebrow">Extension of the NAACL truth-directions study</div>
  <h1>Truth directions in low-resource languages and scripts</h1>
  <div class="scripts">اردو · मराठी · नेपाली · ગુજરાતી · ਪੰਜਾਬੀ · پنجابی · Roman Hindi</div>
  <div class="meta">4 models (Gemma-7B, Qwen3-8B-Base, Apertus-8B-2509, Mistral-7B-v0.3) · 18 conditions ·
  2,000 claims each · mass-mean probes at the paper's frozen blocks · 1,000-draw group bootstrap</div>
</header>
<main>
"""

html = STYLE + HEAD + body + "\n</main>\n</div>\n"
out = ROOT / "report" / "report.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(out, f"{len(html)/1e6:.1f} MB")
