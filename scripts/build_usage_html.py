"""Convert USAGE.md → usage.html with a responsive standalone shell.

Run from the project root:  python scripts/build_usage_html.py
"""
from __future__ import annotations

from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "USAGE.md"
OUT = ROOT / "usage.html"

CSS = """
:root {
  --bg: #ffffff;
  --fg: #1f2328;
  --muted: #57606a;
  --accent: #0969da;
  --accent-soft: #ddf4ff;
  --border: #d0d7de;
  --code-bg: #f6f8fa;
  --code-fg: #1f2328;
  --table-zebra: #f6f8fa;
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --accent-soft: #122036;
    --border: #30363d;
    --code-bg: #161b22;
    --code-fg: #e6edf3;
    --table-zebra: #161b22;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.3);
  }
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 24px; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
header.page {
  padding: 32px clamp(16px, 4vw, 48px);
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--accent-soft) 0%, var(--bg) 100%);
}
header.page h1 {
  margin: 0 0 8px;
  font-size: clamp(1.6rem, 3vw + 0.5rem, 2.4rem);
  font-weight: 600;
  letter-spacing: -0.02em;
}
header.page p { margin: 0; color: var(--muted); max-width: 800px; }

.layout {
  display: grid;
  grid-template-columns: 1fr;
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px clamp(16px, 4vw, 48px);
  gap: 32px;
}
@media (min-width: 1024px) {
  .layout { grid-template-columns: 260px minmax(0, 1fr); }
}

nav.toc {
  font-size: 0.92rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  background: var(--bg);
  box-shadow: var(--shadow);
}
@media (min-width: 1024px) {
  nav.toc { position: sticky; top: 24px; align-self: start; max-height: calc(100vh - 48px); overflow: auto; }
}
nav.toc .toc-title {
  font-weight: 600;
  font-size: 0.78rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc li { margin: 2px 0; }
nav.toc a { color: var(--fg); text-decoration: none; display: block; padding: 4px 8px; border-radius: 6px; }
nav.toc a:hover { background: var(--accent-soft); color: var(--accent); }
nav.toc ul ul { padding-left: 14px; border-left: 1px solid var(--border); margin-left: 8px; }

main.content { min-width: 0; }
main.content h1, main.content h2, main.content h3, main.content h4 {
  line-height: 1.25;
  scroll-margin-top: 24px;
  margin-top: 1.6em;
  margin-bottom: 0.5em;
}
main.content > :first-child { margin-top: 0; }
main.content h1 { font-size: clamp(1.4rem, 1.5vw + 1rem, 1.9rem); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
main.content h2 { font-size: clamp(1.25rem, 1vw + 1rem, 1.55rem); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
main.content h3 { font-size: 1.2rem; }
main.content h4 { font-size: 1.05rem; color: var(--muted); }

main.content a { color: var(--accent); text-decoration: none; }
main.content a:hover { text-decoration: underline; }

main.content p, main.content li { line-height: 1.65; }
main.content ul, main.content ol { padding-left: 1.4em; }
main.content li + li { margin-top: 4px; }

main.content code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.88em;
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 0.15em 0.4em;
  border-radius: 4px;
}
main.content pre {
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 14px 16px;
  border-radius: 8px;
  overflow-x: auto;
  border: 1px solid var(--border);
  line-height: 1.5;
  font-size: 0.88rem;
}
main.content pre code { background: transparent; padding: 0; font-size: inherit; }

main.content blockquote {
  margin: 1em 0;
  padding: 8px 16px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  color: var(--fg);
  border-radius: 0 6px 6px 0;
}
main.content blockquote p:first-child { margin-top: 0; }
main.content blockquote p:last-child { margin-bottom: 0; }

main.content hr {
  border: 0;
  border-top: 1px solid var(--border);
  margin: 2em 0;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 1em 0;
  box-shadow: var(--shadow);
}
main.content table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.93rem;
  min-width: 480px;
}
main.content th, main.content td {
  text-align: left;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
main.content th {
  background: var(--code-bg);
  font-weight: 600;
  position: sticky;
  top: 0;
}
main.content tr:nth-child(2n) td { background: var(--table-zebra); }
main.content td code { white-space: nowrap; }

main.content kbd {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-bottom-width: 2px;
  border-radius: 4px;
  padding: 1px 6px;
  font-family: ui-monospace, monospace;
  font-size: 0.85em;
}

a.anchor { color: var(--muted); margin-right: 8px; text-decoration: none; opacity: 0; transition: opacity 0.1s; font-weight: 400; }
h1:hover a.anchor, h2:hover a.anchor, h3:hover a.anchor, h4:hover a.anchor { opacity: 1; }

footer.page {
  border-top: 1px solid var(--border);
  padding: 24px clamp(16px, 4vw, 48px);
  color: var(--muted);
  font-size: 0.9rem;
  text-align: center;
}
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Ollama Fine-Tune Studio — User Guide</title>
<style>{css}</style>
</head>
<body>
<header class="page">
  <h1>Ollama Fine-Tune Studio</h1>
  <p>User guide: prerequisites, installation, every tab, every parameter, all external APIs, and troubleshooting.</p>
</header>
<div class="layout">
  <nav class="toc" aria-label="Table of contents">
    <div class="toc-title">On this page</div>
    {toc}
  </nav>
  <main class="content">
{body}
  </main>
</div>
<footer class="page">Standalone documentation — open this file directly in any browser. No internet required.</footer>
</body>
</html>
"""


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")

    # The hand-written TOC in the markdown is redundant with the auto-TOC
    # we generate. Strip it to avoid duplication in the HTML.
    if "## Table of Contents" in md_text:
        start = md_text.index("## Table of Contents")
        # Find the next "---" after the TOC, which marks the end.
        end = md_text.index("\n---\n", start) + len("\n---\n")
        md_text = md_text[:start] + md_text[end:]

    md = markdown.Markdown(
        extensions=[
            "extra",            # tables, fenced code, abbreviations, etc.
            "sane_lists",
            TocExtension(toc_depth="2-3", anchorlink=True, permalink=False),
        ],
        output_format="html5",
    )
    body_html = md.convert(md_text)
    toc_html = md.toc

    # Wrap every <table> in a horizontally-scrollable div so they don't
    # blow out the layout on narrow screens.
    body_html = body_html.replace(
        "<table>", '<div class="table-wrap"><table>'
    ).replace("</table>", "</table></div>")

    OUT.write_text(
        HTML_SHELL.format(css=CSS, toc=toc_html, body=body_html),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
