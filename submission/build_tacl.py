#!/usr/bin/env python3
"""Convert the anonymized Markdown manuscript into a TACL-style LaTeX source."""
from pathlib import Path
import re, subprocess, sys, shutil

source = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
text = source.read_text(encoding="utf-8")

# Extract title and abstract/keywords from the manuscript without author metadata.
m = re.match(r"^# (.+?)\n", text)
if not m:
    raise SystemExit("Could not find manuscript title")
title = m.group(1).strip()
text = text[m.end():]
abstract_match = re.search(r"(?ms)^## Abstract\s*\n(.*?)(?=^## Keywords\s*$)", text)
keywords_match = re.search(r"(?ms)^## Keywords\s*\n(.*?)(?=^## 1\. Introduction\s*$)", text)
if not abstract_match or not keywords_match:
    raise SystemExit("Could not locate abstract/keywords sections")
abstract_md = abstract_match.group(1).strip()
keywords = keywords_match.group(1).strip().replace("\n", " ").replace(";", ",")
text = text[re.search(r"(?m)^## 1\. Introduction\s*$", text).start():]

# Convert Markdown body with Pandoc; retain tables, math, and figures.
tmp = outdir / "body.md"
tmp.write_text(text, encoding="utf-8")
body = subprocess.check_output([
    "pandoc", str(tmp), "--from=markdown+pipe_tables+tex_math_dollars",
    "--to=latex", "--wrap=none", "--resource-path=."
], text=True, cwd=Path.cwd())
body = body.replace(r"\includegraphics{../figures/mars_nlp_pipeline.png}",
                    r"\includegraphics[width=0.9\linewidth]{../../figures/mars_nlp_pipeline.png}")
# TACL uses a two-column layout; convert Pandoc longtable environments safely.
# Parse the column-spec argument with balanced braces (Pandoc can emit nested braces).
def convert_longtables(latex):
    out = []
    pos = 0
    marker = r"\begin{longtable}"
    while True:
        idx = latex.find(marker, pos)
        if idx < 0:
            out.append(latex[pos:])
            break
        out.append(latex[pos:idx])
        spec_start = latex.find("{", idx + len(marker))
        if spec_start < 0:
            out.append(latex[idx:])
            break
        depth = 0
        spec_end = None
        for k in range(spec_start, len(latex)):
            if latex[k] == "{":
                depth += 1
            elif latex[k] == "}":
                depth -= 1
                if depth == 0:
                    spec_end = k
                    break
        if spec_end is None:
            out.append(latex[idx:])
            break
        colspec = latex[spec_start + 1:spec_end]
        out.append(r"\begin{table*}[t]\centering\small\begin{tabular}{" + colspec + "}")
        pos = spec_end + 1
    latex = "".join(out)
    for marker in (r"\endhead", r"\endfirsthead", r"\endfoot", r"\endlastfoot"):
        latex = latex.replace(marker, "")
    latex = latex.replace(r"\end{longtable}", r"\end{tabular}\end{table*}")
    return latex

body = convert_longtables(body)
# Remove accidental Markdown-only raw links around the pipeline if pandoc emits paths differently.
(outdir / "body.tex").write_text(body, encoding="utf-8")

tex = r"""\documentclass[11pt]{article}
\providecommand{\tightlist}{}
\usepackage{tacl2021v1}
\usepackage{times}
\usepackage{latexsym}
\usepackage{url}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{microtype}
\title{TITLE_PLACEHOLDER}
\author{Anonymous}
\date{}
\begin{document}
\maketitle
\begin{abstract}
ABSTRACT_PLACEHOLDER
\end{abstract}
\noindent\textbf{Keywords:} KEYWORDS_PLACEHOLDER

BODY_PLACEHOLDER
\end{document}
"""
tex = tex.replace("TITLE_PLACEHOLDER", title.replace("&", r"\&"))
tex = tex.replace("ABSTRACT_PLACEHOLDER", abstract_md.replace("&", r"\&"))
tex = tex.replace("KEYWORDS_PLACEHOLDER", keywords.replace("&", r"\&"))
tex = tex.replace("BODY_PLACEHOLDER", body)
(outdir / "main.tex").write_text(tex, encoding="utf-8")
# Keep the PDF output path deterministic for CI.
print(f"Prepared TACL-style LaTeX source for: {title}")
