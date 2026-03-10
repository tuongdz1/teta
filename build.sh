#!/usr/bin/env bash
set -euo pipefail

rm -f \
  *.aux *.toc *.out *.log *.bbl *.blg *.bcf *.run.xml *.fls .fdb_latexmk \
  main.pdf

xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

echo "✅ Built openbook-full/main.pdf"

