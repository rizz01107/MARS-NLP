#!/usr/bin/env bash
# Build reports/MARS_NLP_Final_Manuscript.pdf from the Markdown manuscript (needs pandoc and wkhtmltopdf).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=reports/.manuscript_tmp.html
pandoc reports/MARS_NLP_Final_Manuscript.md -f gfm -t html5 -s --metadata pagetitle="MARS-NLP manuscript" --css scripts/manuscript.css -o "$TMP"
sed -i 's#href="scripts/manuscript.css"#href="../scripts/manuscript.css"#' "$TMP"
wkhtmltopdf --quiet --enable-local-file-access -s A4 -T 20mm -B 20mm -L 18mm -R 18mm --footer-center "[page]" --footer-font-size 8 "$TMP" reports/MARS_NLP_Final_Manuscript.pdf
rm -f "$TMP"
echo "wrote reports/MARS_NLP_Final_Manuscript.pdf"
