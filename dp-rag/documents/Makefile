all: report.md references.bib
	pandoc -s -o report.pdf --pdf-engine=xelatex --filter pandoc-crossref --citeproc report.md