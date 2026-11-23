# Builds GroupXX_slides.pdf (needs pdflatex + beamer).
# Uses the graphs in ../report/figures.
SLIDES = GroupXX_slides

all: $(SLIDES).pdf

$(SLIDES).pdf: $(SLIDES).tex
	pdflatex -interaction=nonstopmode $(SLIDES).tex > /dev/null
	pdflatex -interaction=nonstopmode $(SLIDES).tex | grep -E "Output written|Error" || true

clean:
	rm -f *.aux *.log *.out *.nav *.snm *.toc *.vrb

.PHONY: all clean
