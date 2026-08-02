# PenBox-DMAS paper build.
#
# The paper cannot be built without first running the model: PAPER.tex reads
# numbers.tex and table_groups.tex, both generated. That dependency is the
# point -- a stale number cannot survive a build.

PY      := python3
LATEX   := pdflatex -interaction=nonstopmode
GEN     := numbers.tex table_groups.tex

.PHONY: all paper model verify clean distclean

all: verify

paper: PAPER.pdf

$(GEN) figs/.stamp: dmas_model.py
	$(PY) dmas_model.py
	@touch figs/.stamp

model: $(GEN)

PAPER.pdf: PAPER.tex $(GEN) figs/.stamp
	$(LATEX) PAPER.tex >/dev/null
	$(LATEX) PAPER.tex >/dev/null
	$(LATEX) PAPER.tex >/dev/null
	@echo "built: $$(pdfinfo PAPER.pdf | awk '/^Pages/{print $$2}') pages"

verify: PAPER.pdf
	$(PY) verify.py

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f PAPER.pdf $(GEN) results.json figs/*.pdf figs/.stamp
