LATEXCMD=/usr/texbin/lualatex
BIBTEXCMD=/usr/texbin/bibtex

proposal.pdf: proposal.tex
	$(LATEXCMD) proposal.tex
	$(BIBTEXCMD) proposal
	$(LATEXCMD) proposal.tex
