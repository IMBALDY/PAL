# PAL Presentation

This directory contains a standalone LaTeX Beamer slide deck for presenting the
current PAL framework in English.

## Files

- `main.tex`: main slide deck

## Build

If `latexmk` is available:

```bash
cd presentation
latexmk -pdf main.tex
```

Or with `pdflatex`:

```bash
cd presentation
pdflatex main.tex
pdflatex main.tex
```

The deck is designed to explain the current PAL pipeline and its existing
innovations without modifying the training code.
