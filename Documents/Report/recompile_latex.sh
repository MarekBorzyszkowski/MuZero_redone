#!/bin/bash
FILE=$1.tex
if [[ ! -f "$FILE" ]]; then
    echo "No file named $1.tex! QUITTING!"
    exit 1
fi
echo "deleting $1.aux $1.bbl $1.bcf $1.blg $1.lof $1.log $1.lol $1.lot $1.out $1.pdf $1.run.xml $1.toc"
rm -f $1.aux $1.bbl $1.bcf $1.blg $1.lof $1.log $1.lol $1.lot $1.out $1.pdf $1.run.xml $1.toc 
echo "deleting complete"
pdflatex $1.tex
biber $1
pdflatex $1.tex
pdflatex $1.tex

