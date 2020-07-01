#!/bin/sh

cd data

wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--codeforce--deepfix-style.zip
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--codeforce--spoc-style.zip
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--orig-deepfix.zip
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--orig-spoc.zip

unzip err-data-compiler--auto-corrupt--codeforce--deepfix-style.zip
unzip err-data-compiler--auto-corrupt--codeforce--spoc-style.zip
unzip err-data-compiler--auto-corrupt--orig-deepfix.zip
unzip err-data-compiler--orig-spoc.zip
