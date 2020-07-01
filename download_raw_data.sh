#!/bin/sh

mkdir raw_data
cd raw_data

wget https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/codeforce_data.zip
wget https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/deepfix_data.zip
wget https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/spoc_data.zip

unzip codeforce_data.zip
unzip deepfix_data.zip
unzip spoc_data.zip


cd ../
cd evaluation/spoc

wget https://nlp.stanford.edu/projects/myasu/DrRepair/evaluation/spoc/translation_preds_test.zip
unzip translation_preds_test.zip
