exit 1


data_dir="err-data-compiler--orig-spoc"
dev_dir="err-dev-compiler--for-SPoC"
vocab_dir="err-vocab-compiler--for-SPoC"

mkdir ${data_dir}
mkdir ${dev_dir}
mkdir ${vocab_dir}



## run everything
for i in {1..5}; do
  NAME=../raw_data/spoc_data/translation_preds/split${i}/yay
  N=$(tail -n+2 ${NAME}.tsv | cut -f 3-6 | uniq | wc -l)
  probno=1
  while [[ ${probno} -le $N ]]; do
    python3 gen-err-dataset--orig-spoc.py --num-preds 30  \
     ${NAME} ${probno} \
     ${data_dir}/s${i}
    probno=$((${probno} + 1))
  done
done


# compute vocab
python3 compute-err-dataset-vocab--compiler--for-SPoC.py  ${data_dir}  ${vocab_dir}


# get dev data
python3 ../model/scripts/gen-err-dev-data.py  ${data_dir}/s5  ${dev_dir}/err-dev.all.jsonl  spoc
python3 subset_dev_data.py
