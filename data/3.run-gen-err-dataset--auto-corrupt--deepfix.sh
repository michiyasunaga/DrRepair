exit 1


data_dir="err-data-compiler--auto-corrupt--orig-deepfix"
dev_dir="err-dev-compiler--for-deepfix"
vocab_dir="err-vocab-compiler--for-deepfix"

mkdir ${data_dir}
mkdir ${dev_dir}
mkdir ${vocab_dir}


#################### original deepfix data ####################
# auto corrupt
program_data_root="../raw_data/deepfix_data"
log_root_dir="log--auto-corrupt--orig-deepfix"
for entry in "$program_data_root"/*  #entry is full path
do
  probid=`basename $entry` #(e.g. prob11)
  python3 -u gen-err-dataset--auto-corrupt--deepfix-style.py \
   --input-code-dir ${program_data_root}/${probid}/correct ${log_root_dir}/cpp-log--orig-deepfix \
   ${data_dir}/${probid}
done

# Compute vocab
python3 -u compute-err-dataset-vocab--compiler--for-deepfix.py  ${data_dir}  ${vocab_dir}


# dev data
python3 -u ../model/scripts/gen-err-dev-data.py \
    ${data_dir}/bin4 \
    ${dev_dir}/err-dev.all.jsonl \
    deepfix

cd ${dev_dir}
shuf ./err-dev.all.jsonl > ./err-dev.all.shuffled.jsonl
head -n 2000  ./err-dev.all.shuffled.jsonl > ./err-dev.2000.jsonl




#################### additional data (pre-train) ####################
out_dir="err-data-compiler--auto-corrupt--additional-codeforce--deepfix-style"
mkdir ${out_dir}
program_data_root="../raw_data/codeforce_data"
log_root_dir="log--auto-corrupt--orig-deepfix"

for entry in "$program_data_root"/*  #entry is full path
do
  probid=`basename $entry` #(e.g. prob11)
  python3 -u gen-err-dataset--auto-corrupt--deepfix-style.py \
   --input-code-dir ${program_data_root}/${probid} --n-samples 30 \
  ${log_root_dir}/cpp-log--addtnl-codeforce--deepfix-style \
  ${out_dir}/${probid}
done
