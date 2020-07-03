exit 1


#################### SPoC style ####################
# auto corrupt
out_dir="err-data-compiler--auto-corrupt--additional-codeforce--spoc-style"
mkdir ${out_dir}
codeforce_root="../raw_data/codeforce_data"
log_root_dir="log--auto-corrupt--additional-codeforce"
for entry in "$codeforce_root"/*  #entry is full path
do
  probid=`basename $entry` #(e.g. 123A)
  python3 -u gen-err-dataset--auto-corrupt--spoc-style.py \
   --input-code-dir ${codeforce_root}/${probid} \
   ${log_root_dir}/cpp-log--addtnl-codeforce--spoc-style \
   ${out_dir}/${probid}
done


# dev data
python3 -u ../model/scripts/gen-err-dev-data.py \
    ${out_dir} \
    err-dev-compiler--for-pretrain-spoc/err-dev.all.jsonl \
    pretrain-spoc

cd err-dev-compiler--for-pretrain-spoc
shuf ./err-dev.all.jsonl > ./err-dev.all.shuffled.jsonl
head -n 1000  ./err-dev.all.shuffled.jsonl > ./err-dev.1000.jsonl
