exit(1);
#----------------------------------#
conda activate DrRepair



######## Prepare ########
repo_root="../../../.."


######## Run test program synthesis ########
#NOTE: run the following in the same directory as this file!!

#NOTE: Pick the name you are testing. Also change the http address below to the server running your model (Get the IP address by running `ifconfig` on the server hosting your model. Port is configured in the command you ran in run_spoc.sh)

name="code-compiler--2l-graph--finetune"
# name="code-compiler-text--2l-graph--finetune"

INPUT=translation_preds_test/testw    #change to testp if you want to evaluate on testp
N=$(tail -n+2 ${INPUT}.tsv | cut -f 3-6 | uniq | wc -l)  # Count the number of programs
interval=10

mkdir -p out_testw/${name}/log        #change to testp if you want to evaluate on testp
cd out_testw/${name}                  #change to testp if you want to evaluate on testp

i=1
while [[ $i -le $N ]]; do
  python -u ../../test_spoc.py -p 100 \
  --compile-budget 100 --n-parallel ${interval} \
  --repairer-server  http://172.24.67.150:8080/pred,http://172.24.67.150:8081/pred \
  ../../${INPUT} $i
  i=$(($i + ${interval}))
done



######### Get accuracy ########
python3 -u ../../collate_spoc.py

#Get back
cd ../../





###################################### for slurm ######################################
#NOTE: run the following in the same directory as this file!!
# name="code-compiler--2l-graph--finetune"
name="code-compiler-text--2l-graph--finetune"

INPUT=translation_preds_test/testw    #change to testp if you want to evaluate on testp
N=$(tail -n+2 ${INPUT}.tsv | cut -f 3-6 | uniq | wc -l)  # Count the number of programs
interval=10

mkdir -p out_testw/${name}/log        #change to testp if you want to evaluate on testp
cd out_testw/${name}                  #change to testp if you want to evaluate on testp


i=1
while [[ $i -le $N ]]; do
  ${repo_root}/nlprun.py --hold -o log/$i.out.txt -c 2 -a DrRepair -q john \
  $'python -u ../../test_spoc.py -p 100 \
  --compile-budget 100 --n-parallel '"${interval}"' \
  --repairer-server  http://172.24.67.150:8082/pred,http://172.24.67.150:8083/pred \
  ../../'"${INPUT}"' '"$i"' '
  i=$(($i + ${interval}))
done


#Get accuracy
python3 -u ../../collate_spoc.py

#Get back
cd ../../
