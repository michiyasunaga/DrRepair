exit(1);
#----------------------------------#
conda activate DrRepair



######## Prepare ########
repo_root="../../../.."
program_data_root=${repo_root}"/raw_data/deepfix_data"
test_split_root=${repo_root}"/data/err-data-compiler--auto-corrupt--orig-deepfix/bin4"


######## Run test repair ########
#NOTE: run the following in the same directory as this file!!

#NOTE: Pick the name you are testing. Also change the http address below to the server running your model (Get the IP address by running `ifconfig` on the server hosting your model. Port is configured in the command you ran in run_deepfix.sh)

# name="code-only"
# name="code-compiler--no-graph"
name="code-compiler--2l-graph"
# name="code-compiler--2l-graph--finetune"

mkdir -p out/${name}/log
cd out/${name}
for entry in ${test_split_root}/*
do
  probid=`basename $entry` #(e.g. prob11)
  python3 -u ../../test_deepfix.py \
  --input-code-dir ${program_data_root}/${probid}/erroneous \
  --repairer-server  http://172.24.67.150:8090/pred \
  > log/${probid}.out.txt
done

######### Get accuracy ########
python3 -u ../../collate_deepfix.py

#Get back
cd ../../





###################################### for slurm ######################################
name="code-only"
# name="code-compiler--no-graph"
# name="code-compiler--2l-graph"
# name="code-compiler--2l-graph--finetune"

mkdir -p out/${name}/log
cd out/${name}

for entry in ${test_split_root}/*  #entry is full path
do
  probid=`basename $entry` #(e.g. prob11)
  ${repo_root}/nlprun.py --hold -o log/${probid}.out.txt -c 4 -a DrRepair -q john \
  $'python3 -u ../../test_deepfix.py \
  --input-code-dir '"${program_data_root}/${probid}/erroneous"' \
  --repairer-server  http://172.24.67.150:8090/pred '
done

#Get accuracy
python3 -u ../../collate_deepfix.py

#Get back
cd ../../
