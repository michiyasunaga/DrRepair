exit 0

conda activate DrRepair

################################# Training #################################
######### without text (pseudocode) #########
##Base
name="code-compiler--no-graph"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler/no-graph--dec-attn-all.yml \
    > out_spoc/${name}/log.txt 2>&1

##Base + graph
name="code-compiler--2l-graph"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_spoc/${name}/log.txt 2>&1


##Pretrain
name="code-compiler--2l-graph--pretrain"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    configs/base.yml  configs/data-spoc/err-data-extra.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_spoc/${name}/log.txt 2>&1

#then fine tune (Base + graph + pretrain)
name="code-compiler--2l-graph--finetune"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    -l out_spoc/code-compiler--2l-graph--pretrain/400000 \
    configs/base.yml  configs/data-spoc/err-data-orig-finetune.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_spoc/${name}/log.txt 2>&1


######### with text (pseudocode) #########
##Base
name="code-compiler-text--no-graph"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler-text/no-graph.yml \
    > out_spoc/${name}/log.txt 2>&1

##Base + graph + pretrain
name="code-compiler-text--2l-graph--finetune"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} train \
    -l out_spoc/code-compiler--2l-graph--pretrain/400000-add_text \
    configs/base.yml  configs/data-spoc/err-data-orig-finetune.yml \
    configs/model-code-compiler-text/2l-graph.yml \
    > out_spoc/${name}/log.txt 2>&1

################################# END Training #################################





################################# For Testing #################################

### Base + graph + pretrain ###
# (NOTE: run two servers for faster communication)
name="SERVER--code-compiler--2l-graph--finetune"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} server -p 8080 \
    -l out_spoc/code-compiler--2l-graph--finetune/550000 \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml

name="SERVER2--code-compiler--2l-graph--finetune"
mkdir -p out_spoc/${name}
CUDA_VISIBLE_DEVICES=1 python3 -u main_spoc.py -o ${name} server -p 8081 \
    -l out_spoc/code-compiler--2l-graph--finetune/550000 \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml



### Base + graph + pretrain (with pseudocode) ###
name="SERVER--code-compiler-text--2l-graph--finetune"
mkdir -p out_spoc/${name}
python3 -u main_spoc.py -o ${name} server -p 8082 \
    -l out_spoc/code-compiler-text--2l-graph--finetune/550000 \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler-text/2l-graph.yml

name="SERVER2--code-compiler-text--2l-graph--finetune"
mkdir -p out_spoc/${name}
CUDA_VISIBLE_DEVICES=1 python3 -u main_spoc.py -o ${name} server -p 8083 \
    -l out_spoc/code-compiler-text--2l-graph--finetune/550000 \
    configs/base.yml  configs/data-spoc/err-data-orig.yml \
    configs/model-code-compiler-text/2l-graph.yml

################################# END For Testing #################################
