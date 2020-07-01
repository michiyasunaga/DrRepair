exit 0

conda activate DrRepair

################################# Training #################################
### No compiler ###
name="code-only"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-only/no-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1


### Base ###
name="code-compiler--no-graph"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-compiler/no-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1


### Base + graph ###
name="code-compiler--2l-graph"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1


### Base + graph + pretrain ###
#first, pretrain
name="code-compiler--2l-graph--pretrain"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    configs/base.yml  configs/data-deepfix/err-data-extra.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1
#then fine tune
name="code-compiler--2l-graph--finetune"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    -l out_deepfix/code-compiler--2l-graph--pretrain/400000 \
    configs/base.yml  configs/data-deepfix/err-data-orig-finetune.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1

################################# END Training #################################




################################# For Testing #################################
### No compiler ###
name="SERVER--code-only"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} server -p 8090 \
    -l out_deepfix/code-only/150000 \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-only/no-graph--dec-attn-all.yml


### Base ###
name="SERVER--code-compiler--no-graph"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} server -p 8091 \
    -l out_deepfix/code-compiler--no-graph/150000 \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-compiler/no-graph--dec-attn-all.yml


### Base + graph ###
name="SERVER--code-compiler--2l-graph"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} server -p 8092 \
    -l out_deepfix/code-compiler--2l-graph/150000 \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml


### Base + graph + pretrain ###
name="SERVER--code-compiler--2l-graph--finetune"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} server -p 8093 \
    -l out_deepfix/code-compiler--2l-graph--finetune/550000 \
    configs/base.yml  configs/data-deepfix/err-data-orig.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml

################################# END For Testing #################################
