#!/bin/bash

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=0
TMPDIR=/tmp

src=mr
tgt=ar
data=weather
model=bart_large

savedir=checkpoints/$data.$model
mkdir -p $savedir

fairseq-train data-prep/$data \
  --restore-file cache/$model/model.pt \
  --max-epoch 500 --patience 20 \
  --max-tokens 2048 \
  --task translation --arch $model \
  --source-lang $src --target-lang $tgt \
  --train-subset train.bpe --valid-subset valid.bpe \
  --truncate-source \
  --layernorm-embedding \
  --share-all-embeddings \
  --share-decoder-input-output-embed \
  --reset-optimizer --reset-lr-scheduler --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 1e-05 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
  --clip-norm 0 \
  --lr-scheduler polynomial_decay --lr 3e-05 \
  --warmup-updates 1000 \
  --skip-invalid-size-inputs-valid-test \
  --find-unused-parameters \
  --dataset-impl raw \
  --save-dir $savedir \
  --no-epoch-checkpoints \
  | tee $savedir/log.txt
