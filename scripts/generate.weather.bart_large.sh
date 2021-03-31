#!/bin/bash

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=0
TMPDIR=/tmp
data=weather
model=bart_large
savedir=checkpoints/$data.$model
testpfx=test
genpfx=gen.$testpfx.constr
#genpfx=gen.$testpfx

fairseq-generate data-prep/$data \
  --user-dir . \
  --constr-dec \
  --encoder-json data-prep/$data/encoder.weather.json \
  --vocab-bpe data-prep/$data/vocab.gpt2.bpe \
  --gen-subset $testpfx.bpe \
  --path $savedir/checkpoint_best.pt \
  --dataset-impl raw \
  --max-tokens 2048 \
  --beam 5 \
  --max-len-a 2 --max-len-b 50 \
  > $savedir/$genpfx.txt

grep ^H- $savedir/$genpfx.txt | sort -n -k 2 -t - | awk -F '\t' '{print $3}' > $savedir/$genpfx.hyp.txt

python scripts/gpt2_bpe_decoder.py \
  data-prep/$data/encoder.$data.json \
  data-prep/$data/vocab.gpt2.bpe \
  $savedir/$genpfx.hyp.txt \
  $savedir/$genpfx.hyp.norm.txt

bash scripts/measure_scores.sh $savedir/$genpfx.hyp.norm.txt data-prep/$data/$testpfx.mr-ar.ar
python scripts/compute_tree_acc.py $savedir/$genpfx.hyp.norm.txt data-prep/$data/$testpfx.mr-ar.mr

if [[ "$genpfx" == *".constr"* ]]; then
  basepfx=$(echo $genpfx | sed 's/.constr//')
  bash scripts/replfail.sh \
    $savedir/$genpfx.txt \
    $savedir/$genpfx.hyp.norm.txt \
    $savedir/$basepfx.hyp.norm.txt \
    > $savedir/$genpfx.hyp.norm.replfail.txt
  bash scripts/measure_scores.sh $savedir/$genpfx.hyp.norm.replfail.txt data-prep/$data/$testpfx.mr-ar.ar
  python scripts/compute_tree_acc.py $savedir/$genpfx.hyp.norm.replfail.txt data-prep/$data/$testpfx.mr-ar.mr
fi
