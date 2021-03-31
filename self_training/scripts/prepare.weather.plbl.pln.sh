#!/bin/bash

cd $(dirname $0)/../..

export CUDA_VISIBLE_DEVICES=$3
TMPDIR=/tmp
src=mr
tgt=ar
data=weather
model=bart_large
orig=data-prep/$data
prep=self_training/data-prep/$data

pct=pct-$1
dest=$pct/shuf.plbl.pln.itr$2

if [[ "$2" == "1" ]]; then
  ckpt=self_training/checkpoints/$data/$pct/shuf.lbl.pln.$model/checkpoint_best.pt
else
  itrprev=$(bc <<< "$2-1")
  ckpt=self_training/checkpoints/$data/$pct/shuf.lbl.pln.itr$itrprev.$model/checkpoint_best.pt
fi

mkdir -p $prep/$dest

fairseq-generate $prep/$pct/shuf.lbl \
  --user-dir . \
  --encoder-json data-prep/$data/encoder.weather.json \
  --vocab-bpe data-prep/$data/vocab.gpt2.bpe \
  --gen-subset train.ulbl.bpe \
  --path $ckpt \
  --dataset-impl raw \
  --max-tokens 2048 \
  --beam 5 \
  --max-len-a 2 --max-len-b 50 | \
  grep ^H- | sort -n -k 2 -t - | awk -F '\t' '{print $3}' \
  > $prep/$dest/train.bpe.$src-$tgt.$tgt

fairseq-generate $prep/$pct/shuf.lbl \
  --user-dir . \
  --encoder-json data-prep/$data/encoder.weather.json \
  --vocab-bpe data-prep/$data/vocab.gpt2.bpe \
  --gen-subset valid.ulbl.bpe \
  --path $ckpt \
  --dataset-impl raw \
  --max-tokens 2048 \
  --beam 5 \
  --max-len-a 2 --max-len-b 50 | \
  grep ^H- | sort -n -k 2 -t - | awk -F '\t' '{print $3}' \
  > $prep/$dest/valid.bpe.$src-$tgt.$tgt

ln -s "../../common/train.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/train.bpe.$src-$tgt.$src"
ln -s "../../common/valid.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/valid.bpe.$src-$tgt.$src"
ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$src.txt"
ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$tgt.txt"
ln -s "../../common/test.bpe.$src-$tgt.$src"       "$prep/$dest/test.bpe.$src-$tgt.$src"
ln -s "../../common/test.bpe.$src-$tgt.$tgt"       "$prep/$dest/test.bpe.$src-$tgt.$tgt"
