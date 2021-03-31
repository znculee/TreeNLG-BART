#!/bin/bash

cd $(dirname $0)/../..

data=weather
src=mr
tgt=ar
orig=data-prep/$data
prep=self_training/data-prep/$data
cpsrc=../TreeNLG/self_training/data-prep/$data

mkdir -p $prep/common

cp "$orig/dict.gpt2.txt"              "$prep/common/dict.gpt2.txt"
cp "$orig/vocab.gpt2.bpe"             "$prep/common/vocab.gpt2.bpe"
cp "$orig/encoder.weather.json"       "$prep/common/encoder.weather.json"
cp "$orig/test.bpe.$src-$tgt.$src"    "$prep/common/test.bpe.$src-$tgt.$src"
cp "$orig/test.bpe.$src-$tgt.$tgt"    "$prep/common/test.bpe.$src-$tgt.$tgt"
cp "$cpsrc/ulbl/train.$src-$tgt.$src" "$prep/common/train.ulbl.$src-$tgt.$src"
cp "$cpsrc/ulbl/valid.$src-$tgt.$src" "$prep/common/valid.ulbl.$src-$tgt.$src"

for split in "train" "valid"; do
  python bpe_encoder/multiprocessing_bpe_encoder.py \
    --encoder-json "$prep/common/encoder.weather.json"           \
    --vocab-bpe    "$prep/common/vocab.gpt2.bpe"                 \
    --inputs       "$prep/common/$split.ulbl.$src-$tgt.$src"     \
    --outputs      "$prep/common/$split.ulbl.bpe.$src-$tgt.$src" \
    --workers      60                                            \
    --keep-empty
done

for p in 01 02 05 10 20 50 1c; do
  dest=pct-$p/shuf.lbl
  mkdir -p $prep/$dest

  cp "$cpsrc/$dest/train.$src-$tgt.$src" "$prep/$dest/train.$src-$tgt.$src"
  cp "$cpsrc/$dest/train.$src-$tgt.$tgt" "$prep/$dest/train.$src-$tgt.$tgt"
  cp "$cpsrc/$dest/valid.$src-$tgt.$src" "$prep/$dest/valid.$src-$tgt.$src"
  cp "$cpsrc/$dest/valid.$src-$tgt.$tgt" "$prep/$dest/valid.$src-$tgt.$tgt"

  for split in "train" "valid"; do
    for lang in $src $tgt; do
      echo "creating $prep/$dest/$split.bpe.$src-$tgt.$lang..."
      python bpe_encoder/multiprocessing_bpe_encoder.py \
        --encoder-json "$prep/common/encoder.weather.json"      \
        --vocab-bpe    "$prep/common/vocab.gpt2.bpe"            \
        --inputs       "$prep/$dest/$split.$src-$tgt.$lang"     \
        --outputs      "$prep/$dest/$split.bpe.$src-$tgt.$lang" \
        --workers      60                                       \
        --keep-empty
    done
  done

  ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$src.txt"
  ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$tgt.txt"
  ln -s "../../common/test.bpe.$src-$tgt.$src"       "$prep/$dest/test.bpe.$src-$tgt.$src"
  ln -s "../../common/test.bpe.$src-$tgt.$tgt"       "$prep/$dest/test.bpe.$src-$tgt.$tgt"
  ln -s "../../common/train.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/train.ulbl.bpe.$src-$tgt.$src"
  ln -s "../../common/valid.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/valid.ulbl.bpe.$src-$tgt.$src"
done
