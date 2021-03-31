#!/bin/bash

cd $(dirname $0)/../..

data=weather
src=mr
tgt=ar
prep=self_training/data-prep/$data

orig=pct-$1/shuf.lbl
dest=pct-$1/shuf.lbl.pln
mkdir -p $prep/$dest

ln -s $(readlink -f $prep/$orig/train.$src-$tgt.$src) $prep/$dest/train.$src-$tgt.$src
sed 's/\[\S\+//g;s/\]//g' $prep/$orig/train.$src-$tgt.$tgt | awk '{$1=$1;print}' > $prep/$dest/train.$src-$tgt.$tgt

ln -s $(readlink -f $prep/$orig/valid.$src-$tgt.$src) $prep/$dest/valid.$src-$tgt.$src
sed 's/\[\S\+//g;s/\]//g' $prep/$orig/valid.$src-$tgt.$tgt | awk '{$1=$1;print}' > $prep/$dest/valid.$src-$tgt.$tgt

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

ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$src.txt"
ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$tgt.txt"
