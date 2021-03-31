#!/bin/bash

cd $(dirname $0)/../..

data=weather
src=mr
tgt=ar
prep=self_training/data-prep/$data

orig=pct-$1/shuf.lbl
dest=pct-$1/shuf.lbl.rev
mkdir -p $prep/$dest

ln -s $(readlink -f $prep/$orig/train.$src-$tgt.$src) $prep/$dest/train.$tgt-$src.$src
sed 's/\[\S\+//g;s/\]//g' $prep/$orig/train.$src-$tgt.$tgt | awk '{$1=$1;print}' > $prep/$dest/train.$tgt-$src.$tgt

ln -s $(readlink -f $prep/$orig/valid.$src-$tgt.$src) $prep/$dest/valid.$tgt-$src.$src
sed 's/\[\S\+//g;s/\]//g' $prep/$orig/valid.$src-$tgt.$tgt | awk '{$1=$1;print}' > $prep/$dest/valid.$tgt-$src.$tgt

for split in "train" "valid"; do
  for lang in $src $tgt; do
    echo "creating $prep/$dest/$split.bpe.$src-$tgt.$lang..."
    python bpe_encoder/multiprocessing_bpe_encoder.py \
      --encoder-json "$prep/common/encoder.weather.json"      \
      --vocab-bpe    "$prep/common/vocab.gpt2.bpe"            \
      --inputs       "$prep/$dest/$split.$tgt-$src.$lang"     \
      --outputs      "$prep/$dest/$split.bpe.$tgt-$src.$lang" \
      --workers      60                                       \
      --keep-empty
  done
done

ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$src.txt"
ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$tgt.txt"
