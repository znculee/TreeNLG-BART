#!/bin/bash

cd $(dirname $0)/../..

data=weather
src=mr
tgt=ar
prep=self_training/data-prep/$data

if [[ "$2" == "0" ]]; then
  exit
fi

pct=pct-$1
itr=itr$2

orig=$pct/shuf.plbl.rrk-rev.$itr
dest=$pct/shuf.plbl.rrk-rev.rev.$itr
mkdir -p $prep/$dest

for split in train valid; do
ln -s $(readlink -f $prep/$orig/$split.bpe.$src-$tgt.$src) $prep/$dest/$split.bpe.$tgt-$src.$src
python scripts/gpt2_bpe_decoder.py \
  data-prep/$data/encoder.$data.json \
  data-prep/$data/vocab.gpt2.bpe \
  $prep/$orig/$split.bpe.$src-$tgt.$tgt \
  $prep/$dest/$split.norm.$src-$tgt.$tgt
sed 's/\[\S\+//g;s/\]//g' $prep/$dest/$split.norm.$src-$tgt.$tgt | awk '{$1=$1;print}' > $prep/$dest/$split.$tgt-$src.$tgt
done

for split in "train" "valid"; do
  echo "creating $prep/$dest/$split.bpe.$src-$tgt.$tgt..."
  python bpe_encoder/multiprocessing_bpe_encoder.py \
    --encoder-json "$prep/common/encoder.weather.json"      \
    --vocab-bpe    "$prep/common/vocab.gpt2.bpe"            \
    --inputs       "$prep/$dest/$split.$tgt-$src.$tgt"     \
    --outputs      "$prep/$dest/$split.bpe.$tgt-$src.$tgt" \
    --workers      60                                       \
    --keep-empty
done

ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$src.txt"
ln -s "../../common/dict.gpt2.txt" "$prep/$dest/dict.$tgt.txt"
