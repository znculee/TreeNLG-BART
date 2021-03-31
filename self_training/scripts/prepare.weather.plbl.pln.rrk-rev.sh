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
beam_size=5

pct=pct-$1
dest=$pct/shuf.plbl.pln.rrk-rev.itr$2

if [[ "$2" == "1" ]]; then
  ckpt=self_training/checkpoints/$data/$pct/shuf.lbl.pln.$model/checkpoint_best.pt
  REVMDLDIR=self_training/checkpoints/$data/$pct/shuf.lbl.pln.rrk-rev.rev.$model
else
  itrprev=$(bc <<< "$2-1")
  ckpt=self_training/checkpoints/$data/$pct/shuf.lbl.pln.rrk-rev.itr$itrprev.$model/checkpoint_best.pt
  REVMDLDIR=self_training/checkpoints/$data/$pct/shuf.lbl.pln.rrk-rev.rev.itr$itrprev.$model
fi

mkdir -p $prep/$dest

beam_repeat () {
  awk -v n="$beam_size" '{for(i=0;i<n;i++)print}'
}

rmtreeinfo () {
  sed 's/\[\S\+//g;s/\]//g' | awk '{$1=$1;print}'
}

for split in train valid; do
  tmp=$(mktemp -d prepare.weather.plbl.rrk-rev.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
  fairseq-generate $prep/$pct/shuf.lbl \
    --user-dir . \
    --encoder-json data-prep/$data/encoder.weather.json \
    --vocab-bpe data-prep/$data/vocab.gpt2.bpe \
    --gen-subset $split.ulbl.bpe \
    --path $ckpt \
    --dataset-impl raw \
    --max-tokens 2048 \
    --beam 5 --nbest $beam_size \
    --max-len-a 2 --max-len-b 50 \
    > $tmp/gen.txt
  grep ^H- $tmp/gen.txt | awk -F '\t' '{print $3}' > $tmp/hyp.bpe
  grep ^S- $tmp/gen.txt | awk -F '\t' '{print $2}' > $tmp/src.bpe
  for x in src hyp; do
    python scripts/gpt2_bpe_decoder.py \
      data-prep/$data/encoder.$data.json \
      data-prep/$data/vocab.gpt2.bpe \
      $tmp/$x.bpe \
      $tmp/$x.norm
  done
  cat $tmp/hyp.norm | rmtreeinfo  > "$tmp/test.ar-mr.ar"
  cat $tmp/src.norm | beam_repeat > "$tmp/test.ar-mr.mr"
  ln -s $(readlink -f $prep/common/dict.gpt2.txt) $tmp/dict.ar.txt
  ln -s $(readlink -f $prep/common/dict.gpt2.txt) $tmp/dict.mr.txt
  for lang in ar mr; do
    echo "creating $tmp/test.bpe.ar-mr.$lang..."
    python bpe_encoder/multiprocessing_bpe_encoder.py \
      --encoder-json "$prep/common/encoder.weather.json" \
      --vocab-bpe    "$prep/common/vocab.gpt2.bpe"       \
      --inputs       "$tmp/test.ar-mr.$lang"             \
      --outputs      "$tmp/test.bpe.ar-mr.$lang"         \
      --workers      60                                  \
      --keep-empty
  done
  fairseq-generate $tmp \
    --user-dir . \
    --gen-subset "test.bpe" \
    --path $REVMDLDIR/checkpoint_best.pt \
    --dataset-impl raw \
    --score-reference | \
    grep ^H- | sort -n -k 2 -t - | \
    awk -F '\t' '{print $2}' \
    > $tmp/score
  paste \
    <(sed 's/^/-/;s/^--//' $tmp/score) \
    <(grep ^S- $tmp/gen.txt | beam_repeat) \
    <(grep ^H- $tmp/gen.txt) \
    <(grep ^P- $tmp/gen.txt) | \
    awk -v n="$beam_size" 'BEGIN{OFS="\t"}{print int((NR-1)/n),$0}' | \
    sort -g -k 1,1 -k 2,2 | \
    awk -v n="$beam_size" 'NR%n==1 {print}' | \
    awk -F '\t' '{printf"%s\t%s\n%s\t%s\t%s\n%s\t%s\n",$3,$4,$5,$6,$7,$8,$9}' | \
    grep ^H- | sort -n -k 2 -t - | awk -F '\t' '{print $3}' \
    > $prep/$dest/$split.bpe.$src-$tgt.$tgt
  rm -rf $tmp
done

ln -s "../../common/train.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/train.bpe.$src-$tgt.$src"
ln -s "../../common/valid.ulbl.bpe.$src-$tgt.$src" "$prep/$dest/valid.bpe.$src-$tgt.$src"
ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$src.txt"
ln -s "../../common/dict.gpt2.txt"                 "$prep/$dest/dict.$tgt.txt"
ln -s "../../common/test.bpe.$src-$tgt.$src"       "$prep/$dest/test.bpe.$src-$tgt.$src"
ln -s "../../common/test.bpe.$src-$tgt.$tgt"       "$prep/$dest/test.bpe.$src-$tgt.$tgt"
