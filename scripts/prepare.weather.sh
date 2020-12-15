#!/bin/bash

cd $(dirname $0)/..

src=mr
tgt=ar
prep=data-prep/weather
orig=data/weather

mkdir -p $prep

echo -e "show data sample...\n"
awk -F '\t' 'NR==1 {print $1,$2}' "$orig/train.tsv" ; echo ""
awk -F '\t' 'NR==1 {print $3}' "$orig/train.tsv" ; echo ""
awk -F '\t' 'NR==1 {print $4}' "$orig/train.tsv" ; echo ""
awk -F '\t' 'NR==1 {print $4}' "$orig/train.tsv" | \
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}' ; echo ""

echo "creating train..."
awk -F '\t' '{printf " %s\n",$3}' "$orig/train.tsv" > "$prep/train.$src-$tgt.$src"
awk -F '\t' '{printf " %s\n",$4}' "$orig/train.tsv" > "$prep/train.$src-$tgt.$tgt"
echo "creating valid..."
awk -F '\t' '{printf " %s\n",$3}' "$orig/val.tsv"   > "$prep/valid.$src-$tgt.$src"
awk -F '\t' '{printf " %s\n",$4}' "$orig/val.tsv"   > "$prep/valid.$src-$tgt.$tgt"
echo "creating test..."
awk -F '\t' '{printf " %s\n",$3}' "$orig/test.tsv"  > "$prep/test.$src-$tgt.$src"
awk -F '\t' '{printf " %s\n",$4}' "$orig/test.tsv"  > "$prep/test.$src-$tgt.$tgt"
echo "creating test.disc..."
awk -F '\t' '{printf " %s\n",$3}' "$orig/disc_test.tsv"  > "$prep/test.disc.$src-$tgt.$src"
awk -F '\t' '{printf " %s\n",$4}' "$orig/disc_test.tsv"  > "$prep/test.disc.$src-$tgt.$tgt"

cp cache/gpt2_bpe/encoder.json $prep/encoder.gpt2.json
cp cache/gpt2_bpe/vocab.bpe $prep/vocab.gpt2.bpe
cp cache/gpt2_bpe/dict.txt $prep/dict.gpt2.txt

echo "creating indivisible_tokens.txt..."
grep -Eo '\[__\S+__' \
  <(cat "$prep/train.$src-$tgt.$src" "$prep/train.$src-$tgt.$tgt") | \
  sort -u \
  > "$prep/indivisible_tokens.txt"

tmp=$(mktemp -d prepare.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
echo "created a temporary folder..."

echo "creating temporary fine-tuning corpus..."
cat $prep/train.$src-$tgt.$src $prep/train.$src-$tgt.$tgt > $tmp/finetune_corpus

echo "creating temporary dictionary of fine-tuning corpus..."
python -m examples.roberta.multiprocessing_bpe_encoder \
  --encoder-json "$prep/encoder.gpt2.json" \
  --vocab-bpe "$prep/vocab.gpt2.bpe" \
  --inputs "$tmp/finetune_corpus" \
  --outputs "$tmp/finetune_corpus.bpe" \
  --workers 60 \
  --keep-empty

build_vocab() {
  sed 's/ /\n/g' $1 | \
  awk '
    {if ($0!="") wc[$0]+=1}
    END {for (w in wc) print w, wc[w]}
  ' | \
  LC_ALL=C sort -k2,2nr -k1,1
}

build_vocab "$tmp/finetune_corpus.bpe" > "$tmp/dict.ftc.txt"

echo "creating encoder.weather.json..."
python scripts/helper.prepare.bpe.py \
  $prep/encoder.gpt2.json \
  $prep/dict.gpt2.txt \
  $tmp/dict.ftc.txt \
  $prep/indivisible_tokens.txt \
  $prep/encoder.weather.json

rm -rf $tmp
echo "deleted the temporary folder..."

for split in "train" "valid" "test" "test.disc"; do
  for lang in $src $tgt; do
    echo "creating $prep/$split.bpe.$src-$tgt.$lang..."
    python bpe_encoder/multiprocessing_bpe_encoder.py \
      --encoder-json "$prep/encoder.weather.json" \
      --vocab-bpe "$prep/vocab.gpt2.bpe" \
      --inputs "$prep/$split.$src-$tgt.$lang" \
      --outputs "$prep/$split.bpe.$src-$tgt.$lang" \
      --workers 60 \
      --keep-empty
  done
done

ln -s dict.gpt2.txt $prep/dict.$src.txt
ln -s dict.gpt2.txt $prep/dict.$tgt.txt
