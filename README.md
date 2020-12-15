# TreeNLG-BART

This repo applies constrained decoding to the large pretrained BART of [fairseq](https://github.com/pytorch/fairseq).
It inherit the code from [v0.1.0](https://github.com/znculee/TreeNLG/releases/tag/v0.1.0) of [znculee/TreeNLG](https://github.com/znculee/TreeNLG).

## Get Started

[fairseq](https://github.com/pytorch/fairseq) should be installed at the very beginning, referring to [Requirements and Installation of Fairseq](https://github.com/pytorch/fairseq#requirements-and-installation).
The code has been tested on commit `e9014fb` of [fairseq](https://github.com/pytorch/fairseq).
The pretrained BART models can be download [here](https://github.com/pytorch/fairseq/tree/master/examples/bart#pre-trained-models).
We use `bart.large` here.
The Weather dataset can be found in [znculee/TreeNLG](https://github.com/znculee/TreeNLG/tree/master/data/weather) or [facebookresearch/TreeNLG](https://github.com/facebookresearch/TreeNLG/tree/master/data/weather).
Then, you are good to follow the next scripts.

```bash
bash cache/gpt2_bpe/download.sh
bash script/prepare.weather.sh
bash script/train.weather.bart_large.sh
bash script/generate.weather.bart_large.sh
```
## Performance

The BLEU score is calculated on just the output text, without any of the tree information.
We use the BLEU evaluation script provided for the E2E challenge [here](https://github.com/tuetschek/e2e-metrics).
The scores of `LSTM-*` are from [znculee/TreeNLG](https://github.com/znculee/TreeNLG#results).

```
Method           | BLEU  | TreeAcc
--               | --    | --
LSTM             | 76.34 | 94.17
LSTM-CD          | 76.88 | 99.84
LSTM-CD-RF       | 77.38 | 99.84
BART_Large       | 78.25 | 97.37
BART_Large-CD    | 78.80 | 99.97 (3120/3121)
BART_Large-CD-RF | 78.81 | 99.97
--
* CD: Constrained Decoding
* RF: Replacing Failures with vanilla decoding
```

## Development Log

- [X] Check if BART can use fairseq-generate directly because it should be a standard Transformer.
- [X] Understand how gpt2-bpe works exactly.
  - The space is considered as a part of the subtokens, instead of using trailing `@@` as in [rsennrich/subwork-nmt](https://github.com/rsennrich/subword-nmt).
  - Cutting a test sentence into characters and apply the merging rules (`vocab.bpe`) to n-grams.
  - Decoding the merged subtokens to ids according to `encoder.json`.
- [X] Overload gpt2-bpe encoder for protecting the special tokens `r"\[__\S+__"`.
- [X] `[__DG_INFORM__` and ` [__DG_INFORM__` are different tokens. Add a space before encoding.
- [X] Recognise unused tokens based on `dict.txt` of both gpt2 and fine-tuning corpus, and substitute them to special tokens in `encoder.json`.
- [X] Train BART with this better tokenization on the Weather dataset.
- [X] Before constrain checking, translate tokens IDs to token strings beforehand.
- [X] Making ignored non-terminals allow multiple tokens inside the brackets.
- [X] Let BART to generate with constrained decoding on the Weather Dataset.

### Why BART won't work directly on Weather dataset?

The subtokens may be too fragmented as follows, so that it will harm the performance.
```
[__DS_JOIN__ [__DG_NO__ [__ARG_TASK__ get_weather_attribute ] ] [__DG_INFORM__ [__ARG_TASK__ get_weather_attribute ] [__ARG_CONDITION_NOT__ rain ] [__ARG_LOCATION__ [__ARG_CITY__ Seattle ] ] [__ARG_DATE_TIME__ [__ARG_COLLOQUIAL__ today ] ] ] ] [__DG_INFORM__ [__ARG_TASK__ get_forecast ] [__ARG_TEMP_LOW__ 2 ] [__ARG_TEMP_HIGH__ 22 ] [__ARG_CLOUD_COVERAGE__ sunny ] [__ARG_DATE_TIME__ [__ARG_COLLOQUIAL__ today ] ] [__ARG_LOCATION__ [__ARG_CITY__ Seattle ] ] ]
[  __  DS   _  JO    IN   __  Ġ[  __  D  G  _  NO    __  Ġ[  __  AR   G  _  T  AS   K  __  Ġget _  weather _  attribute Ġ]   Ġ]   Ġ[  __  D  G  _  IN   FORM  __  Ġ[  __  AR   G  _  T  AS   K  __  Ġget _  weather _  attribute Ġ]   Ġ[  __  AR   G  _  CON   D  ITION _  NOT   __  Ġrain Ġ]   Ġ[  __  AR   G  _  LOC   ATION __  Ġ[  __  AR   G  _  C  ITY  __  ĠSeattle Ġ]   Ġ]   Ġ[  __  AR   G  _  D  ATE  _  TIME  __  Ġ[  __  AR   G  _  CO   LL   O  QUI   AL   __  Ġtoday Ġ]   Ġ]   Ġ]   Ġ]   Ġ[  __  D  G  _  IN   FORM  __  Ġ[  __  AR   G  _  T  AS   K  __  Ġget _  fore cast Ġ]   Ġ[  __  AR   G  _  T  EMP   _  L  OW   __  Ġ2  Ġ]   Ġ[  __  AR   G  _  T  EMP   _  H  IGH   __  Ġ22  Ġ]   Ġ[  __  AR   G  _  CL   OU   D  _  CO   VER  AGE   __  Ġsunny Ġ]   Ġ[  __  AR   G  _  D  ATE  _  TIME  __  Ġ[  __  AR   G  _  CO   LL   O  QUI   AL   __  Ġtoday Ġ]   Ġ]   Ġ[  __  AR   G  _  LOC   ATION __  Ġ[  __  AR   G  _  C  ITY  __  ĠSeattle Ġ]   Ġ]   Ġ]
58 834 5258 62 45006 1268 834 685 834 35 38 62 15285 834 685 834 1503 38 62 51 1921 42 834 651  62 23563   62 42348     2361 2361 685 834 35 38 62 1268 21389 834 685 834 1503 38 62 51 1921 42 834 651  62 23563   62 42348     2361 685 834 1503 38 62 10943 35 17941 62 11929 834 6290  2361 685 834 1503 38 62 29701 6234  834 685 834 1503 38 62 34 9050 834 7312     2361 2361 685 834 1503 38 62 35 6158 62 34694 834 685 834 1503 38 62 8220 3069 46 43702 1847 834 1909   2361 2361 2361 2361 685 834 35 38 62 1268 21389 834 685 834 1503 38 62 51 1921 42 834 651  62 754  2701 2361 685 834 1503 38 62 51 39494 62 43 3913 834 362 2361 685 834 1503 38 62 51 39494 62 39 18060 834 2534 2361 685 834 1503 38 62 5097 2606 35 62 8220 5959 11879 834 27737  2361 685 834 1503 38 62 35 6158 62 34694 834 685 834 1503 38 62 8220 3069 46 43702 1847 834 1909   2361 2361 685 834 1503 38 62 29701 6234  834 685 834 1503 38 62 34 9050 834 7312     2361 2361 2361
```

