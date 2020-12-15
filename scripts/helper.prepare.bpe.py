"""
Build a set of indivisible tokens.
Recognise unused tokens based on the frequency in `dict.txt` of gpt2 and fine-tuning corpus.
Substitute them to indivisible tokens in `encoder.json`.
"""

import argparse
import json

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def main(args):
    with open(args.vocab, 'r') as f:
        tok2id = json.load(f)
    id2tok = {v: k for k, v in tok2id.items()}

    unused_tok_id_gpt2 = set()
    with open(args.dxnry_gpt2, 'r') as f:
        for line in f:
            k, v = line.split()
            if not isint(k):
                continue
            if 30000 <= int(k) < 50256 and int(v) == 0:
                # id >= 30000 can assure the token is not that fundamental
                unused_tok_id_gpt2.add(int(k))

    used_tok_id_ftc = set()
    with open(args.dxnry_ftc, 'r') as f:
        for line in f:
            k, v = line.split()
            if not isint(k):
                continue
            used_tok_id_ftc.add(int(k))

    # assure no used token in fine-tuning dataset will be substituted
    assert not bool(used_tok_id_ftc & unused_tok_id_gpt2)

    unused_tok_id_sorted = sorted(unused_tok_id_gpt2, reverse=True)

    with open(args.indiv, 'r') as f:
        indiv = []
        for line in f:
            indiv.append(line.strip())

    assert len(indiv) <= len(unused_tok_id_sorted)
    for x, nt in zip(unused_tok_id_sorted, indiv):
        id2tok[x] = '\u0120' + nt
    tok2id = {v: k for k, v in id2tok.items()}

    with open(args.output, 'w') as f:
        json.dump(tok2id, f, indent=0, ensure_ascii=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab', help="encoder.json of gpt2")
    parser.add_argument('dxnry_gpt2', help="dict.txt of gpt2")
    parser.add_argument('dxnry_ftc', help="dict.txt of fine-tuning corpus")
    parser.add_argument('indiv', help="indivisible_tokens.txt")
    parser.add_argument('output')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())
