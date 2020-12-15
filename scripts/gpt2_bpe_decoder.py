import argparse
import json

from fairseq.data.encoders.gpt2_bpe_utils import get_encoder

def main(args):
    f_in = open(args.input, 'r')
    f_out = open(args.output, 'w')

    if args.subtok:
        with open(args.vocab, 'r') as f:
            tok2id = json.load(f)
        id2tok = {v: k for k, v in tok2id.items()}
        for line in f_in:
            ids = [int(x) for x in line.split()]
            lex = ' '.join([id2tok[x] for x in ids])
            f_out.write(lex + '\n')
    else:
        encoder = get_encoder(args.vocab, args.bpe)
        for line in f_in:
            # `[__DG_INFORM__` and ` [__DG_INFORM__` are different tokens.
            # Only ` [__DG_INFORM__` are registered.
            # So the leading space should be removed before evaluation.
            ids = [int(x) for x in line.split()]
            lex = encoder.decode(ids).lstrip()
            f_out.write(lex + '\n')

    f_in.close()
    f_out.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab', help="encoder.json")
    parser.add_argument('bpe', help="vocab.bpe")
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--subtok', action='store_true', help="Don't merge bpe")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())
