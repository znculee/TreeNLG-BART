#!/usr/bin/env python3

import argparse
import os
import re
import sys

os.chdir(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))
sys.path.append(os.path.realpath('constrained_decoding'))
from constraint_checking import TreeConstraints

def main(args):
    with open(args.ref, 'r') as f:
        refs = [l.strip() for l in f.readlines()]
    with open(args.hyp, 'r') as f:
        hyps = [l.strip() for l in f.readlines()]
    assert len(refs) == len(hyps)
    correct = 0
    for k, (ref, hyp) in enumerate(zip(refs, hyps)):
        print(f'progress: %{100*(k+1)/len(refs):.2f} ({k+1}/{len(refs)})', end='\r')
        ref_tree = TreeConstraints(ref.strip(), args.order_constr)
        hyp_nt = re.compile(r'(\[\S+|\])').findall(hyp.strip())
        for i, w in enumerate(hyp_nt):
            if not ref_tree.next_token(w, i):
                break
        else:
            if ref_tree.meets_all():
                correct += 1
    print(
        'Tree accuracy: {:.2f} ({} / {})'.format(
            correct / len(refs) * 100, correct, len(refs)
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Compute tree accuracy')
    parser.add_argument('hyp', type=str, help='reference')
    parser.add_argument('ref', type=str, help='hypothesis')
    parser.add_argument('--order-constr', action='store_true', help='activate order constraint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())
