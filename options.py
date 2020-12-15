"""
Overload options
"""

from fairseq import options
from fairseq.options import add_generation_args


def overload_add_generation_args(parser):
    """
    Add options to generator
    """
    group = add_generation_args(parser)
    group.add_argument('--constr-dec', action='store_true', help='activate constrained decoding')
    group.add_argument('--encoder-json', help='encoder.json')
    group.add_argument('--vocab-bpe', help='vocab.bpe')
    group.add_argument('--order-constr', action='store_true', help='activate order constraint')
    return group

options.add_generation_args = overload_add_generation_args
