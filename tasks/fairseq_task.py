"""
Overload fairseq_task.py
"""

from fairseq import search
from fairseq.tasks import FairseqTask

from fairseq.sequence_scorer import SequenceScorer
from fairseq.sequence_generator import SequenceGenerator
from ..constrained_decoding import ConstrainedSequenceGenerator

def build_generator(self, _models, args):
    """
    Overload build_generator()
    """
    if getattr(args, "score_reference", False):
        return SequenceScorer(
            self.target_dictionary,
            compute_alignment=getattr(args, "print_alignment", False),
        )
    search_strategy = search.BeamSearch(self.target_dictionary)
    if args.constr_dec:
        seq_gen_cls = ConstrainedSequenceGenerator
        kwargs = (self.source_dictionary,
                  self.target_dictionary,
                  args.encoder_json,
                  args.vocab_bpe,
                  args.order_constr)
    else:
        seq_gen_cls = SequenceGenerator
        kwargs = (self.target_dictionary,)
    return seq_gen_cls(
        *kwargs,
        beam_size=getattr(args, 'beam', 5),
        max_len_a=getattr(args, 'max_len_a', 0),
        max_len_b=getattr(args, 'max_len_b', 200),
        min_len=getattr(args, 'min_len', 1),
        normalize_scores=(not getattr(args, 'unnormalized', False)),
        len_penalty=getattr(args, 'lenpen', 1),
        unk_penalty=getattr(args, 'unkpen', 0),
        temperature=getattr(args, 'temperature', 1.),
        match_source_len=getattr(args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        search_strategy=search_strategy,
    )

FairseqTask.build_generator = build_generator
