"""
Overload gpt2_bpe.py to handle strutual non-terminals
"""

from fairseq.data.encoders.gpt2_bpe_utils import Encoder


def encode(self, text):
    pat = self.re.compile(r""" \[__\S+__|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    text = " " + text
    bpe_tokens = []
    for token in self.re.findall(pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        if self.re.match(r"\u0120\[__\S+__", token):
            # special indivisible tokens
            bpe_tokens.extend([self.encoder[token]])
        else:
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    return bpe_tokens


Encoder.encode = encode
