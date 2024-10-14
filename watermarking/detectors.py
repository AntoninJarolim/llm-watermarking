import torch
import numpy as np
from scipy import special

from . import utils


class GumbelDetector:
    """
    This code is heavily inspired by the original implementation of the GumbelSoft watermark,
    which can be found at: https://github.com/PorUna-byte/Gumbelsoft
    """

    def __init__(self, tokenizer, seed, shift_max, wmkey_len, device):
        super().__init__(tokenizer, seed)
        self.shift_max = shift_max
        self.wmkey_len = wmkey_len
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.xis = [
            utils.inv_gumbel_cdf(
                torch.rand(self.vocab_size, generator=self.rng).numpy()
            )
            for _ in range(self.wmkey_len)
        ]

    def sequence_score(self, token_ids, xi):
        scores = 0
        for token_id in token_ids:
            scores += xi[token_id]
        return scores

    def get_zscore(self, score, ntoks):
        mu = 0.57721
        sigma = np.pi / np.sqrt(6)
        zscore = (score / ntoks - mu) / (sigma / np.sqrt(ntoks))
        return zscore

    def get_pvalue(self, score, ntoks, eps=1e-200):
        zscore = self.get_zscore(score, ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)

    def detect(self, text, toks=None, eps=1e-200):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if toks is not None:  # Not sure if this is necessary
            token_ids = token_ids[:toks]

        m = len(token_ids)
        seq_scores = [
            self.sequence_score(token_ids, self.xis[shift])
            for shift in range(self.shift_max + 1)
        ]
        p_values = [self.get_pvalue(score, m, eps) for score in seq_scores]
        z_scores = [self.get_zscore(score, m) for score in seq_scores]
        p_value = min(p_values)
        shift = p_values.index(p_value)
        z_score = z_scores[shift]

        return float(z_score), float(p_value)
