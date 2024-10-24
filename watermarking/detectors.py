import math

import torch
import numpy as np
from scipy import special

from . import utils


class GumbelDetector:
    """
    This code is heavily inspired by the original implementation of the GumbelSoft watermark,
    which can be found at: https://github.com/PorUna-byte/Gumbelsoft
    """

    def __init__(
            self, tokenizer, vocab_size, seed=69, shift_max=0, watermark_key_len=256, device="cpu"
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.shift_max = shift_max
        self.watermark_key_len = watermark_key_len
        self.device = device
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        self.xis = [
            utils.inv_gumbel_cdf_np(
                torch.rand(self.vocab_size, generator=self.rng).numpy()
            )
            for _ in range(self.watermark_key_len)
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


class UnigramWatermarkDetector:
    def __init__(self, watermark_key, green_list_size, tokenizer, vocab_size, device="cpu"):
        self.watermark_key = watermark_key
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = vocab_size
        self.green_list_size = green_list_size
        self.green_list = utils.split_vocab(
            green_list_size, self.vocab_size, self.watermark_key, device
        )

    def detect(self, text):
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).squeeze().to(self.device)
        binary_tensor = torch.isin(tokens, self.green_list).int()
        nr_green_tokens = binary_tensor.sum().item()
        n = len(tokens)
        z_statistics = ((nr_green_tokens - self.green_list_size * n) /
                        math.sqrt(n * self.green_list_size * (1 - self.green_list_size)))
        return z_statistics

