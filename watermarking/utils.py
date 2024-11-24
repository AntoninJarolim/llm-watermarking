import torch
import numpy as np


def calc_text_entropy(next_token_probs_list, pad_mask, input_lengths, min_input_length):
    """ Calculate entropy of the generated text using joint probability of tokens """
    joint_probs = torch.stack(next_token_probs_list, dim=1)
    entropies = []
    for i in range(joint_probs.size(0)):
        probs = joint_probs[i].reshape(-1)
        tokens_to_skip = input_lengths[i] - min_input_length
        probs = probs[tokens_to_skip:]
        mask = pad_mask[i][(tokens_to_skip + min_input_length):]
        probs = probs[~mask]
        log_probs = torch.log(probs + 1e-10).reshape(-1)
        entropy = -torch.sum(probs * log_probs)
        entropies.append(entropy.item())

    return entropies


def inv_gumbel_cdf_np(x, mu=0, beta=1, eps=1e-20):
    return mu - beta * np.log(-np.log(x + eps))


def inv_gumbel_cdf(x, mu=0, beta=1, eps=1e-20):
    return mu - beta * torch.log(-torch.log(x + eps))


def top_p(logits, temperature, top_p, device):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    sum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = sum_probs - sorted_probs >= top_p
    sorted_probs[mask] = 0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    next_tok_probs = (
        torch.zeros(logits.shape, dtype=sorted_probs.dtype)
        .to(device)
        .scatter(-1, sorted_indices, sorted_probs)
    )
    return next_tok_probs


def split_vocab(green_list_size, vocab_size, watermark_key, device):
    # Set the seed to get the same split with the same key
    rng = np.random.default_rng(watermark_key)
    vocab_indices_rnd = rng.permutation(np.arange(vocab_size))
    split_index = int(vocab_size * green_list_size)
    green, _ = np.split(vocab_indices_rnd, [split_index])
    green_sorted = torch.sort(torch.tensor(green, dtype=torch.int64)).values.to(device)
    return green_sorted


def count_lines(file_path):
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
