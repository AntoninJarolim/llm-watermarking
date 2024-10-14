import torch


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
        torch.zeros(logits.shape).to(device).scatter(-1, sorted_indices, sorted_probs)
    )
    return next_tok_probs
