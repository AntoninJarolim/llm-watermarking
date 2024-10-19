import dataclasses

import transformers
import torch
from tqdm.auto import tqdm
import numpy as np

from . import utils


class LLM:
    def __init__(self, model_name=None, device="cpu", temperature=1.0, top_p=1.0):
        self.out_cache = None
        self.device = device
        self.name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.vocab_size = self.config.vocab_size
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.config.init_device = device
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.pad_token_id = self.tokenizer.pad_token_id

    def watermark_config(self):
        return {}

    def tokenize_input(self, texts, max_length=256) -> torch.Tensor:
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt").input_ids

    def decode_output(self, output_tokens) -> list:
        return self.tokenizer.batch_decode(output_tokens)

    def next_token_logits(self, input_tokens, current_position, prev_pos):
        self.out_cache = self.model.forward(
            input_tokens[:, prev_pos:current_position],
            use_cache=True,
            past_key_values=self.out_cache.past_key_values if prev_pos > 0 else None
        )
        return self.out_cache.logits

    def generate_text(self, texts, max_length=700, pad_to_shortest=False, disable_tqdm=True):
        input_tokens = self.tokenize_input(texts, max_length).to(self.device)

        # Pad all input tokens to the length of the shortest input
        min_prompt_len = min((input_tokens == self.pad_token_id).type(torch.int).argmax(1))
        if min_prompt_len == 0:
            min_prompt_len = input_tokens.size(1)  # No padding detected in any input text
        if pad_to_shortest:
            input_tokens[:, min_prompt_len:] = self.pad_token_id

        prev_pos = 0
        for current_position in tqdm(range(min_prompt_len, max_length), disable=disable_tqdm):
            # Generate the next token logits
            next_token_logits = self.next_token_logits(
                input_tokens, current_position, prev_pos
            )
            next_token_logits = next_token_logits[:, -1, :]  # Only the last token of each sequence
            next_token_ids = self.select_next_token(next_token_logits)

            # Replace only [PAD] tokens
            pad_mask = (input_tokens[:, current_position] == self.pad_token_id)
            input_tokens[:, current_position][pad_mask] = next_token_ids[pad_mask]

            prev_pos = current_position

        return self.decode_output(input_tokens)

    def select_next_token(self, next_token_logits):
        if self.temperature > 0:
            probs = utils.top_p(next_token_logits, self.temperature, self.top_p, self.device)
            next_token = torch.multinomial(probs, num_samples=1).flatten()
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)

        return next_token


class UnigramWatermarkedLLM(LLM):
    def __init__(self, device="cpu", green_list_size=0.5, wm_strength=2, watermark_key=None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.wm_strength = wm_strength
        self.watermark_key = watermark_key if watermark_key is not None else np.random.SeedSequence().entropy
        self.green_list = utils.split_vocab(green_list_size, self.vocab_size, self.watermark_key, device)
        self.green_list_size = green_list_size

    def __str__(self):
        print(f"UnigramWatermarkedLLM with watermark key: {self.watermark_key}")

    def select_next_token(self, next_token_logits):
        next_token_logits[:, self.green_list] += self.wm_strength
        return super().select_next_token(next_token_logits)

    def watermark_config(self):
        return {
            "green_list_size": self.green_list_size,
            "wm_strength": self.wm_strength,
            "watermark_key": self.watermark_key,
        }


class GumbelWatermarkedLLM(LLM):
    def __init__(
            self,
            watermark_key_len=256,
            shift_max=0,
            seed=69,
            device="cpu",
            **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self.rng = torch.Generator(device="cpu")
        self.watermark_key_len = watermark_key_len
        self.shift_max = shift_max
        self.rng.manual_seed(seed)
        self.xis = [
            utils.inv_gumbel_cdf(torch.rand(self.vocab_size, generator=self.rng))
            for _ in range(self.watermark_key_len)
        ]

    def _get_unique_id(self, batch_size):
        return np.random.randint(self.shift_max + 1, size=batch_size)

    def _key_func(self, r):
        xis = [self.xis[i] for i in r]
        return xis

    def _gamma(self, xis, logits):
        xis = torch.stack(xis).to(self.device)
        return logits + xis

    def watermark_config(self):
        return {
            "watermark_key_len": self.watermark_key_len,
            "shift_max": self.shift_max,
            "seed": self.rng.seed(),
        }

    def select_next_token(self, next_token_logits):
        batch_size = next_token_logits.size(0)
        uid = self._get_unique_id(batch_size)
        xi = self._key_func(uid)
        next_token_logits = self._gamma(xi, next_token_logits)
        return super().select_next_token(next_token_logits)
