import sys

import numpy as np
import transformers
import torch


class LLM:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.name = model_name
        self.config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.vocab_size = self.config.vocab_size
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

    # Possibly use text as input rather than tokens
    def next_token_logits(self, input_tokens, current_position):
        logits = self.model.forward(input_tokens[:, :current_position]).logits
        return logits

    def tokenize_input(self, texts, max_length=256) -> torch.Tensor:
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt").input_ids

    def decode_output(self, output_tokens) -> list:
        return self.tokenizer.batch_decode(output_tokens)

    def generate_text(self, texts, max_length=256):
        input_tokens = self.tokenize_input(texts, max_length).to(self.device)

        # Pad all input tokens to the length of the shortest input
        min_prompt_len = min((input_tokens == self.pad_token_id).type(torch.int).argmax(1))
        input_tokens[:, min_prompt_len:] = self.pad_token_id

        for current_position in range(min_prompt_len, max_length):
            # Generate the next token logits
            next_token_logits = self.next_token_logits(input_tokens, current_position)
            next_token_logits = next_token_logits[:, -1, :]  # Only the last token of each sequence
            next_token_ids = self.select_next_token(next_token_logits)
            input_tokens[:, current_position] = next_token_ids

        return self.decode_output(input_tokens)

    def select_next_token(self, next_token_logits):
        return torch.argmax(next_token_logits, dim=-1)


class UnigramWatermarkedLLM(LLM):
    def __init__(self, model_name, device="cpu", green_list_size=0.5, wm_strength=2, wm_key=None):
        super().__init__(model_name, device)
        self.wm_strength = wm_strength
        self.watermark_key = wm_key if wm_key is not None else np.random.SeedSequence().entropy

        # Split the vocabulary into green and red lists
        rng = np.random.default_rng(self.watermark_key)
        vocab_indices_rnd = rng.permutation(np.arange(self.vocab_size))
        split_index = int(self.vocab_size * green_list_size)
        self.green, self.red = np.split(vocab_indices_rnd, [split_index])
        self.green = torch.sort(torch.tensor(self.green, dtype=torch.int64)).indices.to(device)

    def __str__(self):
        print(f"UnigramWatermarkedLLM with watermark key: {self.watermark_key}")
        print(f"Green list: {self.green}")
        print(f"Red list: {self.red}")

    def select_next_token(self, next_token_logits):
        next_token_logits[:, self.green] += self.wm_strength
        return super().select_next_token(next_token_logits)
