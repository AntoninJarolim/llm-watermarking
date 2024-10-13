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
