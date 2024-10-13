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
    def next_token(self, input_tokens, top_k=10, decode=False):
        input_tokens = input_tokens.to(self.device)

        # Pad all input tokens to the length of the shortest input
        min_prompt_len = min(map(len, input_tokens))
        for k, t in enumerate(input_tokens):
            input_tokens[k, min_prompt_len:] = self.pad_token_id

        logits = self.model.forward(input_tokens[:, :min_prompt_len]).logits
        top_k_ids = torch.topk(logits[:, -1], top_k).indices
        return top_k_ids if not decode else self.tokenizer.batch_decode(top_k_ids)

    def next_token_text(self, texts, top_k=10, decode=False):
        input_tokens = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt").input_ids
        return self.next_token(input_tokens, top_k, decode)

