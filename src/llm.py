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
        self.config.init_device = "cpu"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # Possibly use text as input rather than tokens
    def next_token(self, input_tokens, top_k=10, decode=False):
        pad_id = self.model.config.pad_token_id if self.model.config.pad_token_id else 0
        tokens = torch.full((1, 250), pad_id).to(self.device).long()
        prompt_len = len(input_tokens[0])
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = t.clone().detach().long()

        logits = self.model.forward(tokens[:, :prompt_len].to(self.device)).logits
        top_k_ids = torch.topk(logits[:, -1], top_k).indices[0]
        top_k_ids = top_k_ids.unsqueeze(1)
        return top_k_ids if not decode else self.tokenizer.batch_decode(top_k_ids)

    def next_token_text(self, text, top_k=10, decode=False):
        input_tokens = self.tokenizer.encode(text, return_tensors="pt")
        return self.next_token(input_tokens, top_k, decode)
