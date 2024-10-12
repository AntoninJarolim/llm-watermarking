import torch
import transformers
from transformers import pipeline, AutoModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# name = 'BUT-FIT/csmpt7b'
name = 'meta-llama/Llama-3.1-8B'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.init_device = 'cuda:0'  # For fast initialization directly on GPU!
model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    trust_remote_code=True  
)

model = model.eval()
for param in model.parameters():
    param.requires_grad = False

tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
prompt_tokens = tokenizer.encode('Bedřich vole smetana nebyl žádnej ',
                                 return_tensors='pt')

pad_id = model.config.pad_token_id if model.config.pad_token_id else 0
tokens = torch.full((1, 250), pad_id).to(device).long()
prompt_len = len(prompt_tokens[0])
for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = t.clone().detach().long()


logits = model.forward(tokens[:, :prompt_len]).logits
top_k_ids = torch.topk(logits[:, -1], 10).indices[0]
top_k_ids = top_k_ids.unsqueeze(1)

print(tokenizer.batch_decode(top_k_ids))

