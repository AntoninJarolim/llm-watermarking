import torch

from llm import LLM
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-3.1-8B")

    parser.add_argument("--force_cpu",
                        action="store_true",
                        help="Force CPU usage")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    llm_model = LLM(args.model_name, device=device)
    print(
        llm_model.next_token(
            llm_model.tokenizer.encode("Hello, my name is", return_tensors="pt"),
            decode=True,
        )
    )
