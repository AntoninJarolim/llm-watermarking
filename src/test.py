from pprint import pprint

import torch

from llm import LLM, UnigramWatermarkedLLM
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

    prompts = [
        'Rád bych, milí moji, také věděl, jak se dostat za',
        'Rád bych také věděl, jak se dostat do nebo za',
        'Rád bych také věděl, jak se dostat před',
        'Není mi jasné, jak se dostat za tvoji mamku protoze',
        'Těžko říci, jak se dostat za nebo i před a taky pod',
        'Ale není mi jasné, jak se dostat za, u nebo i před',
    ]

    texts = llm_model.generate_text(
        prompts,
        max_length=150
    )

    del llm_model

    llm_model_w = UnigramWatermarkedLLM(args.model_name, device=device, wm_strength=3)
    texts_w = llm_model_w.generate_text(
        prompts,
        max_length=150
    )

    for x, x_w in zip(texts, texts_w):
        pprint(x)
        pprint(x_w)
        print()
