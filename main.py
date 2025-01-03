from pprint import pprint

import torch

from watermarking.detectors import UnigramWatermarkDetector
from watermarking.llm import LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM, GumbelNGramWatermarkedLLM
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-3.1-8B")

    parser.add_argument("--force_cpu",
                        action="store_true",
                        help="Force CPU usage")

    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--watermark_name", type=str, default="unigram")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    llm_model = LLM(model_name=args.model_name, device=device, top_p=0.9)

    prompts = [
        "Základní zásady správné výživy jsou",
        "Pokud chce být člověk zdravyý, pak by měl",
        "Když se člověk cítí unavený, měl by",
        "V Praze se nachází",
        "V roce 1989 byla v Československu provedena",
        "První defenestrace v Praze se odehrála v roce",
    ]

    texts, entropy = llm_model.generate_text(
        prompts,
        max_length=args.max_length,
    )

    del llm_model

    if args.watermark_name == "unigram":
        llm_model_w = UnigramWatermarkedLLM(
            model_name=args.model_name, device=device, wm_strength=0.5, top_p=0.9
        )
    elif args.watermark_name == "gumbel":
        llm_model_w = GumbelWatermarkedLLM(model_name=args.model_name, device=device)
    else:
        raise ValueError(
            f"Unknown watermark name: {args.watermark_name}, choose from 'unigram' or 'gumbel'"
        )
    texts_w, entropy_w = llm_model_w.generate_text(prompts, max_length=args.max_length)

    for x, x_w in zip(texts, texts_w):
        if args.watermark_name == "unigram":
            detector = UnigramWatermarkDetector(
                llm_model_w.watermark_key, llm_model_w.green_list_size,
                llm_model_w.tokenizer, llm_model_w.vocab_size,
                device=device
            )

            score_x = detector.detect(x)
            score_x_w = detector.detect(x_w)
            print(f"Original text with score: {score_x}")
            pprint(x)
            print(f"Watermarked text with score: {score_x_w}")
            pprint(x_w)
            print()
        else:
            pprint(x)
            pprint(x_w)
            print()
