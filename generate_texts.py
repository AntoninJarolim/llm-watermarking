import argparse
import torch
import json
import os

from watermarking.llm import LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--czech_data_path", type=str, default="./data/czech_data.jsonl"
    )
    parser.add_argument(
        "--english_data_path", type=str, default="./data/english_data.jsonl"
    )
    parser.add_argument("--output_path", type=str, default="./data/output/")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def generate_texts(model, data_path, output_path, batch_size=1):
    text_batch = []
    output_file = os.path.join(output_path, f"{model.model_name}.txt")
    with open(data_path, "r") as f:
        line = f.readline()
        line = json.loads(line)
        text_batch.append(line["text"])

        if len(text_batch) == batch_size:
            generated_texts = model.generate_text(text_batch)
            with open(output_file, "a") as f:
                for text in generated_texts:
                    f.write(text + "\n")
            text_batch = []


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    model_classes = [LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM]
    model_names = ["meta-llama/Llama-3.1-8B", "BUT-FIT/csmpt7b"]

    for model_class in model_classes:
        for model_name in model_names:
            model = model_class(model_name=model_name)
            generate_texts(
                model, args.czech_data_path, args.output_path, args.batch_size
            )
            del model
