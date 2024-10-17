import argparse
import torch
import json
import os

from watermarking.llm import LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--czech_data_path", type=str, default="./data/czech_data.txt")
    parser.add_argument(
        "--english_data_path", type=str, default="./data/english_data.txt"
    )
    parser.add_argument("--output_path", type=str, default="./data/output/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def generate_texts(model, data_path, output_path, max_length, batch_size=1):
    text_batch = []
    output_file = os.path.join(output_path, data_path, f"{model.name}.txt")
    output_dict = {}
    output_dict["model_params"] = model.__dict__
    output_dict["data"] = []
    with open(data_path, "r") as f:
        for line in f:
            text_batch.append(line)
            print(text_batch)

            if len(text_batch) == batch_size:
                generated_texts = model.generate_text(text_batch, max_length=max_length)
                for in_text, out_text in zip(text_batch, generated_texts):
                    output_dict["data"].append(
                        {"prompt": in_text, "generated": out_text}
                    )
                text_batch = []

    if len(text_batch) > 0:
        generated_texts = model.generate_text(text_batch)
        for in_text, out_text in zip(text_batch, generated_texts):
            output_dict["data"].append({"prompt": in_text, "generated": out_text})

    with open(output_file, "w") as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    if args.force_cpu:
        device = "cpu"
    model_classes = [LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM]
    model_names = ["meta-llama/Llama-3.1-8B", "BUT-FIT/csmpt7b"]

    for model_class in model_classes:
        for model_name in model_names:
            model = model_class(model_name=model_name)
            generate_texts(
                model,
                args.czech_data_path,
                args.output_path,
                args.max_length,
                args.batch_size,
            )
            del model
