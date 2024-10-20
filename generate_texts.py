import argparse
import torch
import json
import os

from six import with_metaclass
from tqdm.auto import tqdm

from watermarking.llm import LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM
from watermarking.utils import count_lines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--try_upload", action="store_true", default=False,
                        help="Runs ./upload_data.sh after generating the texts")
    parser.add_argument(
        "--in_data_name", type=str, default="data.jsonl",
        help="File 'data/input/{lang}/{data_name}' will be used for text generation"
    )
    parser.add_argument("--output_path", type=str, default="./data/output/")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name of the model to use for text generation."
                             "Using all models if not specified"
                        )
    parser.add_argument("--lang", type=str, default=None,
                        help="Language of the texts to generate."
                             "Generating both Czech and English texts if not specified"
                        )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def generate_batch(text_batch, output_dict, model, max_length):
    generated_texts = model.generate_text(text_batch, max_length=max_length)
    batch_gen_tokens = 0
    for in_text, out_text in zip(text_batch, generated_texts):
        out_text = out_text[len(in_text):]  # Strip prefix from the generated text
        output_dict["data"].append(
            {"prompt": in_text, "generated": out_text}
        )

        text_gen_tokens = len(out_text)
        batch_gen_tokens += text_gen_tokens
    return batch_gen_tokens


def generate_texts(model, data_path, output_path, max_length, lang, batch_size=1):
    text_batch = []
    model_name = model.name.replace("/", "-")
    wm_class_name = type(model).__name__
    output_file = os.path.join(output_path, f"{lang}-{model_name}-{wm_class_name}.json")
    output_dict = {}
    output_dict["model_params"] = model.watermark_config()
    output_dict["data"] = []
    generated_tokens = 0
    with open(data_path, "r") as f:
        pbar = tqdm(
            f,
            desc=f"Generating {lang} texts with {model_name} - {wm_class_name}",
            total=count_lines(data_path))
        for line in pbar:
            line = json.loads(line)["text"]
            text_batch.append(line)

            if len(text_batch) == batch_size:
                generated_tokens += generate_batch(text_batch, output_dict, model, max_length)
                pbar.set_postfix({"tokens/s": generated_tokens / pbar.format_dict['elapsed']})
                text_batch = []

    # Generate the last batch of remaining texts
    if len(text_batch) > 0:
        generate_batch(text_batch, output_dict, model, max_length)

    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_file, "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    if args.force_cpu:
        device = "cpu"

    model_classes = [LLM, UnigramWatermarkedLLM, GumbelWatermarkedLLM]
    model_names = (
        ["meta-llama/Llama-3.1-8B", "BUT-FIT/csmpt7b"]
        if args.model_name is None
        else [args.model_name]
    )
    langs = (
        ["czech", "english"]
        if args.model_name is None
        else [args.lang]
    )

    for model_class in model_classes:
        for model_name in model_names:
            model = model_class(model_name=model_name, device=device, top_p=0.9, )
            for lang in langs:
                generate_texts(
                    model,
                    f"data/input/{lang}/{args.in_data_name}",
                    args.output_path,
                    args.max_length,
                    lang,
                    args.batch_size,
                )
            del model

    if args.try_upload:
        os.system("./upload_data.sh")
