import argparse
import datetime
from time import sleep
from itertools import product

import torch
import json
import os

from tqdm.auto import tqdm

from watermarking.llm import (
    LLM,
    UnigramWatermarkedLLM,
    GumbelWatermarkedLLM,
    GumbelNGramWatermarkedLLM,
    LanguageGenerationError,
)
from watermarking.utils import count_lines


def generate_batch(text_batch, output_dict, model, max_length):
    try:
        generated_texts, entropies = model.generate_text(text_batch, max_length=max_length)
    except LanguageGenerationError as e:
        # seed generating function based on n-gram has a small chance to generate number > 2^63
        # which causes pytorch to raise RuntimeError in method Generator.manual_seed()
        print(f"Warning: Runtime Error occured: {e}")
        generated_texts = text_batch  # Nothing was generated
        entropies = [None for _ in range(len(text_batch))]

    batch_gen_tokens = 0
    for in_text, out_text, entropy in zip(text_batch, generated_texts, entropies):
        out_text = out_text[len(in_text):]  # Strip prefix from the generated text
        output_dict["data"].append(
            {"prompt": in_text, "generated": out_text, "entropy": entropy}
        )

        text_gen_tokens = len(out_text)
        batch_gen_tokens += text_gen_tokens
    return batch_gen_tokens


def check_doesnt_exist(output_file):
    if os.path.exists(output_file):
        i = 1
        while os.path.exists(output_file):
            replace = f"~repeat_{i}"
            output_file = output_file.replace(".json", f"{replace}.json")
            i += 1
        print(f"File {output_file} already exists. Appending a '{replace}' to the filename.")
    return output_file


def generate_texts(model, data_path, output_path, max_length, lang, batch_size, param_dict, unique_id=None):
    model_params = model.watermark_config()
    model_name = model_params['model_name']
    class_name = model_params['class_name']
    output_file = os.path.join(
        output_path,
        f"{lang}~{model_name}~{class_name}~{unique_id}.json"
    )
    output_file = check_doesnt_exist(output_file)
    output_dict = {
        "model_params": model_params,
        "run_params": param_dict,
        "data": []
    }

    text_batch = []
    generated_tokens = 0
    with open(data_path, "r") as f:
        pbar = tqdm(
            f,
            desc=f"Gen. {lang} /w {model_name} - {class_name}",
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
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)


def parse_model_classes(model_strings):
    available_models = [
        LLM,
        GumbelWatermarkedLLM,
        UnigramWatermarkedLLM,
        GumbelNGramWatermarkedLLM,
    ]
    parsed_classes = []
    for model_string in model_strings:
        for available_model_class in available_models:
            if model_string == available_model_class.__name__:
                parsed_classes.append(available_model_class)

    assert parsed_classes != [], f"None of {model_strings} matches with any of available models"
    return parsed_classes


def get_args():
    parser = argparse.ArgumentParser()
    # Algorithm arguments
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--try_upload", action="store_true", default=False,
                        help="Runs ./upload_data.sh after generating the texts")

    # Data arguments
    parser.add_argument(
        "--in_data_name", type=str, default="data.jsonl",
        help="File 'data/input/{lang}/{data_name}' will be used for text generation"
    )
    parser.add_argument("--output_path", type=str, default="./data/output/",
                        help="Default is './data/output/'")
    parser.add_argument("--output_file_args", nargs="+", required=False,
                        help="Arguments which will be used in output file name for improved human readability.")

    # Model and lang arguments
    parser.add_argument("--model_names", nargs='+', required=True,
                        help="Huggingface identifiers of the pretrained LLM models to use for text generation.")
    parser.add_argument("--class_model_names", nargs='+', required=True,
                        help="Names of the model class to use for text generation.")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language of the texts to generate. ")

    # All models arguments
    parser.add_argument("--temperatures", nargs='+', default=[1.2], type=float, help="Temperature values.")
    parser.add_argument("--top_ps", nargs='+', default=[0.9], help="Top-p values.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed to use. If not specified, random seed is generated for each instance.")

    # GumbelSoftmax arguments
    parser.add_argument("--taus", nargs='+', default=[0.9], help="Tau values.")
    parser.add_argument("--ngrams", nargs='+', default=[3], help="Ngram values.")

    # UnigramWatermarkedLLM arguments
    parser.add_argument("--green_list_sizes", nargs='+', default=[0.5], help="Size of the green list split.")
    parser.add_argument("--wm_strengths", nargs='+', default=[2], help="Watermark strength.")

    # Text generation arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def expand_model_args(model_args):
    # Separate scalar and list values
    keys, values = zip(*[
        (key, (val if isinstance(val, list) else [val]))
        for key, val in model_args.items()
    ])

    # Generate Cartesian product for the values
    combinations = product(*values)

    # Convert combinations back to list of dicts
    return [dict(zip(keys, combination)) for combination in combinations]


def rename_args(args):
    rename_map = {
        'temperatures': 'temperature',
        'top_ps': 'top_p',
        'taus': 'tau',
        'ngrams': 'ngram',
        'green_list_sizes': 'green_list_size',
        'wm_strengths': 'wm_strength',
    }

    return {rename_map.get(key, key): val for key, val in args.items()}

def get_model_param_list(model_class, args):
    # Rename arguments for singular shape

    # Add arguments for all models
    model_args = {
        'temperatures': args.temperatures,
        'top_ps': args.top_ps,
    }
    if model_class is UnigramWatermarkedLLM:
        model_args.update(
            {
                'green_list_sizes': args.green_list_sizes,
                'wm_strengths': args.wm_strengths,
                'seed': args.seed,
            }
        )
    elif model_class is GumbelNGramWatermarkedLLM:
        model_args.update(
            {
                'taus': args.taus,
                'ngrams': args.ngrams,
                'seed': args.seed,
            }
        )
    elif model_class is LLM:
        pass  # Nothing to do here
    else:
        raise AssertionError("Incorrect model string name.")

    model_args = rename_args(model_args)
    model_param_lists = expand_model_args(model_args)
    return model_param_lists


def get_unique_id(model_params, output_file_args):
    if output_file_args is None:
        return ""
    return "~".join([f"{arg}_{model_params[arg]}" for arg in output_file_args])


def generate_data(model_classes, args, device, soft_run=False):
    if soft_run:
        print("This is soft run!")

    for model_class, model_name in product(model_classes, args.model_names):
        model_param_lists = get_model_param_list(model_class, args)
        for model_params in model_param_lists:
            print(f"Generating {model_class.__name__} ({model_name}) with {model_params}")
            print(f"With unique id: {get_unique_id(model_params, args.output_file_args)}")
            if soft_run:
                print()
                continue

            model = model_class(model_name=model_name, device=device, **model_params)

            run_dict = {
                'batch_size': args.batch_size,
                'max_length': args.max_length,
                'temperature': model_params['temperature'],
                'top_p': model_params['top_p'],
                'time': datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            }
            generate_texts(
                model,
                f"data/input/{args.lang}/{args.in_data_name}",
                args.output_path,
                args.max_length,
                args.lang,
                args.batch_size,
                run_dict,
                unique_id=get_unique_id(model_params, args.output_file_args)
            )
            del model

            if args.try_upload:
                os.system("./upload_data.sh")

def main():
    args = get_args()

    device = "cuda:0" if torch.cuda.is_available and not args.force_cpu else "cpu"
    model_classes = parse_model_classes(args.class_model_names)

    generate_data(model_classes, args, device, soft_run=True)
    generate_data(model_classes, args, device, soft_run=False)


if __name__ == "__main__":
    main()
