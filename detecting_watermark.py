import argparse
import json
import os

import torch
import transformers
from tqdm import tqdm

from watermarking.detectors import GumbelDetector, UnigramWatermarkDetector
from watermarking.utils import flatten_list

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Directory with data located in 'data/output' directory")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Skipping all files that do not contain this model name"
                             "Processing all if not specified."
                        )
    return parser.parse_args()


def parse_language(filename):
    if filename.startswith("english-"):
        lang = "english"
    elif filename.startswith("czech-"):
        lang = "czech"
    else:
        raise ValueError("Invalid file name")
    return lang


def parse_watermark_type(filename):
    if "GumbelWatermarkedLLM" in filename:
        watermark_str = "GumbelWatermarkedLLM"
    elif "UnigramWatermarkedLLM":
        watermark_str = "UnigramWatermarkedLLM"
    else:
        raise ValueError("Invalid file name")
    return watermark_str


def parse_model_string(file):
    # Mapping of model names in file to valid model names
    valid_model_names = {
        'BUT-FIT-csmpt7b': 'BUT-FIT/csmpt7b',
        'meta-llama-Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
    }
    for m in valid_model_names.keys():
        if m in file:
            return valid_model_names[m]
    raise ValueError("Invalid model name")


def init_tokenizer(file):
    global tokenizer, vocab_size, last_model_name
    model_name = parse_model_string(file)
    if 'last_model_name' in globals() and model_name == last_model_name:
        return tokenizer, vocab_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    vocab_size = config.vocab_size
    last_model_name = model_name
    return tokenizer, vocab_size


def get_watermark_detector(file, params):
    watermark_str = parse_watermark_type(file)
    tokenizer, vocab_size = init_tokenizer(file)
    detector_params = params.copy()
    detector_params["tokenizer"] = tokenizer
    detector_params["vocab_size"] = vocab_size

    if watermark_str == "GumbelWatermarkedLLM":
        return GumbelDetector(**detector_params)
    elif watermark_str == "UnigramWatermarkedLLM":
        # wm_strength is not used for detection
        if "wm_strength" in params:
            del detector_params["wm_strength"]
        return UnigramWatermarkDetector(**detector_params)
    else:
        raise ValueError("Invalid watermark type")


def get_configurations(file, data_dir):
    """Returns list of configurations to be used on file for detection"""
    lang = parse_language(file)
    watermark_str = parse_watermark_type(file)

    configurations = []
    for file in os.listdir(data_dir):
        if lang in file and watermark_str in file:
            with open(os.path.join(data_dir, file), 'r') as f:
                configurations.append(json.load(f)["model_params"])

    return configurations


def try_detect(watermark_detector, text):
    z_score, p_value, error = None, None, None
    if text == "":
        error = "empty_text"
        print("Warning: found empty text!")
    else:
        try:
            z_score, p_value = watermark_detector.detect(text)
            error = "none"
        except Exception as e:
            error = "error"
            print(f"Failed detection at text: '{text}' with error: \n{e}")

    return {
        "z_score": z_score,
        "p_value": p_value,
        "error": error,
    }


def detect_file(filename, parameters):
    watermark_detector = get_watermark_detector(filename, parameters)
    results = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for d in data["data"]:
            result = try_detect(watermark_detector, d["generated"])
            result.update({f"generated_{k}": v for k, v in data["model_params"].items()})
            result.update({f"detected_{k}": v for k, v in parameters.items()})
            results.append(result)
    return results


def detect(filename, configurations):
    """"Returns list with detection results"""

    detection_results = []
    for parameters in configurations:
        file_results = detect_file(filename, parameters)
        detection_results.append(file_results)

    return flatten_list(detection_results)


def get_files_to_parse(data_dir):
    for file in os.listdir(data_dir):
        # Skip GumbelWatermarkedLLM since it is not working correctly yet
        if "GumbelWatermarkedLLM" in file:
            continue

        # Skip not specified model name
        if args.model_name is not None and args.model_name not in file:
            continue

        yield file


if __name__ == '__main__':
    args = get_args()

    data_dir = os.path.join("data/output", args.data_dir)
    files_to_parse = get_files_to_parse(data_dir)
    for file in (pbar := tqdm(files_to_parse)):
        pbar.set_description(f"Detecting watermark in {file[-50:]}")

        configurations = get_configurations(file, data_dir)
        in_file = os.path.join(data_dir, file)
        get_out = detect(in_file, configurations)

        out_file = os.path.join("data/output", f"{args.data_dir}_detected", file)
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

        with open(out_file, 'w') as f:
            json.dump(get_out, f)
