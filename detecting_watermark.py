import argparse
import json
import os
import transformers
from multiprocessing import Pool

from tqdm import tqdm

from watermarking.detectors import GumbelDetector, UnigramWatermarkDetector, GumbelNGramDetector
from watermarking.utils import flatten_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Directory with data located in 'data/output' directory")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Skipping all files that do not contain this model name"
                             "Processing all if not specified."
                        )
    parser.add_argument('--watermark_name', type=str, default=None,
                        help="Skipping all files that do not contain this watermark name"
                             "Processing all if not specified."
                        )
    return parser.parse_args()


def init_tokenizer(model_name):
    global tokenizer, vocab_size, last_model_name
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


def get_watermark_detector(params, watermark_str, model_name):
    tokenizer, vocab_size = init_tokenizer(model_name)
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
    elif watermark_str == "GumbelNGramWatermarkedLLM":
        for param_name in  ['class_name', 'model_name', 'seeding', 'tau', 'drop_prob']:
            if param_name in params:
                del detector_params[param_name]
        return GumbelNGramDetector(**detector_params)
    else:
        raise ValueError(f"Invalid watermark type: {watermark_str}")


def get_configurations(data_dir, lang, watermark_str):
    """Returns list of configurations to be used on file for detection"""

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
            print(f"{os.getpid()}: Failed detection at text: '{text}' with error: \n{e}")

    return {
        "z_score": z_score,
        "p_value": p_value,
        "error": error,
    }


def detect_file(filename, parameters, model_name, watermark_str):
    watermark_detector = get_watermark_detector(parameters, watermark_str, model_name)
    results = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for i, d in enumerate(data["data"]):
            # pbar.set_description(pbar_description + f" text {i}/{len(data['data'])}")
            result = try_detect(watermark_detector, d["generated"])
            result.update({f"generated_{k}": v for k, v in data["model_params"].items()})
            result.update({f"detected_{k}": v for k, v in parameters.items()})
            results.append(result)
    return results


def detect(filename, configurations, model_name, watermark_str):
    """"Returns list with detection results"""

    detection_results = []
    for parameters in tqdm(configurations, desc=f"Detecting {filename} with config."):
        file_results = detect_file(filename, parameters, model_name, watermark_str)
        detection_results.append(file_results)

    return flatten_list(detection_results)


def get_files_to_parse(data_dir, args):
    for file in os.listdir(data_dir):

        # Skip not specified model name
        if (
                args.model_name not in file
                and args.watermark_name not in file
        ):
            continue

        yield file


def process_file(file):
    print(f"{os.getpid()}: Processing file: {file}")
    valid_model_names = {
        'BUT-FIT-csmpt7b': 'BUT-FIT/csmpt7b',
        'meta-llama-Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
    }

    # pbar.set_description(f"Detecting watermark in {file[-50:]}")

    lang, model_name, watermark_str, *_ = file.split("~")
    model_name = valid_model_names[model_name]

    configurations = get_configurations(data_dir, lang, watermark_str)
    in_file = os.path.join(data_dir, file)
    get_out = detect(in_file, configurations, model_name, watermark_str)

    out_file = os.path.join("data/output", f"{args.data_dir}_detected", file)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with open(out_file, 'w') as f:
        json.dump(get_out, f)


if __name__ == '__main__':
    args = get_args()

    data_dir = os.path.join("data/output", args.data_dir)
    files_to_parse = get_files_to_parse(data_dir, args)

    with Pool() as pool:
        pool.map(process_file, files_to_parse)
