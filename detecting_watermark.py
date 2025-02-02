import argparse
import inspect
import json
import os
import transformers
from multiprocessing import Pool

from tqdm import tqdm

from watermarking.detectors import GumbelDetector, UnigramWatermarkDetector, GumbelNGramDetector
from watermarking.utils import flatten_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_multiprocessing", action="store_true", help="Disable multiprocessing")
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
    available_detectors = {
        'UnigramWatermarkedLLM': UnigramWatermarkDetector,
        'GumbelWatermarkedLLM': GumbelDetector,
        'GumbelNGramWatermarkedLLM': GumbelNGramDetector,
    }
    detector_class = available_detectors[watermark_str]
    init_params = inspect.signature(detector_class.__init__).parameters
    filtered_params = {key: value
                       for key, value in params.items()
                       if key in init_params}
    return detector_class(tokenizer, vocab_size, **filtered_params)


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


def detect_file(filename, detect_parameters, model_name, watermark_str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    watermark_detector = get_watermark_detector(
        detect_parameters,
        watermark_str,
        detect_parameters['model_name']
    )
    results = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for i, d in enumerate(data["data"]):
            # pbar.set_description(pbar_description + f" text {i}/{len(data['data'])}")
            result = try_detect(watermark_detector, d["generated"])
            result.update({f"generated_{k}": v for k, v in data["model_params"].items()})
            result.update({f"detected_{k}": v for k, v in detect_parameters.items()})
            result['generated_token_len'] = len(
                tokenizer.encode(
                    d["generated"],
                    add_special_tokens=False
                )
            )
            results.append(result)
    return results


def detect(filename, configurations, model_name, watermark_str):
    """"Returns list with detection results"""

    print(f"Detecting {filename} with configurations {configurations}")

    detection_results = []
    for parameters in tqdm(configurations, desc=f"Detecting {filename}"):
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


def convert_model_name(model_name):
    valid_hface_prefixes = ['BUT-FIT', 'meta-llama', 'mistralai', 'google']
    for valid_hface_prefix in valid_hface_prefixes:
        if valid_hface_prefix in model_name:
            return f"{valid_hface_prefix}/{model_name.removeprefix(valid_hface_prefix + '-')}"
    raise ValueError(
        f"Model name '{model_name}' does not match with any specified prefixes {valid_hface_prefixes}"
    )


def process_file(file):
    print(f"{os.getpid()}: Processing file: {file}")

    # pbar.set_description(f"Detecting watermark in {file[-50:]}")

    lang, model_name, watermark_str, *_ = file.split("~")
    model_name = convert_model_name(model_name)

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

    if args.disable_multiprocessing:
        for file_to_parse in files_to_parse:
            process_file(file_to_parse)
    else:
        with Pool() as pool:
            pool.map(process_file, files_to_parse)
