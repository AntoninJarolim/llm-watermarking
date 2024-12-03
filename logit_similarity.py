import torch
import argparse
import json
import pickle

from watermarking.llm import LLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-3.1-8B")

    parser.add_argument("--force_cpu",
                        action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--input_file", type=str, default="data.jsonl")
    parser.add_argument("--output_file", type=str, default="logits.bin")
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.force_cpu:
        device = "cpu"

    model = LLM(model_name=args.model_name, device=device, top_p=0.9)

    logits = []
    processed_lines = 0
    files_created = 0
    with open(args.input_file, "r") as f:
        for line in f:
            line = json.loads(line)
            line = line["text"]
            _, logit_list = model.generate_texts_and_logits([line], max_length=args.max_length)
            flat_logits = [item for sublist in logit_list for item in sublist]
            logits.extend(flat_logits)
            processed_lines += 1

            if processed_lines % args.save_every == 0:
                logits = torch.stack(logits, dim=0)
                outfile = f"{args.output_file}_{files_created}.bin"
                files_created += 1
                print(f"Processed {processed_lines} lines and saving to {outfile}")
                with open(outfile, "wb") as f:
                    pickle.dump(logits, f)
                logits = []
