import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Directory with data located in 'data/output' directory")
    return parser.parse_args()


def detect(file_data):
    pass


if __name__ == '__main__':
    args = get_args()

    for file in os.listdir(os.path.join("data/output", args.data_dir)):
        print(file)

        with open(os.path.join("data/output", args.data_dir, file), 'r') as f:
            get_out = detect(json.load(f))

        with open(os.path.join("data/output", args.data_dir, "_detected", file), 'w') as f:
            json.dump(get_out, f)
