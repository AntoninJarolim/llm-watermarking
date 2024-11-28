#!/bin/bash

conda deactivate && conda activate llm
# download upload data script
cp ~/watermark_data/upload_data.sh .

# download input data from merlin
rsync -rz xjarol06@merlin:/pub/tmp/watermark_data/data/input data --info=progress2 --info=name0

# get current code
git pull

# Run generation code

for _ in {1..4}; do
  # python generate_texts.py --batch_size 8 --max_length 512 --try_upload --lang english --model_name meta-llama/Llama-3.1-8B --in_data_name data.jsonl --output_path ./data/output/different_keys --class_model_names GumbelNGramWatermarkedLLM
  python generate_texts.py --batch_size 8 --max_length 512 --try_upload --lang english --model_name meta-llama/Llama-3.1-8B --in_data_name squad.jsonl --output_path ./data/output/different_keys_squad --class_model_names GumbelNGramWatermarkedLLM
done

# python detecting_watermark.py --data_dir different_keys --model_name meta-llama/Llama-3.1-8B --watermark_name GumbelNGramWatermarkedLLM
python detecting_watermark.py --data_dir different_keys_squad --model_name meta-llama/Llama-3.1-8B --watermark_name GumbelNGramWatermarkedLLM

