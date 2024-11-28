#!/bin/bash

# download upload data script
cp ~/watermark_data/upload_data.sh .

# download input data from merlin
rsync -rz xjarol06@merlin:/pub/tmp/watermark_data/data/input data --info=progress2 --info=name0

# get current code
git pull

# Run generation code

#!/bin/bash

# Define variables for commonly used arguments
BATCH_SIZE=8
MAX_LENGTH=512
TRY_UPLOAD="--try_upload"
LANGUAGE="english"
INPUT_FILE="data_10_lines.jsonl"
OUTPUT_DIR="./data/output/different_models_test"
CLASS_MODELS="GumbelNGramWatermarkedLLM UnigramWatermarkedLLM"
MODEL_NAMES="
 google/gemma-2-9b-it
 google/gemma-2-2b-it
 meta-llama/Llama-3.1-8B
 meta-llama/Llama-3.2-3B
 mistralai/Ministral-8B-Instruct-2410
 mistralai/Mistral-7B-v0.3"

# Loop to run the command multiple times
python generate_texts.py \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  $TRY_UPLOAD \
  --lang $LANGUAGE \
  --in_data_name $INPUT_FILE \
  --output_path $OUTPUT_DIR \
  --class_model_names $CLASS_MODELS \
  --model_names $MODEL_NAMES \
  --seed 1615453454534154


python detecting_watermark.py --data_dir different_models_test --watermark_name GumbelNGramWatermarkedLLM --model_name ""

python detecting_watermark.py --data_dir different_models_test --watermark_name UnigramWatermarkedLLM --model_name "" 
