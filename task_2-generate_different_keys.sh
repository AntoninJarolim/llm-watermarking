
conda deactivate && conda activate llm
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
INPUT_FILE="data_500_lines.jsonl"
OUTPUT_DIR="./data/output/different_models"
CLASS_MODELS="GumbelNGramWatermarkedLLM UnigramWatermarkedLLM"
MODEL_NAMES="meta-llama/Llama-3.1-8B"

# Loop to run the command multiple times
for _ in {1..5}; do
  python generate_texts.py \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    $TRY_UPLOAD \
    --lang $LANGUAGE \
    --in_data_name $INPUT_FILE \
    --output_path $OUTPUT_DIR \
    --class_model_names "$CLASS_MODELS" \
    --model_names $MODEL_NAMES
done


python detecting_watermark.py --data_dir different_models --watermark_name GumbelNGramWatermarkedLLM

python detecting_watermark.py --data_dir different_models --watermark_name UnigramWatermarkedLLM

