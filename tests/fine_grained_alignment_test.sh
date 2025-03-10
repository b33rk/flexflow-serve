#! /usr/bin/env bash
set -x
set -e

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}
MEMORY_PER_GPU=${MEMORY_PER_GPU:-14000}
ZCOPY_MEMORY=${ZCOPY_MEMORY:-40000}
TP_DEGREE=${TP_DEGREE:-2}
PP_DEGREE=${PP_DEGREE:-2}
CACHE_PATH=${FF_CACHE_PATH:-"~/.cache/flexflow"}
NUM_STEPS=${NUM_STEPS:-2}
FULL_PRECISION=${FULL_PRECISION:-false}
FUSION=${FUSION:-true}

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

cleanup() {
    eval rm -rf "${CACHE_PATH}/debug" /tmp/fine_grained_alignment_config.json ./inference/output/fine_grained_alignment_test_ff.txt ./inference/output/fine_grained_alignment_test_hf.txt
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/.."

# Initial cleanup
cleanup

# Create test prompt file
mkdir -p ./inference/prompt
echo '["Three tips for staying healthy are: "]' > ./inference/prompt/test.json

# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1
export FF_DEBG_NO_WEIGHTS=1



# Check if the Python code executed successfully
if ! PROMPT_LENGTH=$(python -c "
from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained(\"$MODEL_NAME\")
tokens = tokenizer.tokenize('Three tips for staying healthy are: ')
print(len(tokens))
");
then
    echo "Error: Failed to execute Python code"
    exit 1
fi

MAX_LENGTH=$((PROMPT_LENGTH + NUM_STEPS + 1))
if [ "$FULL_PRECISION" = "true" ]; then full_precision_flag="--use-full-precision"; else full_precision_flag=""; fi
if [ "$FUSION" = "true" ]; then fusion_flag="--fusion"; else fusion_flag=""; fi

eval python ./tests/inference/huggingface_inference.py \
    --model-name "${MODEL_NAME}" \
    --max-length "${MAX_LENGTH}" \
    --prompt-file ../../inference/prompt/test.json \
    --output-file ../../inference/output/fine_grained_alignment_test_hf.json \
    "${full_precision_flag}" --inference-debugging

NUM_GPUS=$((TP_DEGREE * PP_DEGREE))
json_config=$(cat <<-END
    {
        "num_gpus": ${NUM_GPUS},
        "memory_per_gpu": ${MEMORY_PER_GPU},
        "zero_copy_memory_per_node": ${ZCOPY_MEMORY},
        "num_cpus": 4,
        "legion_utility_processors": 4,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": ${TP_DEGREE},
        "pipeline_parallelism_degree": ${PP_DEGREE},
        "inference_debugging": true,
        "fusion": ${FUSION},
        "refresh_cache": false,
        "llm_model": "${MODEL_NAME}",
        "cache_path": "${CACHE_PATH}",
        "full_precision": ${FULL_PRECISION},
        "prompt": "./inference/prompt/test.json",
        "max_length": $MAX_LENGTH,
        "output_file": "./inference/output/fine_grained_alignment_test_ff.json"
    }
END
)
echo "$json_config" > /tmp/fine_grained_alignment_config.json

python ./inference/python/incr_decoding.py -config-file /tmp/fine_grained_alignment_config.json

# C++ test
echo "C++ test"
eval ./build/inference/incr_decoding/incr_decoding \
    -ll:gpu "${NUM_GPUS}" -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree "${TP_DEGREE}" \
    -pipeline-parallelism-degree "${PP_DEGREE}" \
    -ll:fsize "${MEMORY_PER_GPU}" -ll:zsize "${ZCOPY_MEMORY}" \
    -llm-model "${MODEL_NAME}" \
    -prompt ./inference/prompt/test.json \
    --max-length $MAX_LENGTH \
    "${full_precision_flag}" "${fusion_flag}" --inference-debugging

# Check alignment
python ./tests/inference/inference_alignment_test.py -m "$MODEL_NAME" -tp "$TP_DEGREE" -n "$NUM_STEPS"

# Print succeess message
echo ""
echo "Inference alignment tests passed (model ${MODEL_NAME})!"
echo ""

# Cleanup after the test
cleanup
