#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ~/.cache/flexflow/debug
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/.."

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Clean up before test (just in case)
cleanup

# Create test prompt file
mkdir -p ./inference/prompt
echo '["Two things are infinite: "]' > ./inference/prompt/peft.json
echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ./inference/prompt/peft_dataset.json


# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Download test model
python ./inference/utils/download_peft_model.py goliaro/llama-160m-lora

# # Run PEFT in Huggingface to get ground truth tensors
python ./tests/peft/hf_finetune.py --peft-model-id goliaro/llama-160m-lora --save-peft-tensors --use-full-precision -lr 0.001

# Python test
echo "Python test"
json_config=$(cat <<-END
    {
        "num_gpus": 4,
        "memory_per_gpu": 14000,
        "zero_copy_memory_per_node": 10000,
        "num_cpus": 4,
        "legion_utility_processors": 4,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 4,
        "pipeline_parallelism_degree": 1,
        "enable_peft": true,
        "inference_debugging": true,
        "fusion": false,
        "refresh_cache": false,
        "base_model": "JackFram/llama-160m",
        "inference_peft_model_id": "goliaro/llama-160m-lora",
        "finetuning_peft_model_id": "goliaro/llama-160m-lora",
        "cache_path": "${FF_CACHE_PATH:-}",
        "full_precision": true,
        "prompt": "",
        "finetuning_dataset": "./inference/prompt/peft_dataset.json",
        "output_file": "",
        "max_requests_per_batch": 1,
        "max_seq_length": 128,
        "max_tokens_per_batch": 128,
        "max_concurrent_adapters": 1
    }
END
)
echo "$json_config" > /tmp/peft_config.json
python ./inference/python/ff_peft.py -config-file /tmp/peft_config.json
# Check alignment
python ./tests/peft/peft_alignment_test.py -tp 4 -lr 0.001

# C++ test
echo "C++ test"
./build/inference/peft/peft \
    -ll:gpu 4 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 4 \
    -ll:fsize 8192 -ll:zsize 12000 \
    --max-requests-per-batch 1 \
    --max-sequence-length 128 \
    --max-tokens-per-batch 128 \
    -llm-model JackFram/llama-160m \
    -finetuning-dataset ./inference/prompt/peft_dataset.json \
    -peft-model goliaro/llama-160m-lora \
    -enable-peft \
    --use-full-precision \
    --inference-debugging
# Check alignment
python ./tests/peft/peft_alignment_test.py -tp 4 -lr 0.001

# Print succeess message
echo ""
echo "PEFT tests passed!"
echo ""

# Cleanup after the test
cleanup
