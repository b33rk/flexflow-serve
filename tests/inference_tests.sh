#! /usr/bin/env bash
set -x
set -e

# Cd into root directory of repo
cd "${BASH_SOURCE[0]%/*}/.."

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Enable additional C++ tests, (off by default)
CPP_INFERENCE_TESTS=${CPP_INFERENCE_TESTS:-OFF}

# Clean up before test
rm -rf inference/prompt inference/output inference/inf_test_configs || true
# Create test prompt file
mkdir -p ./inference/prompt
echo '["Three tips for staying healthy are: "]' > ./inference/prompt/test.json
# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

############## Run inference in flexflow-serve ##############

echo "Running inference in flexflow-serve..."

# Generate test configs
rm -rf ./inference/inf_test_configs/*.json || true
python ./tests/inference/generate_inf_test_configs.py

# Loop through .json files in the ./inference/inf_test_configs dir 
for file in ./inference/inf_test_configs/*.json; do
    # Check filename prefix
    if [[ $file == *"incr_dec"* ]]; then
      script="./inference/python/incr_decoding.py"
    elif [[ $file == *"spec_infer"* ]]; then  
      script="./inference/python/spec_infer.py"
    fi
    # Run script
    python "$script" -config-file "$file" 
done

##############  Run inference in HuggingFace ##############

echo "Running inference in huggingface..."

model_names=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "facebook/opt-6.7b"
    "facebook/opt-125m"
)
for model_name in "${model_names[@]}"; do
    # set model_name_ to the content of model_name after the first "/", transformed into lowercase
    model_name_=$(echo "$model_name" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    python ./tests/inference/huggingface_inference.py \
        --model-name "$model_name" \
        --prompt-file "${PWD}/inference/prompt/test.json" \
        --output-file "${PWD}/inference/output/huggingface_$model_name_.txt"
done

##############  Check alignment between results ##############
echo "Checking alignment of results..."
pytest -v ./tests/inference/test_inference_output.py


############## Run additional C++ inference tests if enabled ##############
if [[ "$CPP_INFERENCE_TESTS" == "ON" ]]; then
    # Manually download the weights in both half and full precision
    echo "Running additional C++ inference tests..."
    ./tests/inference/cpp_inference_tests.sh
fi

echo "All tests passed!"
