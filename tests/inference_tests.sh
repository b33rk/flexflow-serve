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
# cat << 'EOF' > ./inference/prompt/test.json
# [
#     "The largest ocean on Earth is",
#     "The inventor of the telephone was",
#     "The speed of light is",
#     "The tallest mountain in the world is",
#     "The first man on the moon was"
# ]
# EOF
# cat << 'EOF' > ./inference/prompt/test.json
# [
#   "In the year 2075, artificial intelligence has become deeply integrated into every aspect of human life. Autonomous robots manage infrastructure, AI-powered doctors perform complex surgeries with unmatched precision, and personalized AI assistants anticipate people's needs before they even express them. Despite these advancements, ethical concerns continue to grow. One of the most pressing debates surrounding AI development in this era is whether",
#   "The rapid development of space exploration has led humanity to establish permanent settlements beyond Earth. With bases on the Moon and Mars, scientists and engineers work tirelessly to create sustainable ecosystems that can support human life in the long term. However, numerous challenges remain, from radiation exposure to psychological effects of isolation in deep space. One of the most critical issues that must be addressed before humanity can expand further into the solar system is",
#   "Throughout history, scientific discoveries have continuously reshaped our understanding of the universe. The shift from a geocentric to a heliocentric model, the theory of relativity, and the advent of quantum mechanics have all challenged previous assumptions and opened new frontiers of knowledge. As we continue to explore the cosmos, scientists are now focused on solving one of the most perplexing mysteries of all: the nature of dark matter and dark energy. If researchers were to uncover definitive proof regarding their existence, it could mean that",
#   "The emergence of advanced genetic engineering techniques has revolutionized modern medicine, allowing scientists to edit DNA with unprecedented precision. With technologies like CRISPR, researchers have already corrected genetic mutations that cause severe diseases and are even exploring the potential of enhancing human traits such as intelligence and longevity. However, this progress raises profound ethical concerns, as the ability to manipulate the human genome could lead to unforeseen consequences. One of the major dilemmas in the future of genetic engineering revolves around",
#   "Climate change has become the defining challenge of the 21st century, with rising global temperatures, extreme weather events, and melting ice caps threatening ecosystems and human populations worldwide. Scientists and policymakers are racing against time to develop sustainable solutions, from carbon capture technologies to alternative energy sources like nuclear fusion. Despite these efforts, one of the biggest obstacles to achieving global climate stability is the fact that"
# ]
# EOF

# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

############## Run inference in flexflow-serve ##############

echo "Running inference in flexflow-serve (python)..."

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
        --max-length 255 \
        --prompt-file "${PWD}/inference/prompt/test.json" \
        --output-file "${PWD}/inference/output/huggingface_$model_name_.json"
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
