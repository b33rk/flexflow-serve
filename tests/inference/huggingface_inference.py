import argparse
import json
import os
import shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    GenerationConfig,
)
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "peft"))
from hf_utils import *

def main():
    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=255)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--do-sample", action="store_true", help="Use sampling")
    parser.add_argument(
        "--inference-debugging",
        action="store_true",
        help="Print debugging info and save hidden states/weights to file",
    )
    args = parser.parse_args()
    # Check if max-length is greater than 0
    if args.max_length <= 0:
        print("Error: max-length must be greater than 0.")
        return
    # Check if prompt-file exists
    if not os.path.isfile(args.prompt_file):
        print(f"Error: {args.prompt_file} does not exist.")
        return

    # Read prompt-file into a list of strings
    with open(args.prompt_file, "r") as f:
        try:
            prompt_list = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Unable to parse {args.prompt_file} as JSON.")
            return

    # Set default tensor type depending on argument indicating the float type to use
    if not args.use_full_precision:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)
    
    # Run huggingface model
    # Get Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, attn_implementation="eager", device_map="auto")
    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model_name)
    generation_config.do_sample = args.do_sample
    if not args.do_sample:
        generation_config.num_beams=1
        generation_config.temperature = None
        generation_config.top_p = None
    ################# debugging #################
    if args.inference_debugging:
        # Print model and configs
        print(hf_config)
        print(model)
        make_debug_dirs()
        register_inference_hooks(model)
        # Save weights
        save_model_weights(model, target_modules=["lora", "lm_head", "final_layer_norm", "self_attn_layer_norm", "out_proj", "fc1", "fc2"])

    ###############################################
    # Generate output
    output_list = []
    for i, prompt in enumerate(prompt_list):
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        generated = model.generate(
            batch["input_ids"],
            max_length=args.max_length,
            generation_config=generation_config,
        )
        prompt_token_ids = list(batch["input_ids"].cpu().numpy()[0])
        response_token_ids = list(generated[0].cpu().numpy())[len(prompt_token_ids):]
        # Remove eos token if present at the end
        if response_token_ids[-1] == tokenizer.eos_token_id:
            response_token_ids = response_token_ids[:-1]
        response = tokenizer.decode(response_token_ids)
        output_list.append({
            "req_idx": i,
            "req_type": "inference",
            "prompt_length": len(prompt_token_ids),
            "response_length": len(response_token_ids),
            "prompt": prompt,
            "response": response,
            "input_tokens": ",".join(str(x) for x in prompt_token_ids),
            "output_tokens": ",".join(str(x) for x in response_token_ids),
            "num_decoding_steps": len(response_token_ids),
        })
    with open(args.output_file, "w") as f:
        json.dump(output_list, f, indent=2)


if __name__ == "__main__":
    main()
