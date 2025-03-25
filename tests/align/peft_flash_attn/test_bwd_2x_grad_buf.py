# test whether we need to copy dq, dk, dv to device ptr


import torch
import os
from collections import defaultdict
import re

def load_tensor_from_flexflow(base_path="/root/.cache/flexflow/debug/flexflow"):
    """
    Load FlexFlow pre_dq2, pre_dk2, pre_dv2, post_dq2, post_dk2, post_dv2 tensors from .pt files and organize them in a nested dictionary.

    Args:
        base_path (str): Base path to the FlexFlow debug directory

    Returns:
        dict: A nested dictionary with structure:
            {step_id: {shard_id: {layer_id: {tensor_type: torch.Tensor}}}}
    """
    # Create nested defaultdict for storing tensors
    # Use a defaultdict that creates a defaultdict for each tensor type with a dict for pre/post
    tensors = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # Check if base directory exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        print("Please ensure the FlexFlow debug directory is present")
        return tensors

    print(f"Scanning directory: {base_path}")
    print(f"Found contents: {os.listdir(base_path)}")

    # Define the pattern for different tensor types
    tensor_types = [
        "pre_dq2",
        "pre_dk2",
        "pre_dv2",
        "post_dq2",
        "post_dk2",
        "post_dv2",
    ]

    # Walk through the directory structure
    for root_dir in os.listdir(base_path):
        print(f"Processing directory: {root_dir}")

        # the attn tensors are only present in the bwd directory
        if root_dir != "bwd":
            print(f"Skipping non-bwd directory: {root_dir}")
            continue

        root_path = os.path.join(base_path, root_dir)
        print(f"Found bwd directory at {root_path}")

        # Look for step directories
        for step_dir in os.listdir(root_path):
            if not step_dir.startswith("step_"):
                print(f"Skipping non-step directory: {step_dir}")
                continue

            # Extract step number from "step_X" or "step_X_pre"
            step_num = step_dir.split("_")[1]
            try:
                step_id = int(step_num)
            except ValueError:
                print(f"Warning: Could not parse step number from {step_dir}")
                continue

            step_path = os.path.join(root_path, step_dir)
            print(f"Found step {step_id} at {step_path}")
            # print(f"Step directory contents: {os.listdir(step_path)}")

            for shard_dir in os.listdir(step_path):
                if not shard_dir.startswith("shard_"):
                    print(f"Skipping non-shard directory: {shard_dir}")
                    continue

                shard_id = int(shard_dir.split("_")[1])  # Extract shard number
                shard_path = os.path.join(step_path, shard_dir)
                print(f"Processing shard {shard_id} at {shard_path}")

                for file_name in os.listdir(shard_path):
                    # Use regex to parse the file name
                    pattern = r"layers\.(\d+)\.layers\.(\d+)\.(self_attn)\.(.*?)\.pt$"
                    match = re.match(pattern, file_name)

                    if not match:
                        # print(f"Skipping file with unexpected format: {file_name}")
                        continue

                    outer_layer_id = int(match.group(1))
                    inner_layer_id = int(match.group(2))
                    module_name = match.group(3)
                    tensor_type = match.group(4)

                    # Combine layer IDs to create a unique layer identifier
                    layer_id = outer_layer_id

                    # Check if this is a tensor type we're interested in
                    full_tensor_type = f"{module_name}.{tensor_type}"
                    if tensor_type not in tensor_types:
                        print(f"Skipping unknown tensor type: {full_tensor_type}")
                        continue
                    else:
                        print(f"Loading tensor: {full_tensor_type}")

                    # Load tensor using torch.jit.load
                    tensor_path = os.path.join(shard_path, file_name)
                    tensor = torch.jit.load(tensor_path)
                    tensor = list(tensor.parameters())[0]

                    # Determine the tensor base type and index (pre=0, post=1)
                    if tensor_type.startswith("pre_"):
                        base_type = tensor_type[4:]  # Remove 'pre_' prefix
                        index = 0
                    elif tensor_type.startswith("post_"):
                        base_type = tensor_type[5:]  # Remove 'post_' prefix
                        index = 1
                    else:
                        print(f"Unknown tensor prefix in: {tensor_type}")
                        continue

                    # Store with combined layer ID, pair of pre and post tensors
                    tensors[step_id][shard_id][layer_id][base_type][index] = tensor
                    print(f"Loaded {tensor_type} for step {step_id}, shard {shard_id}, layer {layer_id}")

    if not tensors:
        print(
            "Warning: No tensors were loaded. Please check if the directory structure and file naming are correct."
        )
    else:
        print(f"Successfully loaded tensors for {len(tensors)} steps")
        # Sort steps
        for step_id in sorted(tensors.keys()):
            print(f"Step {step_id}: {len(tensors[step_id])} shards")
            # Sort shards
            for shard_id in sorted(tensors[step_id].keys()):
                print(f"  Shard {shard_id}: {len(tensors[step_id][shard_id])} layers")
                # Sort layers
                for layer_id in sorted(tensors[step_id][shard_id].keys()):
                    print(
                        f"    Layer {layer_id}: "
                        f"{len(tensors[step_id][shard_id][layer_id])} tensors"
                    )

    return tensors

def compare_tensors(tensors: dict, atol: float = 1e-5, rtol: float = 1e-5):
    """Compare two tensors element-wise"""
    for step_id in sorted(tensors.keys()):
        for shard_id in sorted(tensors[step_id].keys()):
            for layer_id in sorted(tensors[step_id][shard_id].keys()):
                for tensor_type in sorted(tensors[step_id][shard_id][layer_id].keys()):
                    if 0 not in tensors[step_id][shard_id][layer_id][tensor_type] or 1 not in tensors[step_id][shard_id][layer_id][tensor_type]:
                        print(f"Missing pre or post tensor for {tensor_type} at step {step_id}, shard {shard_id}, layer {layer_id}")
                        continue
                        
                    pre_tensor = tensors[step_id][shard_id][layer_id][tensor_type][0]
                    post_tensor = tensors[step_id][shard_id][layer_id][tensor_type][1]
                    assert pre_tensor.shape == post_tensor.shape
                    if not torch.allclose(pre_tensor, post_tensor, atol, rtol):
                        print(f"Mismatch in {tensor_type} for step {step_id}, shard {shard_id}, layer {layer_id}")
                        print(f"Pre tensor: {pre_tensor}")
                        print(f"Post tensor: {post_tensor}")
                    else:
                        print(f"Tensors match for {tensor_type} at step {step_id}, shard {shard_id}, layer {layer_id}")

if __name__ == "__main__":
    # load pre_dq2, pre_dk2, pre_dv2, post_dq2, post_dk2, post_dv2
    tensors = load_tensor_from_flexflow()
    # compare pre_dp2 with post_dp2, pre_dk2 with post_dk2, pre_dv2 with post_dv2
    compare_tensors(tensors)


    
    
    