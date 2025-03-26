# align flexflow intermediate attn results to python flash-attn library results
# the tensor loading depends on the flexflow debug directory+file structure

import torch
from flash_attn import flash_attn_func
import math
import os
from collections import defaultdict
import re


def env_check():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")


def load_tensor_from_flexflow(base_path="/root/.cache/flexflow/debug/flexflow"):
    """
    Load FlexFlow attention tensors from .pt files and organize them in a nested dictionary.

    Args:
        base_path (str): Base path to the FlexFlow debug directory

    Returns:
        dict: A nested dictionary with structure:
            {step_id: {shard_id: {layer_id: {tensor_type: torch.Tensor}}}}
    """
    # Create nested defaultdict for storing tensors
    tensors = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Check if base directory exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        print("Please ensure the FlexFlow debug directory is present")
        return tensors

    print(f"Scanning directory: {base_path}")
    print(f"Found contents: {os.listdir(base_path)}")

    # Define the pattern for different tensor types
    tensor_types = [
        "self_attn.fwd_q",
        "self_attn.fwd_k",
        "self_attn.fwd_v",
        "self_attn.fwd_out",
        "self_attn.fwd_softmax_lse",
        "self_attn.bwd_q",
        "self_attn.bwd_k",
        "self_attn.bwd_v",
        "self_attn.bwd_softmax_lse",
        "self_attn.dq",
        "self_attn.dk",
        "self_attn.dv",
        "self_attn.dout",
        "self_attn.fwd_alibi_slopes",
        "self_attn.bwd_alibi_slopes",  # place holder for alibi slopes
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
                    if full_tensor_type not in tensor_types:
                        # print(f"Skipping unknown tensor type: {full_tensor_type}")
                        continue

                    # Load tensor using torch.jit.load
                    tensor_path = os.path.join(shard_path, file_name)
                    tensor = torch.jit.load(tensor_path)
                    tensor = list(tensor.parameters())[0]

                    # Store with combined layer ID
                    tensors[step_id][shard_id][layer_id][full_tensor_type] = tensor
                    # print(
                    #     f"Successfully loaded tensor: {full_tensor_type} for step {step_id}, "
                    #     f"shard {shard_id}, layer {layer_id}"
                    # )

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


def check_closeness_between_forward_and_backward_pass(
    fwd_tensor, bwd_tensor, tensor_type, atol=1e-5, rtol=1e-5
):
    # check if both fwd_tensor and bwd_tensor have overflow
    if torch.isnan(fwd_tensor).any() and torch.isnan(bwd_tensor).any():
        print(f"Overflow found in {tensor_type} from forward and backward pass")
        return False

    comparison = torch.allclose(fwd_tensor, bwd_tensor, atol=atol, rtol=rtol)
    if not comparison:
        print(f"Difference found between {tensor_type} from forward and backward pass")
        print(f"fwd_tensor: {fwd_tensor}")
        print(f"bwd_tensor: {bwd_tensor}")
        return False
    return True


def flash_attention(q, k, v, is_causal, dout, alibi_slopes=None):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

    # print shape
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"v shape: {v.shape}")

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    head_size = q.shape[-1]

    # Define scaling factor
    scaling_factor = 1.0 / math.sqrt(head_size)

    # Default softmax scale in flash_attn_func is 1/sqrt(head_size)
    flash_attn_out, flash_softmax_lse, S_dmask = flash_attn_func(
        q,
        k,
        v,
        causal=is_causal,
        softmax_scale=scaling_factor,  # Explicitly pass the scaling factor
        return_attn_probs=True,
        alibi_slopes=alibi_slopes,
    )

    # # Permute to match our manual implementation's output shape [batch_size, num_heads, seqlen_q, head_size]
    # flash_attn_out_permuted = flash_attn_out.permute(0, 2, 1, 3)

    # backward pass
    flash_attn_out.backward(dout)

    return flash_attn_out, flash_softmax_lse, q.grad, k.grad, v.grad


def check_closeness_between_flexflow_and_flash_attn(
    flexflow_tensor, flash_attn_tensor, tensor_type, atol=1e-5, rtol=1e-5
):
    # check if both flexflow_tensor and flash_attn_tensor have overflow
    if torch.isnan(flexflow_tensor).any() and torch.isnan(flash_attn_tensor).any():
        print(f"Overflow found in {tensor_type} from flexflow and flash-attn")
        return False

    comparison = torch.allclose(
        flexflow_tensor, flash_attn_tensor, atol=atol, rtol=rtol
    )
    # if tensor_type == "dq" or tensor_type == "dk" or tensor_type == "dv":
    #     print(f"FlexFlow tensor of {tensor_type}: {flexflow_tensor}")
    #     print(f"Flash-attn tensor of {tensor_type}: {flash_attn_tensor}")
        
    if not comparison:
        print(f"Difference found between {tensor_type} from flexflow and flash-attn")
        print(f"FlexFlow tensor: {flexflow_tensor}")
        print(f"Flash-attn tensor: {flash_attn_tensor}")
        return False
    return True


def perform_closeness_test(tensors):
    """
    Compare FlexFlow attention results with flash-attention results.

    Args:
        tensors: Nested dictionary containing tensors from FlexFlow
            {step_id: {shard_id: {layer_id: {tensor_type: torch.Tensor}}}}
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_success = True

    # Iterate through steps, shards, and layers
    for step_id in sorted(tensors.keys()):
        for shard_id in sorted(tensors[step_id].keys()):
            for layer_id in sorted(tensors[step_id][shard_id].keys()):
                layer_tensors = tensors[step_id][shard_id][layer_id]

                print(f"\nTesting step {step_id}, shard {shard_id}, layer {layer_id}")

                # Forward pass test
                if all(
                    k in layer_tensors
                    for k in [
                        "self_attn.fwd_q",
                        "self_attn.fwd_k",
                        "self_attn.fwd_v",
                        "self_attn.fwd_out",
                        "self_attn.fwd_softmax_lse",
                        "self_attn.dout",
                        "self_attn.bwd_q",
                        "self_attn.bwd_k",
                        "self_attn.bwd_v",
                        "self_attn.bwd_softmax_lse",
                        "self_attn.dq",
                        "self_attn.dk",
                        "self_attn.dv",
                    ]
                ):
                    print("Computing by flash-attn...")

                    # Get input tensors from forward pass
                    fwd_q = layer_tensors["self_attn.fwd_q"].to(device)
                    fwd_k = layer_tensors["self_attn.fwd_k"].to(device)
                    fwd_v = layer_tensors["self_attn.fwd_v"].to(device)
                    fwd_alibi_slopes = None
                    # skip alibi slopes if not present
                    if "self_attn.fwd_alibi_slopes" in layer_tensors:
                        fwd_alibi_slopes = layer_tensors[
                            "self_attn.fwd_alibi_slopes"
                        ].to(device)

                    # Get output tensors from forward pass
                    fwd_out = layer_tensors["self_attn.fwd_out"].to(device)
                    fwd_softmax_lse = layer_tensors["self_attn.fwd_softmax_lse"].to(
                        device
                    )

                    # Get input tensors from backward pass
                    dout = layer_tensors["self_attn.dout"].to(device)
                    bwd_q = layer_tensors["self_attn.bwd_q"].to(device)
                    bwd_k = layer_tensors["self_attn.bwd_k"].to(device)
                    bwd_v = layer_tensors["self_attn.bwd_v"].to(device)
                    bwd_softmax_lse = layer_tensors["self_attn.bwd_softmax_lse"].to(
                        device
                    )
                    bwd_alibi_slopes = None
                    # skip alibi slopes if not present
                    if "self_attn.bwd_alibi_slopes" in layer_tensors:
                        bwd_alibi_slopes = layer_tensors[
                            "self_attn.bwd_alibi_slopes"
                        ].to(device)

                    # Get output tensors from backward pass
                    bwd_dq = layer_tensors["self_attn.dq"].to(device)
                    bwd_dk = layer_tensors["self_attn.dk"].to(device)
                    bwd_dv = layer_tensors["self_attn.dv"].to(device)

                    # check closeness of the same tensors from forward and backward pass
                    q_closeness = check_closeness_between_forward_and_backward_pass(
                        fwd_q, bwd_q, "q"
                    )
                    k_closeness = check_closeness_between_forward_and_backward_pass(
                        fwd_k, bwd_k, "k"
                    )
                    v_closeness = check_closeness_between_forward_and_backward_pass(
                        fwd_v, bwd_v, "v"
                    )

                    if not q_closeness or not k_closeness or not v_closeness:
                        # return False
                        test_success = False

                    softmax_lse_closeness = (
                        check_closeness_between_forward_and_backward_pass(
                            fwd_softmax_lse, bwd_softmax_lse, "softmax_lse"
                        )
                    )
                    if not softmax_lse_closeness:
                        # return False
                        test_success = False

                    # check alibi slopes closeness
                    if fwd_alibi_slopes is not None and bwd_alibi_slopes is not None:
                        alibi_slopes_closeness = (
                            check_closeness_between_forward_and_backward_pass(
                                fwd_alibi_slopes, bwd_alibi_slopes, "alibi_slopes"
                            )
                        )
                        if not alibi_slopes_closeness:
                            return False
                    elif fwd_alibi_slopes is not None or bwd_alibi_slopes is not None:
                        print(
                            "Difference found in alibi slopes tensors between forward and backward pass: only one of the tensors is present"
                        )
                        # return False
                        test_success = False

                    print(f"step {step_id}, shard {shard_id}, layer {layer_id}: fwd and bwd buffer matched")
                    # Run flash attention pass (package from pip install flash-attn)
                    is_causal = True
                    flash_out, flash_softmax_lse, flash_dq, flash_dk, flash_dv = (
                        flash_attention(
                            fwd_q,
                            fwd_k,
                            fwd_v,
                            is_causal=is_causal,
                            dout=dout,
                            alibi_slopes=fwd_alibi_slopes,
                        )
                    )

                    # check output closeness
                    out_closeness = check_closeness_between_flexflow_and_flash_attn(
                        fwd_out, flash_out, "out"
                    )
                    if not out_closeness:
                        test_success = False

                    # check softmax_lse closeness
                    softmax_lse_closeness = (
                        check_closeness_between_flexflow_and_flash_attn(
                            fwd_softmax_lse, flash_softmax_lse, "softmax_lse"
                        )
                    )
                    if not softmax_lse_closeness:
                        test_success = False

                    # check dq closeness
                    dq_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dq, flash_dq, "dq"
                    )
                    if not dq_closeness:
                        test_success = False

                    # check dk closeness
                    dk_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dk, flash_dk, "dk"
                    )
                    if not dk_closeness:
                        test_success = False

                    # check dv closeness
                    dv_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dv, flash_dv, "dv"
                    )
                    if not dv_closeness:
                        test_success = False

    return test_success


if __name__ == "__main__":
    env_check()

    # load tensors
    tensors = load_tensor_from_flexflow()

    # closeness test
    success = perform_closeness_test(tensors)
    if success:
        print("Closeness test passed")
    else:
        print("Closeness test failed")
