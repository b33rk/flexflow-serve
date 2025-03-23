# align flexflow intermediate attn results to python flash-attn library results
# the tensor loading depends on the flexflow debug directory+file structure

import torch
from flash_attn import flash_attn_func
import math
import os
from collections import defaultdict


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
    for step_dir in os.listdir(base_path):
        print(f"Processing directory: {step_dir}")
        
        try:
            if not step_dir.startswith("bwd"):
                print(f"Skipping non-bwd directory: {step_dir}")
                continue

            parts = step_dir.split("_")
            if len(parts) < 2:
                print(f"Warning: Unexpected directory name format: {step_dir}")
                continue
                
            step_id = int(parts[1])
            step_path = os.path.join(base_path, step_dir)
            print(f"Found step {step_id} at {step_path}")
            print(f"Step directory contents: {os.listdir(step_path)}")

            for shard_dir in os.listdir(step_path):
                try:
                    shard_parts = shard_dir.split("_")
                    if len(shard_parts) < 2:
                        print(f"Warning: Unexpected shard directory format: {shard_dir}")
                        continue
                        
                    shard_id = int(shard_parts[1])  # Extract shard number
                    shard_path = os.path.join(step_path, shard_dir)
                    print(f"Processing shard {shard_id} at {shard_path}")
                    print(f"Shard directory contents: {os.listdir(shard_path)}")

                    for file_name in os.listdir(shard_path):
                        try:
                            # Parse layer number and tensor type
                            parts = file_name.split(".")
                            if len(parts) < 5 or not parts[0].startswith("layers"):
                                continue

                            layer_id = int(parts[1])
                            # Extract tensor type (e.g., fwd_q, bwd_k, etc.)
                            tensor_type = ".".join(parts[3:]).replace(".pt", "")
                            if tensor_type not in tensor_types:
                                continue

                            # Load tensor using torch.jit.load
                            tensor_path = os.path.join(shard_path, file_name)
                            try:
                                tensor = torch.jit.load(tensor_path)
                                tensor = list(tensor.parameters())[0]
                                tensors[step_id][shard_id][layer_id][tensor_type] = tensor
                                print(f"Successfully loaded tensor: {tensor_type} for step {step_id}, shard {shard_id}, layer {layer_id}")
                            except Exception as e:
                                print(f"Error loading tensor from {tensor_path}: {e}")
                        except Exception as e:
                            print(f"Error processing file {file_name}: {e}")
                except Exception as e:
                    print(f"Error processing shard directory {shard_dir}: {e}")
        except Exception as e:
            print(f"Error processing step directory {step_dir}: {e}")

    if not tensors:
        print("Warning: No tensors were loaded. Please check if the directory structure and file naming are correct.")
    else:
        print(f"Successfully loaded tensors for {len(tensors)} steps")
        for step_id in tensors:
            print(f"Step {step_id}: {len(tensors[step_id])} shards")
            for shard_id in tensors[step_id]:
                print(f"  Shard {shard_id}: {len(tensors[step_id][shard_id])} layers")

    return tensors


def check_closeness_between_forward_and_backward_pass(
    fwd_tensor, bwd_tensor, tensor_type, atol=1e-5, rtol=1e-5
):
    comparison = torch.allclose(fwd_tensor, bwd_tensor, atol=atol, rtol=rtol)
    if not comparison:
        print(f"Difference found between {tensor_type} from forward and backward pass")
        return False
    return True


def flash_attention(q, k, v, is_causal, dout, alibi_slopes=None):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

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

    # Permute to match our manual implementation's output shape [batch_size, num_heads, seqlen_q, head_size]
    flash_attn_out_permuted = flash_attn_out.permute(0, 2, 1, 3)

    # backward pass
    flash_attn_out.backward(dout)

    return flash_attn_out_permuted, flash_softmax_lse, q.grad, k.grad, v.grad


def check_closeness_between_flexflow_and_flash_attn(
    flexflow_tensor, flash_attn_tensor, tensor_type, atol=1e-5, rtol=1e-5
):
    comparison = torch.allclose(
        flexflow_tensor, flash_attn_tensor, atol=atol, rtol=rtol
    )
    if not comparison:
        print(f"Difference found between {tensor_type} from flexflow and flash-attn")
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

    # Iterate through steps, shards, and layers
    for step_id in tensors:
        for shard_id in tensors[step_id]:
            for layer_id in tensors[step_id][shard_id]:
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
                        return False

                    softmax_lse_closeness = (
                        check_closeness_between_forward_and_backward_pass(
                            fwd_softmax_lse, bwd_softmax_lse, "softmax_lse"
                        )
                    )
                    if not softmax_lse_closeness:
                        return False

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
                        return False

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
                        return False

                    # check softmax_lse closeness
                    softmax_lse_closeness = (
                        check_closeness_between_flexflow_and_flash_attn(
                            fwd_softmax_lse, flash_softmax_lse, "softmax_lse"
                        )
                    )
                    if not softmax_lse_closeness:
                        return False

                    # check dq closeness
                    dq_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dq, flash_dq, "dq"
                    )
                    if not dq_closeness:
                        return False

                    # check dk closeness
                    dk_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dk, flash_dk, "dk"
                    )
                    if not dk_closeness:
                        return False

                    # check dv closeness
                    dv_closeness = check_closeness_between_flexflow_and_flash_attn(
                        bwd_dv, flash_dv, "dv"
                    )
                    if not dv_closeness:
                        return False

                    return True


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
