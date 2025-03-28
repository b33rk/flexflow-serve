import argparse
import os
import torch
import numpy as np


def compare_tensors(tensor1, tensor2, rtol=1e-2, atol=1e-2):
    """
    Compare two tensors and return True if they are equal within tolerance.
    
    Args:
        tensor1: First PyTorch tensor
        tensor2: Second PyTorch tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        bool: True if tensors are close, False otherwise
    """
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False

    # Check if values are close
    if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        return True
    else:
        # Calculate statistics about differences
        abs_diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        
        print(f"Tensor values differ:")
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # show the 2 entries with the largest difference
        print(f"Entries with largest difference:")
        print(f"tensor1: {tensor1.flatten()[abs_diff.argmax()]}")
        print(f"tensor2: {tensor2.flatten()[abs_diff.argmax()]}")
        return False


def load_and_compare_tensors(path1, path2, rtol=1e-5, atol=1e-8):
    """
    Load two tensors from given paths and compare them.
    
    Args:
        path1: Path to first tensor
        path2: Path to second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        bool: True if tensors are close, False otherwise
    """
    # 
    
    tensor1 = torch.jit.load(path1)
    tensor1 = list(tensor1.parameters())[0]
    


    # reorder the last two dimensions of tensor1
    tensor1 = tensor1.permute(0, 1, 2, 3)

    # merge the last two dimensions of tensor1
    tensor1 = tensor1.reshape(tensor1.shape[0], tensor1.shape[1], -1)

    # # squeeze the first dimension of tensor1
    tensor1 = tensor1.squeeze(0)

    print(f"Loading tensor from: {path1}, {tensor1.shape}")
    
    tensor2 = torch.jit.load(path2)
    tensor2 = list(tensor2.parameters())[0]
    print(f"Loading tensor from: {path2}, {tensor2.shape}")
    
    print("Comparing tensors...")
    result = compare_tensors(tensor1, tensor2, rtol=rtol, atol=atol)
    
    if result:
        print("✅ Tensors match within tolerance")
    else:
        print("❌ Tensors do not match within tolerance")
        print(f"tensor1: {tensor1}")
        print(f"tensor2: {tensor2}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare two PyTorch tensors")
    parser.add_argument("path1", type=str, help="Path to first tensor")
    parser.add_argument("path2", type=str, help="Path to second tensor")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    
    args = parser.parse_args()
    
    # Verify files exist
    for path in [args.path1, args.path2]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
    
    # Load and compare tensors
    success = load_and_compare_tensors(args.path1, args.path2, args.rtol, args.atol)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

# align dv resutls from non-flash and flash flexflow
# python3 /root/flexflow-serve/tests/align/peft_flash_attn/align_2_tensors_from_pt.py /root/flexflow-serve/tests/align/peft_flash_attn/dv/layers.11.layers.11.self_attn.dv.pt /root/flexflow-serve/tests/align/peft_flash_attn/dv/layers.11.layers.11.self_attn.v_proj.input_gradient_0.pt

'''
Loading tensor from: /root/flexflow-serve/tests/align/peft_flash_attn/dv/layers.11.layers.11.self_attn.dv.pt, torch.Size([24, 768])
Loading tensor from: /root/flexflow-serve/tests/align/peft_flash_attn/dv/layers.11.layers.11.self_attn.v_proj.input_gradient_0.pt, torch.Size([24, 768])
Comparing tensors...
Tensor values differ:
Max absolute difference: 0.75
Mean absolute difference: 0.00576019287109375
Entries with largest difference:
tensor1: 63.75
tensor2: 64.5
❌ Tensors do not match within tolerance
tensor1: tensor([[ 8.1641e+00, -1.1852e+01, -8.3281e+00,  ...,  2.3156e+01,
          1.2852e+00,  1.8938e+01],
        [-5.6446e-05, -9.1362e-04, -4.9174e-05,  ..., -1.1292e-03,
          5.9700e-04, -1.6212e-03],
        [ 2.9802e-05, -6.7472e-05, -3.4690e-05,  ..., -3.1590e-06,
          5.3883e-05, -5.0187e-05],
        ...,
        [ 4.1723e-07, -1.7881e-06,  1.2875e-05,  ..., -6.5565e-07,
          1.2040e-05, -5.0068e-06],
        [ 2.9802e-07,  1.6093e-06,  5.4836e-06,  ..., -3.0398e-06,
          1.0431e-05,  8.8811e-06],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.float16)
tensor2: tensor([[ 8.2812e+00, -1.1664e+01, -8.4062e+00,  ...,  2.3500e+01,
          1.2734e+00,  1.8953e+01],
        [-4.9770e-05, -9.4318e-04, -6.5148e-05,  ..., -1.1272e-03,
          5.6314e-04, -1.6165e-03],
        [ 2.9266e-05, -6.7711e-05, -3.4332e-05,  ..., -2.5034e-06,
          5.2869e-05, -5.0187e-05],
        ...,
        [ 4.1723e-07, -1.7881e-06,  1.2875e-05,  ..., -6.5565e-07,
          1.2159e-05, -5.0068e-06],
        [ 3.5763e-07,  1.6093e-06,  5.4240e-06,  ..., -3.0994e-06,
          1.0550e-05,  9.0003e-06],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.float16)
'''
