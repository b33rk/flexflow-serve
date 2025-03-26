import argparse
import os
import torch
import numpy as np


def compare_tensors(tensor1, tensor2, rtol=1e-5, atol=1e-8):
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
    tensor1 = tensor1.permute(0, 1, 3, 2)

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
