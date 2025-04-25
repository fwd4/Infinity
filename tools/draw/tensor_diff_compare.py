import torch

def load_pkl_tensors(file_path):
    """Load tensors from a pickle file."""
    return torch.load(file_path)

def compare_tensors(pkl_file1, pkl_file2):
    """Compare tensors from two pickle files."""
    tensors1 = load_pkl_tensors(pkl_file1)
    tensors2 = load_pkl_tensors(pkl_file2)
    
    if len(tensors1) != len(tensors2):
        raise ValueError(f"Number of tensors mismatch: {len(tensors1)} vs {len(tensors2)}")
    
    for i, (t1, t2) in enumerate(zip(tensors1, tensors2)):
        try:
            torch.testing.assert_close(t1, t2, rtol=1e-5, atol=1e-5)
            print(f"Tensor {i}: Match ✓")
        except AssertionError as e:
            import pdb; pdb.set_trace()
            print(f"Tensor {i}: Mismatch ✗")
            print(f"Shape: {t1.shape}")
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python tensor_diff_compare.py file1.pkl file2.pkl")
        sys.exit(1)
        
    compare_tensors(sys.argv[1], sys.argv[2])