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
    
    # w1, x1 = tensors1[0], tensors1[3]
    # w2, x2 = tensors2[0], tensors2[3]
    # x2_ = x2[:,:48*48,:]

    # L1 = torch.nn.Linear(1536, 4096, bias=False, device=x1.device)
    # L2 = torch.nn.Linear(1536, 4096, bias=False, device=x2.device)
    # L1.load_state_dict({"weight": w1})
    # L2.load_state_dict({"weight": w2})
    # import pdb; pdb.set_trace()
    # with torch.autocast(
    #         "cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True
    #     ):
    #     y1 = L1(x1)
    #     #y2 = L1(x2[:,:3072,:])
    #     y2 = L1(x2)
    #     y2_ = L1(x2_)

    #     torch.testing.assert_close(x1, x2_, rtol=1e-5, atol=1e-5)
    #     torch.testing.assert_close(y2_, y1, rtol=1e-5, atol=1e-5)
    #     torch.testing.assert_close(y2[:,:48*48,:], y1, rtol=1e-5, atol=1e-5)
    
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