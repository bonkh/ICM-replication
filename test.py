import torch
import numpy as np
import gc

# Define the fallback logic
def try_gpu_then_numpy(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"[OOM] Falling back to NumPy for: {fn.__name__ if hasattr(fn, '__name__') else str(fn)}")
            torch.cuda.empty_cache()
            gc.collect()

            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif isinstance(x, (list, tuple)):
                    return type(x)(to_numpy(i) for i in x)
                elif isinstance(x, dict):
                    return {k: to_numpy(v) for k, v in x.items()}
                else:
                    return x

            def to_tensor(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x)
                elif isinstance(x, (list, tuple)):
                    return type(x)(to_tensor(i) for i in x)
                elif isinstance(x, dict):
                    return {k: to_tensor(v) for k, v in x.items()}
                else:
                    return x

            args_np = to_numpy(args)
            kwargs_np = to_numpy(kwargs)

            try:
                result_np = fn(*args_np, **kwargs_np)
                result_tensor = to_tensor(result_np)
                return result_tensor
            except Exception as e_np:
                print(f"[ERROR] NumPy fallback also failed: {e_np}")
                raise e_np
        else:
            raise


# ------------------------------
# Simulate example usage
# ------------------------------

# Create very large tensors to force CUDA OOM (adjust size based on your GPU)
try:
    a = torch.randn((30000, 30000), device='cuda')  # Large tensor
    b = torch.randn((30000, 30000), device='cuda')

    # Define a pure function like addition
    def safe_add(x, y):
        return torch.add(x, y)

    # Run with fallback
    result = try_gpu_then_numpy(safe_add, a, b)

    print("Result shape:", result.shape)
    print("Result device:", result.device)
except RuntimeError as e:
    print("[ERROR] Operation failed completely:", str(e))
