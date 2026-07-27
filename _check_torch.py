import torch
print("torch.Tensor exists:", hasattr(torch, "Tensor"))
print("torch.nn exists:", hasattr(torch, "nn"))
try:
    import torch.nn as nn
    print("torch.nn imported OK")
    print("nn.Module exists:", hasattr(nn, "Module"))
except Exception as e:
    print("torch.nn import failed:", e)
print("HAS_TORCH would be:", hasattr(torch, "nn"))
