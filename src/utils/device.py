import torch

def get_device():
    print("PyTorch version:", torch.__version__)
    print("is cuda available:", torch.cuda.is_available())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")