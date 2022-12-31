import torch
standard = torch.device("cuda" if torch.cuda.is_available() else "cpu")