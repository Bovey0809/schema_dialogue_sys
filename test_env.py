import os

import torch

assert torch.cuda.is_available() is True

print(os.environ['CUDA_VISIBLE_DEVICES'])