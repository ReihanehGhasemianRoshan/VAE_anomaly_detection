import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    )

def rand_dataset(num_rows=60_000, num_columns=100) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))