import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
from dataset import BeeDataset
from pathlib import Path
import helpers
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm #progressbar
from timeit import default_timer as timer

random.seed("woof")
torch.manual_seed(420)
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu" 
data_path = Path("../data")
train_dir = data_path / "train"
test_dir = data_path / "test"
eval_dir = data_path / "val"
image_path_list = list(data_path.glob("*/*/*.png"))

print(train_dir) 
print(test_dir)

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

train_data = datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,transform=transform)
eval_data = datasets.ImageFolder(root=eval_dir,transform=transform)

# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=32, # how many samples per batch?
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=64, 
                             shuffle=False)

eval_dataloader = DataLoader(dataset=eval_data, 
                             batch_size=64, 
                             shuffle=False)

# BASELINE MODEL
flatten_model = nn.Flatten()

class BeeModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

input_shape = 3*32*32
hidden_units = 10
output_shape = len(train_data.classes)

model_0 = BeeModelV0(
        input_shape=input_shape, 
        hidden_units=hidden_units,
        output_shape=output_shape
        )

model_0.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model_0.parameters(), lr=1e-2)

# TRAINING
# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 10

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss = 0 
    train_acc = 0
    model.to(device)

    for batch, (X, y) in enumerate(train_dataloader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. fwd pass
        y_pred = model(X)

        # 2. loss per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        # 3. optim zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward() #backprop

        # 5. optimizer step
        optimizer.step()
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            # 1. Fwd pass
            test_pred = model(X)

            # 2. calc loss
            test_loss += loss_fn(test_pred, y)

            # 3. calc accuracy
            test_acc += helpers.accuracy_fn(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_step(
        data_loader=train_dataloader,
        model=model_0,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=helpers.accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_0,
        loss_fn=loss_fn,
        accuracy_fn=helpers.accuracy_fn
    )

# Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = helpers.print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))

model_0_results = eval_model(model_0, eval_dataloader, loss_fn, helpers.accuracy_fn, device)
print(model_0_results)
