import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm #progressbar
from timeit import default_timer as timer
from torch.utils.data import DataLoader

import helpers
from baseModels import BeeModelV0

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

# training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    helpers.train_step(
        data_loader=train_dataloader,
        model=model_0,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=helpers.accuracy_fn
    )
    helpers.test_step(data_loader=test_dataloader,
        model=model_0,
        loss_fn=loss_fn,
        accuracy_fn=helpers.accuracy_fn
    )

# Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = helpers.print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))

model_0_results = helpers.eval_model(model_0, eval_dataloader, loss_fn, helpers.accuracy_fn, device)
print(model_0_results)
