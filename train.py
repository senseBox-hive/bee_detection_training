import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm.auto import tqdm #progressbar
from timeit import default_timer as timer

import os
from pathlib import Path
import random

import helpers
from baseModels import BeeModelV0

random.seed("woof")
torch.manual_seed(420)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu" 
data_path = Path("../data")
train_dir = data_path / "train"
test_dir = data_path / "test"
eval_dir = data_path / "val"
image_path_list = list(data_path.glob("*/*/*.png"))

def main():
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
    NUM_WORKERS = os.cpu_count()-1

    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=32, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=64,
                                num_workers=NUM_WORKERS,
                                shuffle=False)

    eval_dataloader = DataLoader(dataset=eval_data, 
                                batch_size=64,
                                num_workers=NUM_WORKERS, 
                                shuffle=False)



    # MODEL
    model_0 = BeeModelV0(
            input_shape=3*32*32, 
            hidden_units=10,
            output_shape=len(train_data.classes)
            )

    # TRAINING
    train_and_test(
        model_0, 
        train_dataloader, 
        test_dataloader, 
        eval_dataloader, 
        loss_fn = nn.CrossEntropyLoss(), 
        optimizer = optim.SGD(params=model_0.parameters(), lr=1e-2),
        device=DEVICE
    )
    

def train_and_test(model, train_dataloader, test_dataloader, eval_dataloader, loss_fn, optimizer, epochs=10, seed=42, device: torch.device = DEVICE):
    # Set the seed and start the timer
    model.to(device)
    torch.manual_seed(seed)
    train_time_start_on_cpu = timer()

    # training and testing loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")

        helpers.train_step(
            data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=helpers.accuracy_fn,
            device=device
        )
        helpers.test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=helpers.accuracy_fn,
            device=device
        )

    # Calculate training time      
    train_time_end_on_cpu = timer()
    total_train_time_model = helpers.print_train_time(start=train_time_start_on_cpu, 
                                            end=train_time_end_on_cpu,
                                            device=str(next(model.parameters()).device))

    model_results = helpers.eval_model(model, eval_dataloader, loss_fn, helpers.accuracy_fn, device)
    print(model_results)

if __name__ == "__main__":
    main()