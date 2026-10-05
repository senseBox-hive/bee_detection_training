import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm.auto import tqdm #progressbar
from timeit import default_timer as timer
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

import os
from pathlib import Path
import random

import helpers
from baseModels import BeeModelConv

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
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.5,1), contrast=(1,1), saturation=(1,1), hue=(0,0)),
            transforms.ToTensor()
        ])

    train_data = datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,transform=transform)
    eval_data = datasets.ImageFolder(root=eval_dir,transform=transform)
    # Turn train and test Datasets into DataLoaders
    NUM_WORKERS = os.cpu_count()-1
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=16, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=32,
                                num_workers=NUM_WORKERS,
                                shuffle=False)

    eval_dataloader = DataLoader(dataset=eval_data, 
                                batch_size=32,
                                num_workers=NUM_WORKERS, 
                                shuffle=False)

    ### CNN
    model = BeeModelConv(
        input_shape=3,
        hidden_units=48,
        output_shape=len(train_data.classes)
    )

    train_acc_array = []
    test_acc_array = []

    train_and_test(
        model,
        train_dataloader,
        test_dataloader,
        eval_dataloader,

        epochs=100,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer = optim.SGD(params=model.parameters(), 
                             lr=0.001,
                             momentum=0.9),
        device=DEVICE
    )
    
    eval(model, eval_dataloader)

    #helpers.predict_and_plot(model, train_data, DEVICE)
    #plt.plot(train_acc_array,test_acc_array)


def train_and_test(
    model, 
    train_dataloader, 
    test_dataloader, 
    eval_dataloader, 
    loss_fn, 
    optimizer, 
    epochs=10, 
    seed=42, 
    device: torch.device = DEVICE, 
    train_acc_array = None,
    test_acc_array = None
    ):
    # Set the seed and start the timer
    model.to(device)
    torch.manual_seed(seed)
    train_time_start_on_cpu = timer()

    # training and testing loop
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n-----")

        train_loss, train_acc = helpers.train_step(
            data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=helpers.accuracy_fn,
            device=device
        )
        test_loss, test_acc = helpers.test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=helpers.accuracy_fn,
            device=device
        )
        if train_acc_array:
            train_acc_array(train_acc)
        if test_acc_array:
            test_acc_array.append(test_acc)

    # Calculate training time      
    train_time_end_on_cpu = timer()
    total_train_time_model = helpers.print_train_time(start=train_time_start_on_cpu, 
                                            end=train_time_end_on_cpu,
                                            device=str(next(model.parameters()).device))

    model_results = helpers.eval_model(model, eval_dataloader, loss_fn, helpers.accuracy_fn, device)
    print(model_results)

    torch.save(model.state_dict(), f"model_{model.__class__.__name__}.pth")

def eval(
    model, 
    eval_dataloader,
):
    model.eval()
    #use eval data to evaluate the model
    #get y_true and y_pred
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in eval_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_true.append(y.cpu().numpy())
            y_pred.append(model(X).argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # get accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # get confusion matrix
    class_names = eval_dataloader.dataset.classes
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    label_width = max(len(name) for name in class_names)
    count_width = max(len(str(cm.max())), 5)
    
    print("Confusion Matrix (rows: actual, columns: predicted):")
    print(f"{'':>{label_width}}  " + "  ".join(f"{name:>{count_width}}" for name in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:>{label_width}}  " + "  ".join(f"{count:>{count_width}}" for count in row))


if __name__ == "__main__":
    main()