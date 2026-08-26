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
            transforms.ToTensor()
        ])

    train_data = datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,transform=transform)
    eval_data = datasets.ImageFolder(root=eval_dir,transform=transform)

    ###3 Neural Net

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
    model_0 = BeeModelV0(
            input_shape=3*32*32, 
            hidden_units=10,
            output_shape=len(train_data.classes)
            )
    #train_and_test(
    #    model_0, 
    #    train_dataloader, 
    #    test_dataloader, 
    #    eval_dataloader, 
    #    loss_fn = nn.CrossEntropyLoss(), 
    #    optimizer = optim.SGD(params=model_0.parameters(), lr=1e-2),
    #    device=DEVICE
    #)

    ### CNN
    model_1 = BeeModelConv(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data.classes)
    )

    train_and_test(
        model_1,
        train_dataloader,
        test_dataloader,
        eval_dataloader,

        loss_fn=nn.CrossEntropyLoss(),
        optimizer = optim.SGD(params=model_1.parameters(), 
                             lr=0.1),
        device=DEVICE
    )

    import random
    random.seed(420)
    # Sample a few images from the training data to visualise predictions
    sample_k = 25
    samples = []
    labels = []
    for img, lbl in random.sample(list(train_data), k=sample_k):
        samples.append(img)
        labels.append(lbl)

    # View the first sample shape and label
    print(f"Sample image shape: {samples[0].shape}\nSample label: {labels[0]}({train_data.classes[labels[0]]})")

    # Make predictions on the sampled training images
    pred_probs = make_predictions(model=model_1, data=samples)
    pred_classes = pred_probs.argmax(dim=1)

    import matplotlib.pyplot as plt

    # Plot predictions (convert CHW tensors to HWC RGB images)
    plt.figure(figsize=(9, 9))
    nrows = 5
    ncols = 5
    for i, sample in enumerate(samples):
        plt.subplot(nrows, ncols, i+1)

        # sample is a Tensor in (C, H, W) with values in [0,1]
        if isinstance(sample, torch.Tensor):
            img = sample.permute(1, 2, 0).cpu().numpy()
        else:
            img = sample

        # Show the RGB image
        plt.imshow(img)

        # Prediction and ground-truth labels (text form)
        pred_label = train_data.classes[int(pred_classes[i].item())]
        truth_label = train_data.classes[int(labels[i])]

        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        color = "g" if pred_label == truth_label else "r"
        plt.title(title_text, fontsize=10, c=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()



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

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = DEVICE):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

if __name__ == "__main__":
    main()