import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from pathlib import Path
import numpy as np

from baseModels import BeeModelConv as Net

from esp_ppq.api import espdl_quantize_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("../data")
train_dir = data_path / "train"
test_dir = data_path / "test"
eval_dir = data_path / "val"

class SimulateRGB565:
    """Match the camera's 5/6/5 colour depth. Drop if you feed RGB888."""
    def __call__(self, img):
        a = np.array(img, dtype=np.uint8)
        a[..., 0] = (a[..., 0] & 0xF8) | (a[..., 0] >> 5)
        a[..., 1] = (a[..., 1] & 0xFC) | (a[..., 1] >> 6)
        a[..., 2] = (a[..., 2] & 0xF8) | (a[..., 2] >> 5)
        return Image.fromarray(a)

class FeatureOnlyDataset(Dataset):
    def __init__(self, original_dataset):
        self.features = []
        for item in original_dataset:
            self.features.append(item[0])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_fn2(batch):
    features = torch.stack(batch)
    return features.to(DEVICE)


if __name__ == '__main__':
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, 32, 32]
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    ESPDL_MODEL_PATH = "model_BeeModelConv.espdl"

    transform = transforms.Compose([
        SimulateRGB565(),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)
    test_dataset = datasets.ImageFolder(root=test_dir,transform=transform)
    #eval_dataset = datasets.ImageFolder(root=eval_dir,transform=transform)

    image = Image.open("../data/train/bee_slow/bee_0973__x243_y81_a60_202509031.png").convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    feature_only_test_data = FeatureOnlyDataset(test_dataset)

    testDataLoader = torch.utils.data.DataLoader(
        dataset=feature_only_test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn2
    )

    model = Net(
        input_shape=3, 
        hidden_units=48,
        output_shape=len(train_dataset.classes)
    ).to(DEVICE)
    model.load_state_dict(torch.load("../model_loss0.19_acc92.98_hl48_lessblocks.pth", map_location=DEVICE))
    # ! labels are determined by alphabetical order of the folder names in the dataset.
    # background = 0, bee_fast = 1, bee_slow = 2
    model.eval()

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=testDataLoader,
        calib_steps=256,
        input_shape=[1] + INPUT_SHAPE,
        inputs=[input_tensor],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None
    )