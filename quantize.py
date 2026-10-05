import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

from baseModels import BeeModelConv as Net

from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor import TorchExecutor

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


def eval_quantized_model(graph, data_loader, device=DEVICE):
    executor = TorchExecutor(graph=graph, device=device)
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for X, y in data_loader:
            outputs = executor(X.to(device))
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = (y_true == y_pred).mean()
    print(f"Quantized model accuracy: {accuracy:.4f}")

    class_names = data_loader.dataset.classes
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    label_width = max(len(name) for name in class_names)
    column_width = max(label_width, len(str(cm.max())), 5)
    
    print("Quantized confusion matrix (rows: actual, columns: predicted):")
    print(f"{'':>{label_width}}  " + "  ".join(f"{name:>{column_width}}" for name in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:>{label_width}}  " + "  ".join(f"{count:>{column_width}}" for count in row))

    return accuracy, cm




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
    eval_dataset = datasets.ImageFolder(root=eval_dir, transform=transform)
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
    )

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
    #model.load_state_dict(torch.load("../model_loss0.18_acc93.66_hl48.pth", map_location=DEVICE))
    model.load_state_dict(torch.load("./model_BeeModelConv.pth", map_location=DEVICE))
    # ! labels are determined by alphabetical order of the folder names in the dataset.
    # background = 0, bee_fast = 1, bee_slow = 2
    model.eval()

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=testDataLoader,
        calib_steps=512,
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
    eval_quantized_model(quant_ppq_graph, eval_dataloader)