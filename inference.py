# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import cv2
import PIL
from PIL import Image

import skimage.measure
import skimage as ski

from skimage import data, color, feature
from skimage.transform import rescale, resize
from skimage import img_as_float
from skimage import exposure
from skimage import filters

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(jboysen_mri_and_alzheimers_path)

longitudinal_data = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_longitudinal.csv")
display(longitudinal_data)

cross_sectional = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_cross-sectional.csv")
display(cross_sectional)
from google.colab import sheets
sheet = sheets.InteractiveSheet(df=longitudinal_data)
import glob
image_data = glob.glob(ninadaithal_imagesoasis_path + "/*/*/*.jpg")

first_image = image_data[0]
img = Image.open(first_image)
plt.imshow(img)
plt.show()
# shows the first image from the MRIs


print(image_data[0])
print(os.path.basename(image_data[0]))
print(os.path.basename(image_data[0]).split('_'))
print("_".join(os.path.basename(image_data[0]).split('_')[:3]))

images = []
for path in image_data:
    filename = os.path.basename(path)
    subject_id = "_".join(filename.split('_')[:3])
    images.append((subject_id, path))
# creates a dictionary of the new MRI IDS

image_df = pd.DataFrame(images, columns=['ID', 'ImageFiles'])
# dictionary --> dataframe

matched_ids = np.isin(cross_sectional['ID'],image_df['ID'])
print("Total matches: ", matched_ids.sum())
image_df.groupby("ID").agg({"ImageFiles":"nunique"}).sort_values("ImageFiles")
for i, row in image_df.loc[image_df.ID=='OAS1_0294_MR1'].iterrows():
    first_image = row.ImageFiles
    img = Image.open(first_image)
    plt.title(first_image)
    plt.imshow(img)
    #plt.show()
from tqdm import tqdm

entropy_calculations = []

for subject_id, path in tqdm(images, desc="Calculating Entropy"):
    img = Image.open(path).resize((64, 64)).convert('L')
    img_np = np.array(img)
    entropy = skimage.measure.shannon_entropy(img_np)
    entropy_calculations.append((subject_id, path, entropy))


entropy_df = pd.DataFrame(entropy_calculations, columns=['ID', 'ImageFiles', 'Entropy'])

highest_entropy_df = entropy_df.sort_values(['ID', 'Entropy'], ascending=[True, False]).groupby('ID').head(10).reset_index(drop=True)
entropy_df.groupby('ID').agg({'Entropy':'var'}).hist()
plt.title("Entropy Variance per Subject")
plt.xlabel("Entropy Variance")
plt.ylabel("Number of Subjects")
plt.show()
# also show histogram of # of slices per person to show why we need top 10 based on entropy -- bc done on previous paper + reduce # of data per person (add as a slide)

num_of_slices = entropy_df.groupby('ID').size()

# plot histogram of slice counts
plt.hist(num_of_slices, bins=4, color='skyblue', edgecolor='black')
plt.title("Number of MRI Slices per Subject")
plt.xlabel("Number of Slices")
plt.ylabel("Number of Subjects")
plt.grid(True)
plt.show()
display(entropy_df.loc[entropy_df.ID=="OAS1_0028_MR1"].sort_values(by="Entropy",ascending=False))
highest_entropy_df.loc[highest_entropy_df.ID=="OAS1_0028_MR1"]
first_image = highest_entropy_df.loc[0, "ImageFiles"]
img = Image.open(first_image)
img_np = np.array(img)
plt.imshow(img_np)
plt.title("Sample Original Image")
plt.show()

top_left = img_np[:200, :200]
plt.imshow(top_left)
plt.title("Top Left Corner")
plt.show()


print(img_np.shape)






image_rescaled = rescale(img_np, 0.25, anti_aliasing=False)
plt.imshow(image_rescaled,cmap='gray')
plt.title("Rescaled Image - No Anti-Aliasing")
plt.show()


image_resized = resize(
    img_np, (img_np.shape[0] // 4, img_np.shape[1] // 4), anti_aliasing=True
)
plt.imshow(image_resized)
plt.title("Resized Image - Anti-Aliasing")
plt.show()




blobs = feature.blob_dog(img_np[:,:,0], max_sigma=100, threshold=0.07)
print(blobs)

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(img_np, cmap='gray')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r * np.sqrt(2), color='blue', linewidth=2, fill=False)
    ax.add_patch(c)

ax.set_title("Image with Blobs")
plt.axis('off')
plt.show()







def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

gamma_corrected = exposure.adjust_gamma(img_np, 2)

# Logarithmic
logarithmic_corrected = exposure.adjust_log(img_np, 1)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=object)
axes[0, 0] = plt.subplot(2, 3, 1)
axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
axes[1, 0] = plt.subplot(2, 3, 4)
axes[1, 1] = plt.subplot(2, 3, 5)
axes[1, 2] = plt.subplot(2, 3, 6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
ax_img.set_title('Gamma correction')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
ax_img.set_title('Logarithmic correction')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()







def plot_img_hist_cdf(image, axes_img, axes_hist, axes_cdf, bins=256):
    """Plot image, histogram, and CDF in separate axes."""
    image = img_as_float(image)

    # Image
    axes_img.imshow(image, cmap=plt.cm.gray)
    axes_img.set_axis_off()

    # Histogram
    axes_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    axes_hist.set_title('Frequency of Pixel Intensities')
    axes_hist.set_xlabel('Pixel Intensity')
    axes_hist.set_ylabel('Number of Pixels')
    axes_hist.set_xlim(0, 1)
    axes_hist.set_ylim(0, 35000)

    # CDF
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    axes_cdf.plot(bins, img_cdf, 'r')
    axes_cdf.set_title('Cumulative Fraction')
    axes_cdf.set_xlabel('Pixel Intensity')
    axes_cdf.set_ylabel('Fraction of Total Intensity')
    axes_cdf.set_xlim(0, 1)
    axes_cdf.set_ylim(0,1.1)

# Create images
gamma_corrected = exposure.adjust_gamma(img_np, 2)
logarithmic_corrected = exposure.adjust_log(img_np, 1)

# Setup figure
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

# Plot original
plot_img_hist_cdf(img_np, axes[0, 0], axes[0, 1], axes[0, 2])
axes[0, 0].set_title('Original image')

# Plot gamma corrected
plot_img_hist_cdf(gamma_corrected, axes[1, 0], axes[1, 1], axes[1, 2])
axes[1, 0].set_title('Gamma Corrected')

# Plot log corrected
plot_img_hist_cdf(logarithmic_corrected, axes[2, 0], axes[2, 1], axes[2, 2])
axes[2, 0].set_title('Logarithmic Corrected')

plt.tight_layout()
plt.show()
edges = filters.sobel(img_np)

plt.imshow(edges, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')
plt.show()
import torch.nn as nn
import torchvision.models as models

# Model 1: Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*64*64, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*64*64)
        return self.fc1(x)

# Model 2: ResNet18 (pretrained)
def get_resnet(num_classes=3):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Model 3: Vision Transformer (ViT)
from torchvision.models.vision_transformer import vit_b_16
def get_vit(num_classes=3):
    model = vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        history["train_loss"].append(running_loss/len(train_loader))
        history["val_loss"].append(val_loss/len(val_loader))
        history["val_acc"].append(acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, Val Acc: {acc:.2f}%")

    return model, history

#!/usr/bin/env python3
"""
train_backbones.py

Train the same dataset with different backbones (resnet18, resnet50, vit_b_16),
compare validation accuracy and save the best models.

Usage examples:
    python train_backbones.py --data_dir /path/to/data --backbone resnet18 --epochs 10 --batch_size 16
    python train_backbones.py --data_dir /path/to/data --backbone vit_b_16 --epochs 15 --batch_size 8
"""

import argparse
import os
import copy
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

# ----------------------------
# Utilities
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train multiple backbones on same dataset")
    parser.add_argument("--data_dir", required=True, help="Root directory with 'train' and 'val' subfolders")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18","resnet50","vit_b_16"],
                        help="Backbone model to use")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detected if not set)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--input_size", type=int, default=224, help="Input size (resize/crop)")
    return parser.parse_args()

def create_dataloaders(data_dir, input_size, batch_size, num_workers):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("Expecting 'train' and 'val' subdirectories in data_dir")

    # Basic transforms; adjust if MRI/CT require different normalization
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, transform=data_transforms["val"])
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=="train"),
                      num_workers=num_workers, pin_memory=True)
        for x in ["train","val"]
    }
    class_names = image_datasets["train"].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train","val"]}

    return dataloaders, class_names, dataset_sizes

# ----------------------------
# Model factory
# ----------------------------
def get_model(backbone_name, num_classes, pretrained=True, device="cpu"):
    backbone_name = backbone_name.lower()
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported resnet: " + backbone_name)
        # Replace final fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone_name == "vit_b_16":
        # torchvision ViT (available in recent torchvision)
        model = models.vit_b_16(pretrained=pretrained)
        # ViT heads name may be head or classifier depending on version
        if hasattr(model, "heads"):
            in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
            # replace head
            model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
        elif hasattr(model, "heads") is False and hasattr(model, "head") :
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            # fallback to classifier attribute
            if hasattr(model, "classifier"):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            else:
                raise RuntimeError("Unknown ViT structure for this torchvision version.")
    else:
        raise ValueError("Unsupported backbone: " + backbone_name)

    model = model.to(device)
    return model

# ----------------------------
# Train / Evaluate
# ----------------------------
def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler=None, num_epochs=10, save_path=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*20)

        # Each epoch has a training and validation phase
        for phase in ["train","val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    # some torchvision models return dicts or named tuples; ensure tensor
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase=="train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # record
            if phase=="train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
                if scheduler:
                    scheduler.step()
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                # deep copy the model if best
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if save_path:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch+1,
                            "best_acc": best_acc
                        }, save_path)
                        print(f"Saved best model to {save_path}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)
    return model, history, best_acc

# ----------------------------
# Main
# ----------------------------
def main():
    args = get_args()
    device = torch.device(args.device)

    dataloaders, class_names, dataset_sizes = create_dataloaders(args.data_dir, args.input_size, args.batch_size, args.num_workers)
    num_classes = args.num_classes if args.num_classes else len(class_names)
    print(f"Detected classes: {class_names} (num: {num_classes})")
    print(f"Dataset sizes: {dataset_sizes}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = args.backbone
    save_path = os.path.join(args.save_dir, f"{model_name}_best.pth")

    model = get_model(model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)

    # Freeze layers optionally (uncomment if you want to freeze the feature extractor)
    # for name, param in model.named_parameters():
    #     if "fc" not in name and "head" not in name and "classifier" not in name and "heads" not in name:
    #         param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    # Only parameters that require grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = None

    trained_model, history, best_acc = train_model(model, dataloaders, dataset_sizes, device,
                                                  criterion, optimizer, scheduler,
                                                  num_epochs=args.epochs, save_path=save_path)

    print(f"\nFinished training {model_name}. Best Val Acc: {best_acc:.4f}")
    # Save final model (not necessarily best)
    final_save = os.path.join(args.save_dir, f"{model_name}_final.pth")
    torch.save(trained_model.state_dict(), final_save)
    print(f"Saved final model to {final_save}")

    # Print a small summary
    print("\nSummary (last epoch):")
    for k, v in history.items():
        print(f"{k}: {v[-1]:.4f}" if len(v) else f"{k}: -")

if __name__ == "__main__":
    main()

import os
print(os.listdir())
print(merged_df.columns)
import torch.nn as nn
import torchvision.models as models
import torch
import pandas as pd # Import pandas

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if merged_df exists and has the 'Group' column
if 'merged_df' not in globals() or 'Group' not in merged_df.columns:
    print("Error: merged_df is not defined or does not have the 'Group' column. Please run the data loading and merging cell first.")
else:
    # Determine the number of classes
    num_classes = len(merged_df['Group'].unique())

    # Model 1: Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * (128 // 2) * (128 // 2), num_classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(-1, 16 * (128 // 2) * (128 // 2))
            return self.fc1(x)

    # Model 2: ResNet18 (pretrained)
    def get_resnet(num_classes):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Model 3: Vision Transformer (ViT)
    from torchvision.models.vision_transformer import vit_b_16
    def get_vit(num_classes):
        model = vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    # Initialize models and move to device
    cnn_model = SimpleCNN(num_classes=num_classes).to(device)
    resnet_model = get_resnet(num_classes=num_classes).to(device)
    vit_model = get_vit(num_classes=num_classes).to(device)

    models_dict = {
        "CNN": cnn_model,
        "ResNet18": resnet_model,
        "ViT": vit_model
    }

    print("Models initialized and moved to device.")
# Reload the data paths
import kagglehub
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')

# Load the dataframes
cross_sectional = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_cross-sectional.csv")
longitudinal_data = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_longitudinal.csv")

# Regenerate highest_entropy_df (as it was created in a previous cell)
from tqdm import tqdm
from PIL import Image
import numpy as np
import skimage.measure
import glob
import os

image_data = glob.glob(ninadaithal_imagesoasis_path + "/*/*/*.jpg")

images = []
for path in image_data:
    filename = os.path.basename(path)
    subject_id = "_".join(filename.split('_')[:3])
    images.append((subject_id, path))

entropy_calculations = []

for subject_id, path in tqdm(images, desc="Calculating Entropy"):
    img = Image.open(path).resize((64, 64)).convert('L')
    img_np = np.array(img)
    entropy = skimage.measure.shannon_entropy(img_np)
    entropy_calculations.append((subject_id, path, entropy))

entropy_df = pd.DataFrame(entropy_calculations, columns=['ID', 'ImageFiles', 'Entropy'])
highest_entropy_df = entropy_df.sort_values(['ID', 'Entropy'], ascending=[True, False]).groupby('ID').head(10).reset_index(drop=True)


# Merge cross_sectional and highest_entropy_df on 'ID'
merged_df = pd.merge(cross_sectional, highest_entropy_df, on='ID', how='inner')

# Handle missing CDR values by dropping rows
merged_df = merged_df.dropna(subset=['CDR'])

# Create 'Group' column based on CDR
# Assuming CDR > 0 indicates Demented, CDR = 0 indicates Non-demented
merged_df['Group'] = merged_df['CDR'].apply(lambda x: 'Demented' if x > 0 else 'Non-demented')


# Display the first few rows and unique groups
display(merged_df.head())
display(merged_df['Group'].unique())
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = self.dataframe['Group'].astype('category').cat.codes.values
        self.image_paths = self.dataframe['ImageFiles'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Reduced size for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
])

# Create the dataset
dataset = MRIDataset(merged_df, transform=transform)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from tqdm import tqdm

# Reload the data paths from the initial data source import cell
import kagglehub
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')


# ==========================
# 1. Load Tabular Data
# ==========================
cross_df = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_cross-sectional.csv")
# Assuming highest_entropy_df is already loaded from a previous step or file
# If not, it needs to be loaded or created. Based on the context, it was created and saved.
entropy_df = pd.read_csv("entropy_df.csv") # Corrected path
longitudinal_df = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_longitudinal.csv") # Corrected path


# Example: use "CDR" column (Clinical Dementia Rating) as label
cross_df = cross_df.dropna(subset=["CDR"])
cross_df["label"] = (cross_df["CDR"] > 0).astype(int)  # 0=Non-demented, 1=Demented
# ==========================
# 2. Custom Dataset
# ==========================
class MRIDataset(Dataset):
    def __init__(self, image_paths, df, transform=None):
        self.image_paths = image_paths
        self.df = df
        self.transform = transform

        # map subject/session ID
        self.labels = []
        for path in image_paths:
            subject_id = os.path.basename(path).split("_")[0]  # adjust as per filename pattern
            label_row = df[df["ID"] == subject_id]
            if not label_row.empty:
                self.labels.append(int(label_row["label"].values[0]))
            else:
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

# Define a working MRIDataset class within this cell
class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        # Ensure 'Group' is correctly mapped to numerical labels
        self.labels = self.dataframe['Group'].astype('category').cat.codes.values
        self.image_paths = self.dataframe['ImageFiles'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB') # Convert to RGB for 3 channels
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

# ==========================
# 3. Transformations
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Use ImageNet mean/std for 3-channel RGB images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset using the pre-processed merged_df
# merged_df is available from cell `e43bvmU0hFNH` and contains 'ImageFiles' and 'Group' columns.
dataset = MRIDataset(merged_df, transform=transform)

# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Ensure there are samples to split
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the data loading and merging steps.")
if train_size == 0 or test_size == 0:
    # If the dataset is too small for an 80/20 split, adjust sizes or raise an error
    if len(dataset) < 2:
        raise ValueError("Dataset has fewer than 2 samples, cannot perform train/test split.")
    # If only one split would be 0, assign at least 1 sample to each
    if train_size == 0: train_size = 1; test_size = len(dataset) - 1
    if test_size == 0: test_size = 1; train_size = len(dataset) - 1

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 4. Models
# ==========================
# CNN (Custom)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*56*56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

# Vision Transformer (ViT from torchvision)
vit = models.vit_b_16(pretrained=True)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

models_dict = {
    "CNN": SimpleCNN().to(device),
    "ResNet18": resnet18.to(device),
    "ViT": vit.to(device)
}

# ==========================
# 5. Training Function
# ==========================
def train_model(model, train_loader, test_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc, test_acc = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc.append(correct / total)

        # evaluate
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = correct / total
        test_acc.append(acc)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {acc:.4f}")

    return train_acc, test_acc, all_labels, all_preds


# ==========================
# 6. Train & Evaluate
# ==========================
results = {}
for name, model in models_dict.items():
    print(f"\nTraining {name}...")
    train_acc, test_acc, labels, preds = train_model(model, train_loader, test_loader, epochs=3)
    results[name] = {"train_acc": train_acc, "test_acc": test_acc, "labels": labels, "preds": preds}


import os
import argparse
import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# --- Removing CLI and Data loading parts, using existing notebook state ---

# Assuming `train_loader`, `test_loader`, `device` are defined in previous cells
# Assuming `merged_df` is defined in previous cells and contains 'Group' column
# Assuming `dataset` is defined in previous cells (from 0-DxbQoSiuEx)

# Determine num_classes and class_names from the existing dataset
num_classes = dataset.dataframe['Group'].nunique()
class_names = dataset.dataframe['Group'].astype('category').cat.categories.tolist()

# ---------------------------
# Simple CNN (aligned with AHSh_9izijtn's SimpleCNN, ensuring correct output size for 224x224 input)
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Input size 224x224 -> after pool1 (112x112) -> after pool2 (56x56)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Model factory for ResNet (from original cell, adapted for num_classes)
# ---------------------------
def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# ---------------------------
# Model factory for ViT (from original cell, adapted for num_classes)
# ---------------------------
def get_vit(num_classes, pretrained=True, device="cpu"):
    model = models.vit_b_16(pretrained=pretrained)
    # ViT heads name may be head or classifier depending on version
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        # fallback to classifier attribute if neither 'heads' nor 'head' is present
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown ViT structure for this torchvision version.")
    return model.to(device)


# Initialize models using the determined num_classes and device
models_to_evaluate = {
    "CNN": SimpleCNN(num_classes=num_classes).to(device),
    "ResNet18": get_resnet("resnet18", num_classes=num_classes, pretrained=True, device=device),
    "ViT": get_vit(num_classes=num_classes, pretrained=True, device=device)
}

# ---------------------------
# Train helper (adapted from original train_model in EIa_zpzORgWE)
# Uses train_loader and test_loader directly for training and validation
# ---------------------------
def train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=5, save_path=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # validate
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Ensure labels are of type torch.long for validation as well
                if labels.dtype != torch.long:
                    labels = labels.to(torch.long)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(model.state_dict(), save_path)
    model.load_state_dict(best_model_wts)
    return model, best_acc

# ---------------------------
# Evaluation / metrics (from original cell)
# ---------------------------
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long for evaluation
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_logits.append(probs.cpu().numpy())  # shape (N, C)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.vstack(all_logits)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    # also compute macro averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    # confusion
    cm = confusion_matrix(labels, preds)
    # confidence: mean probability of the predicted class overall and per-class
    predicted_probs = probs[np.arange(len(preds)), preds]
    mean_confidence = float(predicted_probs.mean())
    # per-class confidence: average predicted probability when model predicted that class
    per_class_confidence = {}
    for cls_idx, name in enumerate(class_names):
        mask = preds == cls_idx
        if mask.sum() > 0:
            per_class_confidence[name] = float(predicted_probs[mask].mean())
        else:
            per_class_confidence[name] = None

    # per-class metrics packaging
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "sensitivity": float(recall[i]),  # sensitivity = recall
            "f1": float(f1[i]),
            "support": int(support[i]),
            "avg_confidence_when_predicted": per_class_confidence[name]
        }

    summary = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "mean_confidence": mean_confidence,
        "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON compatibility
        "per_class": per_class_metrics
    }
    return summary, labels, preds, probs

# ---------------------------
# Pretty print + csv export (from original cell)
# ---------------------------
def summarize_and_save(name, summary, save_dir):
    rows = []
    cm = np.array(summary["confusion_matrix"]) # Convert back to numpy if needed for printing
    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro Precision: {summary['precision_macro']:.4f}, Macro Recall: {summary['recall_macro']:.4f}, Macro F1: {summary['f1_macro']:.4f}")
    print(f"Mean confidence (predicted class probability): {summary['mean_confidence']:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-class metrics:")
    for cls, m in summary["per_class"].items():
        # Handle None for confidence if no predictions were made for a class
        conf_str = f"{m['avg_confidence_when_predicted']:.4f}" if m['avg_confidence_when_predicted'] is not None else "N/A"
        print(f" {cls}: prec={m['precision']:.4f} rec={m['recall']:.4f} sens={m['sensitivity']:.4f} f1={m['f1']:.4f} support={m['support']} conf={conf_str}")
        rows.append({
            "model": name,
            "class": cls,
            "precision": m['precision'],
            "recall": m['recall'],
            "sensitivity": m['sensitivity'],
            "f1": m['f1'],
            "support": m['support'],
            "avg_confidence_when_predicted": m['avg_confidence_when_predicted'],
            "accuracy_overall": summary['accuracy'],
            "mean_confidence_overall": summary['mean_confidence']
        })
    # Export CSV per model
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class metrics to {csv_path}")
    return df

# ---------------------------
# Main Execution Block (adapted for Colab notebook)
# ---------------------------

save_dir = "./model_evaluation_results"
os.makedirs(save_dir, exist_ok=True)

all_results_df = pd.DataFrame()

for name, model in models_to_evaluate.items():
    print(f"\nTraining and Evaluating {name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default learning rate

    model_save_path = os.path.join(save_dir, f"{name}_best.pth")
    trained_model, best_val_acc = train_model(
        model, train_loader, test_loader, device, criterion, optimizer, num_epochs=3, save_path=model_save_path
    )
    print(f"Finished training {name}. Best Val Acc: {best_val_acc:.4f}")

    # Evaluate on the test set
    summary, _, _, _ = evaluate_model(trained_model, test_loader, device, class_names)
    model_df = summarize_and_save(name, summary, save_dir)
    all_results_df = pd.concat([all_results_df, model_df], ignore_index=True)

# Compare overall metrics in a single CSV
summary_rows = []
for name in models_to_evaluate.keys():
    # Aggregate overall metrics from per-class entries if necessary, or ensure 'accuracy_overall' is consistent
    # The summarize_and_save function already adds overall metrics to each row, so we can pick from there.
    model_overall_metrics = all_results_df[all_results_df['model'] == name].drop_duplicates(subset=['model'])
    if not model_overall_metrics.empty:
        row = model_overall_metrics.iloc[0]
        summary_rows.append(
            {
                "model": name,
                "accuracy": row["accuracy_overall"],
                "precision_macro": row["precision"], # These should ideally be the true macro values from summary
                "recall_macro": row["recall"],
                "f1_macro": row["f1"],
                "mean_confidence": row["mean_confidence_overall"]
            }
        )
    else:
        print(f"Warning: No summary found for model {name} in all_results_df.")

comp_df = pd.DataFrame(summary_rows)
comp_csv = os.path.join(save_dir, "models_comparison_summary.csv")
comp_df.to_csv(comp_csv, index=False)
print("\nSaved comparison summary to", comp_csv)
print("\nComparison table:")
print(comp_df.to_string(index=False))

import numpy as np

def evaluate_model(model_name, y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall    = recall_score(y_true, y_pred, average='macro')
    f1        = f1_score(y_true, y_pred, average='macro')

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def compare_models(scan_type, modelA_results, modelB_results):
    print("\n==============================")
    print(" ALZHEIMER DETECTION SUMMARY ")
    print("==============================\n")

    print(f"Input Scan Type: {scan_type.upper()}\n")

    print("Models Compared:")
    print(f" 1. {modelA_results['model']}")
    print(f" 2. {modelB_results['model']}\n")

    print("Model Performances:")
    print(f"{modelA_results['model']} → Accuracy: {modelA_results['accuracy']:.4f}, F1-Score: {modelA_results['f1_score']:.4f}")
    print(f"{modelB_results['model']} → Accuracy: {modelB_results['accuracy']:.4f}, F1-Score: {modelB_results['f1_score']:.4f}\n")

    # Select best model
    best_model = modelA_results if modelA_results["accuracy"] > modelB_results["accuracy"] else modelB_results

    print("Best Model for This Scan Type:")
    print(f"✔ {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")
    print("\n==============================\n")

    return best_model



# ------------------ SAMPLE USAGE ------------------

# Example predictions (Replace with your model predictions)
y_true = np.array([0,1,1,0,1,0])
y_pred_cnn = np.array([0,1,1,0,0,0])
y_pred_resnet = np.array([0,1,1,1,1,0])

# Evaluate both models
cnn_results = evaluate_model("CNN Model", y_true, y_pred_cnn)
resnet_results = evaluate_model("ResNet50 Model", y_true, y_pred_resnet)

# Compare models for MRI scan input
best_model = compare_models("MRI Scan", cnn_results, resnet_results)

# Determine best model (with tie handling)
accuracies = {
    cnn_results["model"]: cnn_results["accuracy"],
    resnet_results["model"]: resnet_results["accuracy"]
}

max_acc = max(accuracies.values())

# Find all models with highest accuracy
best_models = [name for name, acc in accuracies.items() if acc == max_acc]

if len(best_models) == 1:
    print("Best Model for This Scan Type:")
    print(f"✔ {best_models[0]} (Accuracy = {max_acc:.4f})")
else:
    print("Best Model for This Scan Type (Tie):")
    for m in best_models:
        print(f"✔ {m} (Accuracy = {max_acc:.4f})")
import random

# Determine the number of images to sample
num_samples = min(8000, len(image_data))

# Randomly sample image paths
sampled_image_data = random.sample(image_data, num_samples)

print(f"Original number of image paths: {len(image_data)}")
print(f"Sampled number of image paths: {len(sampled_image_data)}")
images = []
for path in sampled_image_data:
    filename = os.path.basename(path)
    subject_id = "_".join(filename.split('_')[:3])
    images.append((subject_id, path))

# creates a dictionary of the new MRI IDS

entropy_calculations = []

for subject_id, path in tqdm(images, desc="Calculating Entropy for Sampled Images"):
    img = Image.open(path).resize((64, 64)).convert('L')
    img_np = np.array(img)
    entropy = skimage.measure.shannon_entropy(img_np)
    entropy_calculations.append((subject_id, path, entropy))

entropy_df = pd.DataFrame(entropy_calculations, columns=['ID', 'ImageFiles', 'Entropy'])

highest_entropy_df = entropy_df.sort_values(['ID', 'Entropy'], ascending=[True, False]).groupby('ID').head(10).reset_index(drop=True)

print("Entropy calculation complete for sampled images.")
display(highest_entropy_df.head())
import kagglehub
import pandas as pd
import numpy as np
from PIL import Image
import skimage.measure
import glob
import os
from tqdm import tqdm

# Reload the data paths
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')

# Load the dataframes
cross_sectional = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_cross-sectional.csv")
longitudinal_data = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_longitudinal.csv")

# highest_entropy_df was already created from sampled images in the previous cell (fb14030c)
# We will use that existing highest_entropy_df for merging.

# Merge cross_sectional and highest_entropy_df on 'ID'
merged_df = pd.merge(cross_sectional, highest_entropy_df, on='ID', how='inner')

# Handle missing CDR values by dropping rows
merged_df = merged_df.dropna(subset=['CDR'])

# Create 'Group' column based on CDR
# Assuming CDR > 0 indicates Demented, CDR = 0 indicates Non-demented
merged_df['Group'] = merged_df['CDR'].apply(lambda x: 'Demented' if x > 0 else 'Non-demented')


# Display the first few rows and unique groups
display(merged_df.head())
display(merged_df['Group'].unique())
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = self.dataframe['Group'].astype('category').cat.codes.values
        self.image_paths = self.dataframe['ImageFiles'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Reduced size for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
])

# Create the dataset
dataset = MRIDataset(merged_df, transform=transform)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import torch.nn as nn
import torchvision.models as models
import torch # Ensure torch is imported if not globally available

# ==========================
# 4. Models
# ==========================
# CNN (Custom)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*56*56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

# Vision Transformer (ViT from torchvision)
vit = models.vit_b_16(pretrained=True)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

models_dict = {
    "CNN": SimpleCNN().to(device),
    "ResNet18": resnet18.to(device),
    "ViT": vit.to(device)
}
import torch.nn as nn
import torchvision.models as models
import torch # Ensure torch is imported if not globally available

# ==========================
# 4. Models
# ==========================
# CNN (Custom)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*56*56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

# Vision Transformer (ViT from torchvision)
vit = models.vit_b_16(pretrained=True)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

models_dict = {
    "CNN": SimpleCNN().to(device),
    "ResNet18": resnet18.to(device),
    "ViT": vit.to(device)
}
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = self.dataframe['Group'].astype('category').cat.codes.values
        self.image_paths = self.dataframe['ImageFiles'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Changed size to 224x224 for ViT compatibility
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
])

# Create the dataset
dataset = MRIDataset(merged_df, transform=transform)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import os
import argparse
import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# --- Removing CLI and Data loading parts, using existing notebook state ---

# Assuming `train_loader`, `test_loader`, `device` are defined in previous cells
# Assuming `merged_df` is defined in previous cells and contains 'Group' column
# Assuming `dataset` is defined in previous cells (from 0-DxbQoSiuEx)

# Determine num_classes and class_names from the existing dataset
num_classes = dataset.dataframe['Group'].nunique()
class_names = dataset.dataframe['Group'].astype('category').cat.categories.tolist()

# ---------------------------
# Simple CNN (aligned with AHSh_9izijtn's SimpleCNN, ensuring correct output size for 224x224 input)
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Input size 224x224 -> after pool1 (112x112) -> after pool2 (56x56)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Model factory for ResNet (from original cell, adapted for num_classes)
# ---------------------------
def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# ---------------------------
# Model factory for ViT (from original cell, adapted for num_classes)
# ---------------------------
def get_vit(num_classes, pretrained=True, device="cpu"):
    model = models.vit_b_16(pretrained=pretrained)
    # ViT heads name may be head or classifier depending on version
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        # fallback to classifier attribute if neither 'heads' nor 'head' is present
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown ViT structure for this torchvision version.")
    return model.to(device)


# Initialize models using the determined num_classes and device
models_to_evaluate = {
    "CNN": SimpleCNN(num_classes=num_classes).to(device),
    "ResNet18": get_resnet("resnet18", num_classes=num_classes, pretrained=True, device=device),
    "ViT": get_vit(num_classes=num_classes, pretrained=True, device=device)
}

# ---------------------------
# Train helper (adapted from original train_model in EIa_zpzORgWE)
# Uses train_loader and test_loader directly for training and validation
# ---------------------------
def train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=5, save_path=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # validate
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Ensure labels are of type torch.long for validation as well
                if labels.dtype != torch.long:
                    labels = labels.to(torch.long)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(model.state_dict(), save_path)
    model.load_state_dict(best_model_wts)
    return model, best_acc

# ---------------------------
# Evaluation / metrics (from original cell)
# ---------------------------
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long for evaluation
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_logits.append(probs.cpu().numpy())  # shape (N, C)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.vstack(all_logits)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    # also compute macro averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    # confusion
    cm = confusion_matrix(labels, preds)
    # confidence: mean probability of the predicted class overall and per-class
    predicted_probs = probs[np.arange(len(preds)), preds]
    mean_confidence = float(predicted_probs.mean())
    # per-class confidence: average predicted probability when model predicted that class
    per_class_confidence = {}
    for cls_idx, name in enumerate(class_names):
        mask = preds == cls_idx
        if mask.sum() > 0:
            per_class_confidence[name] = float(predicted_probs[mask].mean())
        else:
            per_class_confidence[name] = None

    # per-class metrics packaging
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "sensitivity": float(recall[i]),  # sensitivity = recall
            "f1": float(f1[i]),
            "support": int(support[i]),
            "avg_confidence_when_predicted": per_class_confidence[name]
        }

    summary = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "mean_confidence": mean_confidence,
        "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON compatibility
        "per_class": per_class_metrics
    }
    return summary, labels, preds, probs

# ---------------------------
# Pretty print + csv export (from original cell)
# ---------------------------
def summarize_and_save(name, summary, save_dir):
    rows = []
    cm = np.array(summary["confusion_matrix"]) # Convert back to numpy if needed for printing
    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro Precision: {summary['precision_macro']:.4f}, Macro Recall: {summary['recall_macro']:.4f}, Macro F1: {summary['f1_macro']:.4f}")
    print(f"Mean confidence (predicted class probability): {summary['mean_confidence']:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-class metrics:")
    for cls, m in summary["per_class"].items():
        # Handle None for confidence if no predictions were made for a class
        conf_str = f"{m['avg_confidence_when_predicted']:.4f}" if m['avg_confidence_when_predicted'] is not None else "N/A"
        print(f" {cls}: prec={m['precision']:.4f} rec={m['recall']:.4f} sens={m['sensitivity']:.4f} f1={m['f1']:.4f} support={m['support']} conf={conf_str}")
        rows.append({
            "model": name,
            "class": cls,
            "precision": m['precision'],
            "recall": m['recall'],
            "sensitivity": m['sensitivity'],
            "f1": m['f1'],
            "support": m['support'],
            "avg_confidence_when_predicted": m['avg_confidence_when_predicted'],
            "accuracy_overall": summary['accuracy'],
            "mean_confidence_overall": summary['mean_confidence']
        })
    # Export CSV per model
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class metrics to {csv_path}")
    return df

# ---------------------------
# Main Execution Block (adapted for Colab notebook)
# ---------------------------

save_dir = "./model_evaluation_results"
os.makedirs(save_dir, exist_ok=True)

all_results_df = pd.DataFrame()

for name, model in models_to_evaluate.items():
    print(f"\nTraining and Evaluating {name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default learning rate

    model_save_path = os.path.join(save_dir, f"{name}_best.pth")
    trained_model, best_val_acc = train_model(
        model, train_loader, val_loader, device, criterion, optimizer, num_epochs=3, save_path=model_save_path
    )
    print(f"Finished training {name}. Best Val Acc: {best_val_acc:.4f}")

    # Evaluate on the test set
    summary, _, _, _ = evaluate_model(trained_model, val_loader, device, class_names)
    model_df = summarize_and_save(name, summary, save_dir)
    all_results_df = pd.concat([all_results_df, model_df], ignore_index=True)

# Compare overall metrics in a single CSV
summary_rows = []
for name in models_to_evaluate.keys():
    # Aggregate overall metrics from per-class entries if necessary, or ensure 'accuracy_overall' is consistent
    # The summarize_and_save function already adds overall metrics to each row, so we can pick from there.
    model_overall_metrics = all_results_df[all_results_df['model'] == name].drop_duplicates(subset=['model'])
    if not model_overall_metrics.empty:
        row = model_overall_metrics.iloc[0]
        summary_rows.append(
            {
                "model": name,
                "accuracy": row["accuracy_overall"],
                "precision_macro": row["precision"], # These should ideally be the true macro values from summary
                "recall_macro": row["recall"],
                "f1_macro": row["f1"],
                "mean_confidence": row["mean_confidence_overall"]
            }
        )
    else:
        print(f"Warning: No summary found for model {name} in all_results_df.")

comp_df = pd.DataFrame(summary_rows)
comp_csv = os.path.join(save_dir, "models_comparison_summary.csv")
comp_df.to_csv(comp_csv, index=False)
print("\nSaved comparison summary to", comp_csv)
print("\nComparison table:")
print(comp_df.to_string(index=False))

import random
import glob
import os
import kagglehub

# Ensure data paths are defined (as they might not be in a new session)
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')

# Define image_data as it was in cell VMHbl4K8wI8r
image_data = glob.glob(ninadaithal_imagesoasis_path + "/*/*/*.jpg")

# Determine the number of images to sample, limiting to 2500
num_samples = min(2500, len(image_data))

# Randomly sample image paths
sampled_image_data = random.sample(image_data, num_samples)

print(f"Original number of image paths: {len(image_data)}")
print(f"Sampled number of image paths: {len(sampled_image_data)}")

from tqdm import tqdm
from PIL import Image
import numpy as np
import skimage.measure
import pandas as pd

images = []
for path in sampled_image_data:
    filename = os.path.basename(path)
    subject_id = "_".join(filename.split('_')[:3])
    images.append((subject_id, path))

# creates a dictionary of the new MRI IDS

entropy_calculations = []

for subject_id, path in tqdm(images, desc="Calculating Entropy for Sampled Images"):
    img = Image.open(path).resize((64, 64)).convert('L')
    img_np = np.array(img)
    entropy = skimage.measure.shannon_entropy(img_np)
    entropy_calculations.append((subject_id, path, entropy))

entropy_df = pd.DataFrame(entropy_calculations, columns=['ID', 'ImageFiles', 'Entropy'])

highest_entropy_df = entropy_df.sort_values(['ID', 'Entropy'], ascending=[True, False]).groupby('ID').head(10).reset_index(drop=True)

print("Entropy calculation complete for sampled images.")
display(highest_entropy_df.head())
import kagglehub
import pandas as pd
import numpy as np
from PIL import Image
import skimage.measure
import glob
import os
from tqdm import tqdm

# Reload the data paths
jboysen_mri_and_alzheimers_path = kagglehub.dataset_download('jboysen/mri-and-alzheimers')
ninadaithal_imagesoasis_path = kagglehub.dataset_download('ninadaithal/imagesoasis')

# Load the dataframes
cross_sectional = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_cross-sectional.csv")
longitudinal_data = pd.read_csv(jboysen_mri_and_alzheimers_path + "/oasis_longitudinal.csv")

# highest_entropy_df was already created from sampled images in the previous cell (fb14030c)
# We will use that existing highest_entropy_df for merging.

# Merge cross_sectional and highest_entropy_df on 'ID'
merged_df = pd.merge(cross_sectional, highest_entropy_df, on='ID', how='inner')

# Handle missing CDR values by dropping rows
merged_df = merged_df.dropna(subset=['CDR'])

# Create 'Group' column based on CDR
# Assuming CDR > 0 indicates Demented, CDR = 0 indicates Non-demented
merged_df['Group'] = merged_df['CDR'].apply(lambda x: 'Demented' if x > 0 else 'Non-demented')


# Display the first few rows and unique groups
display(merged_df.head())
display(merged_df['Group'].unique())
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = self.dataframe['Group'].astype('category').cat.codes.values
        self.image_paths = self.dataframe['ImageFiles'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Changed size to 224x224 for ViT compatibility
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
])

# Create the dataset
dataset = MRIDataset(merged_df, transform=transform)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import torch.nn as nn
import torchvision.models as models
import torch # Ensure torch is imported if not globally available

# ==========================
# 4. Models
# ==========================
# CNN (Custom)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # For 224x224 input:
        # 224/2 = 112, 112/2 = 56
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*56*56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

# Vision Transformer (ViT from torchvision)
vit = models.vit_b_16(pretrained=True)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)

models_dict = {
    "CNN": SimpleCNN().to(device),
    "ResNet18": resnet18.to(device),
    "ViT": vit.to(device)
}
import os
import argparse
import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# --- Removing CLI and Data loading parts, using existing notebook state ---

# Assuming `train_loader`, `test_loader`, `device` are defined in previous cells
# Assuming `merged_df` is defined in previous cells and contains 'Group' column
# Assuming `dataset` is defined in previous cells (from 0-DxbQoSiuEx)

# Determine num_classes and class_names from the existing dataset
num_classes = dataset.dataframe['Group'].nunique()
class_names = dataset.dataframe['Group'].astype('category').cat.categories.tolist()

# ---------------------------
# Simple CNN (aligned with AHSh_9izijtn's SimpleCNN, ensuring correct output size for 224x224 input)
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Input size 224x224 -> after pool1 (112x112) -> after pool2 (56x56)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Model factory for ResNet (from original cell, adapted for num_classes)
# ---------------------------
def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# ---------------------------
# Model factory for ViT (from original cell, adapted for num_classes)
# ---------------------------
def get_vit(num_classes, pretrained=True, device="cpu"):
    model = models.vit_b_16(pretrained=pretrained)
    # ViT heads name may be head or classifier depending on version
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        # fallback to classifier attribute if neither 'heads' nor 'head' is present
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown ViT structure for this torchvision version.")
    return model.to(device)


# Initialize models using the determined num_classes and device
models_to_evaluate = {
    "CNN": SimpleCNN(num_classes=num_classes).to(device),
    "ResNet18": get_resnet("resnet18", num_classes=num_classes, pretrained=True, device=device),
    "ViT": get_vit(num_classes=num_classes, pretrained=True, device=device)
}

# ---------------------------
# Train helper (adapted from original train_model in EIa_zpzORgWE)
# Uses train_loader and test_loader directly for training and validation
# ---------------------------
def train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=5, save_path=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # validate
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Ensure labels are of type torch.long for validation as well
                if labels.dtype != torch.long:
                    labels = labels.to(torch.long)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(model.state_dict(), save_path)
    model.load_state_dict(best_model_wts)
    return model, best_acc

# ---------------------------
# Evaluation / metrics (from original cell)
# ---------------------------
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Ensure labels are of type torch.long for evaluation
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_logits.append(probs.cpu().numpy())  # shape (N, C)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.vstack(all_logits)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    # also compute macro averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    # confusion
    cm = confusion_matrix(labels, preds)
    # confidence: mean probability of the predicted class overall and per-class
    predicted_probs = probs[np.arange(len(preds)), preds]
    mean_confidence = float(predicted_probs.mean())
    # per-class confidence: average predicted probability when model predicted that class
    per_class_confidence = {}
    for cls_idx, name in enumerate(class_names):
        mask = preds == cls_idx
        if mask.sum() > 0:
            per_class_confidence[name] = float(predicted_probs[mask].mean())
        else:
            per_class_confidence[name] = None

    # per-class metrics packaging
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "sensitivity": float(recall[i]),  # sensitivity = recall
            "f1": float(f1[i]),
            "support": int(support[i]),
            "avg_confidence_when_predicted": per_class_confidence[name]
        }

    summary = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "mean_confidence": mean_confidence,
        "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON compatibility
        "per_class": per_class_metrics
    }
    return summary, labels, preds, probs

# ---------------------------
# Pretty print + csv export (from original cell)
# ---------------------------
def summarize_and_save(name, summary, save_dir):
    rows = []
    cm = np.array(summary["confusion_matrix"]) # Convert back to numpy if needed for printing
    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro Precision: {summary['precision_macro']:.4f}, Macro Recall: {summary['recall_macro']:.4f}, Macro F1: {summary['f1_macro']:.4f}")
    print(f"Mean confidence (predicted class probability): {summary['mean_confidence']:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-class metrics:")
    for cls, m in summary["per_class"].items():
        # Handle None for confidence if no predictions were made for a class
        conf_str = f"{m['avg_confidence_when_predicted']:.4f}" if m['avg_confidence_when_predicted'] is not None else "N/A"
        print(f" {cls}: prec={m['precision']:.4f} rec={m['recall']:.4f} sens={m['sensitivity']:.4f} f1={m['f1']:.4f} support={m['support']} conf={conf_str}")
        rows.append({
            "model": name,
            "class": cls,
            "precision": m['precision'],
            "recall": m['recall'],
            "sensitivity": m['sensitivity'],
            "f1": m['f1'],
            "support": m['support'],
            "avg_confidence_when_predicted": m['avg_confidence_when_predicted'],
            "accuracy_overall": summary['accuracy'],
            "mean_confidence_overall": summary['mean_confidence']
        })
    # Export CSV per model
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class metrics to {csv_path}")
    return df

# ---------------------------
# Main Execution Block (adapted for Colab notebook)
# ---------------------------

save_dir = "./model_evaluation_results"
os.makedirs(save_dir, exist_ok=True)

all_results_df = pd.DataFrame()

for name, model in models_to_evaluate.items():
    print(f"\nTraining and Evaluating {name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default learning rate

    model_save_path = os.path.join(save_dir, f"{name}_best.pth")
    trained_model, best_val_acc = train_model(
        model, train_loader, val_loader, device, criterion, optimizer, num_epochs=3, save_path=model_save_path
    )
    print(f"Finished training {name}. Best Val Acc: {best_val_acc:.4f}")

    # Evaluate on the test set
    summary, _, _, _ = evaluate_model(trained_model, val_loader, device, class_names)
    model_df = summarize_and_save(name, summary, save_dir)
    all_results_df = pd.concat([all_results_df, model_df], ignore_index=True)

# Compare overall metrics in a single CSV
summary_rows = []
for name in models_to_evaluate.keys():
    # Aggregate overall metrics from per-class entries if necessary, or ensure 'accuracy_overall' is consistent
    # The summarize_and_save function already adds overall metrics to each row, so we can pick from there.
    model_overall_metrics = all_results_df[all_results_df['model'] == name].drop_duplicates(subset=['model'])
    if not model_overall_metrics.empty:
        row = model_overall_metrics.iloc[0]
        summary_rows.append(
            {
                "model": name,
                "accuracy": row["accuracy_overall"],
                "precision_macro": row["precision"], # These should ideally be the true macro values from summary
                "recall_macro": row["recall"],
                "f1_macro": row["f1"],
                "mean_confidence": row["mean_confidence_overall"]
            }
        )
    else:
        print(f"Warning: No summary found for model {name} in all_results_df.")

comp_df = pd.DataFrame(summary_rows)
comp_csv = os.path.join(save_dir, "models_comparison_summary.csv")
comp_df.to_csv(comp_csv, index=False)
print("\nSaved comparison summary to", comp_csv)
print("\nComparison table:")
print(comp_df.to_string(index=False))
best_model_name = comp_df.loc[comp_df['accuracy'].idxmax()]['model']
best_accuracy = comp_df['accuracy'].max()

summary_text = """
### Model Comparison Summary Review

Based on the `comp_df` results:

"""

for index, row in comp_df.iterrows():
    model_name = row['model']
    accuracy = row['accuracy']
    precision = row['precision_macro']
    recall = row['recall_macro']
    f1 = row['f1_macro']
    mean_confidence = row['mean_confidence']

    summary_text += f"*   **{model_name}** achieved an accuracy of {accuracy:.3f}, macro precision of {precision:.3f}, macro recall of {recall:.3f}, and macro F1-score of {f1:.3f}. Its mean confidence was {mean_confidence:.3f}.\n"

summary_text += f"\n**Conclusion:** For this specific dataset and experimental setup, **{best_model_name}** is the best-performing model for Alzheimer's detection with an accuracy of {best_accuracy:.3f}.\n"

# Special note for ViT if it performed poorly (as observed in previous run)
if 'ViT' in comp_df['model'].values:
    vit_row = comp_df[comp_df['model'] == 'ViT'].iloc[0]
    if vit_row['f1_macro'] == 0.0:
        summary_text += "\n_Note: The ViT model performed poorly, likely classifying all instances into a single class ('Non-demented' as observed in the confusion matrix). This could be due to the relatively small dataset size after sampling, which might not be sufficient for a complex model like ViT to learn effectively, or an imbalance in the dataset. Further investigation would be needed to understand this behavior._\n"

print(summary_text)
best_model_row = comp_df.loc[comp_df['accuracy'].idxmax()]
best_model_name = best_model_row['model']
best_accuracy = best_model_row['accuracy']

print(f"Best performing model: {best_model_name}")
print(f"Highest accuracy: {best_accuracy:.4f}")
import torch
import os

# Identify the best model from the comp_df (already done in previous cell)
# best_model_name and best_accuracy are available from the previous step

print(f"The best performing model is: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Get the model instance for the best performing model
# The models_to_evaluate dictionary contains the trained models,
# and the train_model function ensures that the best weights are loaded back
# into the model object before it's returned.
best_model = models_to_evaluate[best_model_name]

# Define the path to save the best model's state dictionary
save_dir = "./model_deployment"
os.makedirs(save_dir, exist_ok=True)
best_model_weights_path = os.path.join(save_dir, f"{best_model_name}_best_weights.pth")

# Save the state dictionary of the best model
torch.save(best_model.state_dict(), best_model_weights_path)

print(f"State dictionary of the best model ({best_model_name}) saved to: {best_model_weights_path}")
pip install streamlit
import streamlit as st

# Application title
st.title('Alzheimer’s Disease Prediction App')

# Application description
st.markdown("""
This application predicts Alzheimer's Disease progression based on MRI scans.
It uses a pre-trained ResNet18 model to classify individuals into 'Demented' or 'Non-demented' categories.
""")

print("Streamlit app.py content generated.")
print("""
### Instructions to create `app.py`:

1.  **Create a new file**: In your Colab environment, go to `File > New File`.
2.  **Rename the file**: Rename the new file to `app.py`.
3.  **Copy and paste the content**: Copy the following Python code and paste it into the `app.py` file:

```python
import streamlit as st

# Application title
st.title('Alzheimer’s Disease Prediction App')

# Application description
st.markdown('''
This application predicts Alzheimer's Disease progression based on MRI scans.
It uses a pre-trained ResNet18 model to classify individuals into 'Demented' or 'Non-demented' categories.
''')

# Placeholder for file uploader and prediction logic
st.subheader("Upload an MRI Scan")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # TODO: Add model loading and prediction logic here
    # Example: prediction = predict_function(uploaded_file)
    # st.write(f"Prediction: {prediction}")

```

4.  **Save the file**: Save the `app.py` file.

Once `app.py` is created, the Streamlit application can be run.
""")
print("""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# --- Model Definition (as in the notebook) ---

def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# --- Preprocessing Transformation (as in the notebook) ---

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Consistent with training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standards
])

# --- Model Loading ---

# Determine device to load model (CPU is safer for deployment)
device = torch.device("cpu")
num_classes = 2 # 'Demented' and 'Non-demented'
class_names = ['Demented', 'Non-demented']

# Instantiate the model architecture
best_model_name = "ResNet18" # From previous analysis
best_model = get_resnet(best_model_name.lower(), num_classes=num_classes, pretrained=False, device=device)

# Path to the saved weights (adjust if your Streamlit app is in a different directory)
# Assuming model_deployment folder is in the same directory as app.py
model_weights_path = f"./model_deployment/{best_model_name}_best_weights.pth"

try:
    best_model.load_state_dict(torch.load(model_weights_path, map_location=device))
    best_model.eval() # Set model to evaluation mode
    st.success(f"Successfully loaded {best_model_name} model weights.")
except FileNotFoundError:
    st.error(f"Error: Model weights not found at {model_weights_path}. Please ensure the file exists.")
    st.stop() # Stop the app if model weights can't be loaded
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Streamlit Application UI ---

st.title('Alzheimer’s Disease Prediction App')

st.markdown('''
This application predicts Alzheimer's Disease progression based on MRI scans.
It uses a pre-trained ResNet18 model to classify individuals into 'Demented' or 'Non-demented' categories.
''')

st.subheader("Upload an MRI Scan")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Read image from uploaded file
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert('RGB')

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension

        # Move tensor to device
        input_tensor = input_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            output = best_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class = class_names[predicted_class_idx.item()]
            confidence = probabilities[0, predicted_class_idx.item()].item()

        st.success(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


""")
print("""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import pandas as pd

# --- Model Definitions (as in the notebook) ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # For 224x224 input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def get_vit(num_classes, pretrained=True, device="cpu"):
    model = models.vit_b_16(pretrained=pretrained)
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown ViT structure for this torchvision version.")
    return model.to(device)

# --- Preprocessing Transformation (consistent with training) ---

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model Loading (using st.cache_resource) ---

# Determine device to load model (CPU is safer for deployment)
device = torch.device("cpu")
num_classes = 2 # 'Demented' and 'Non-demented'
class_names = ['Demented', 'Non-demented']

@st.cache_resource
def load_and_prepare_model(model_type, model_name, num_classes, device):
    model = None
    if model_type == "CNN":
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == "ResNet18":
        model = get_resnet(model_name.lower(), num_classes=num_classes, pretrained=False, device=device)
    elif model_type == "ViT":
        model = get_vit(num_classes=num_classes, pretrained=False, device=device)
    else:
        st.error(f"Unknown model type: {model_type}")
        return None

    # Path to the saved weights (adjust if your Streamlit app is in a different directory)
    model_weights_path = f"./model_evaluation_results/{model_name}_best.pth"
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Error: Model weights for {model_name} not found at {model_weights_path}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        st.stop()
    return None

# Load all three models
cnn_model = load_and_prepare_model("CNN", "CNN", num_classes, device)
resnet_model = load_and_prepare_model("ResNet18", "ResNet18", num_classes, device)
vit_model = load_and_prepare_model("ViT", "ViT", num_classes, device)

# --- Prediction Function ---

def predict_image(model, image, transform, device, class_names):
    if model is None: # Handle cases where model loading failed
        return "N/A", 0.0
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim and move to device

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_class_idx.item()]
        confidence = probabilities[0, predicted_class_idx.item()].item()
    return predicted_class, confidence

# --- Streamlit Application UI ---

st.title('Alzheimer’s Disease Prediction App')

st.markdown('''
This application predicts Alzheimer's Disease progression based on MRI scans.
It utilizes three different deep learning models (CNN, ResNet18, ViT) to classify
individuals into 'Demented' or 'Non-demented' categories and provides a comparative summary.
''')

st.subheader("Upload an MRI Scan")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption='Uploaded MRI Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions with each model
    results = []
    models_to_test = {"CNN": cnn_model, "ResNet18": resnet_model, "ViT": vit_model}

    for model_name, model_instance in models_to_test.items():
        if model_instance:
            predicted_class, confidence = predict_image(model_instance, image, transform, device, class_names)
            results.append({"Model": model_name, "Prediction": predicted_class, "Confidence": f"{confidence:.2f}"})
            st.write(f"**{model_name} Prediction:** {predicted_class} (Confidence: {confidence:.2f})")
        else:
            results.append({"Model": model_name, "Prediction": "Loading Failed", "Confidence": "N/A"})
            st.warning(f"Model {model_name} could not be loaded for prediction.")

    # Display comparative summary
    st.subheader("Comparative Summary")
    results_df = pd.DataFrame(results)
    st.table(results_df)

else:
    st.info("Please upload an image to get predictions.")

""")
print_content_of_cell(cell_id='06cdd6fd')
print("""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import pandas as pd

# --- Model Definitions (as in the notebook) ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # For 224x224 input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_resnet(backbone, num_classes, pretrained=True, device="cpu"):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("unsupported resnet")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def get_vit(num_classes, pretrained=True, device="cpu"):
    model = models.vit_b_16(pretrained=pretrained)
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features if hasattr(model.heads, "head") else model.heads[0].in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown ViT structure for this torchvision version.")
    return model.to(device)

# --- Preprocessing Transformation (consistent with training) ---

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model Loading (using st.cache_resource) ---

# Determine device to load model (CPU is safer for deployment)
device = torch.device("cpu")
num_classes = 2 # 'Demented' and 'Non-demented'
class_names = ['Demented', 'Non-demented']

@st.cache_resource
def load_and_prepare_model(model_type, model_name, num_classes, device):
    model = None
    if model_type == "CNN":
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == "ResNet18":
        model = get_resnet(model_name.lower(), num_classes=num_classes, pretrained=False, device=device)
    elif model_type == "ViT":
        model = get_vit(num_classes=num_classes, pretrained=False, device=device)
    else:
        st.error(f"Unknown model type: {model_type}")
        return None

    # Path to the saved weights (adjust if your Streamlit app is in a different directory)
    model_weights_path = f"./model_evaluation_results/{model_name}_best.pth"
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Error: Model weights for {model_name} not found at {model_weights_path}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        st.stop()
    return None

# Load all three models
cnn_model = load_and_prepare_model("CNN", "CNN", num_classes, device)
resnet_model = load_and_prepare_model("ResNet18", "ResNet18", num_classes, device)
vit_model = load_and_prepare_model("ViT", "ViT", num_classes, device)

# --- Prediction Function ---

def predict_image(model, image, transform, device, class_names):
    if model is None: # Handle cases where model loading failed
        return "N/A", 0.0
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim and move to device

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_class_idx.item()]
        confidence = probabilities[0, predicted_class_idx.item()].item()
    return predicted_class, confidence

# --- Streamlit Application UI ---

st.title('Alzheimer’s Disease Prediction App')

st.markdown('''
This application predicts Alzheimer's Disease progression based on MRI scans.
It utilizes three different deep learning models (CNN, ResNet18, ViT) to classify
individuals into 'Demented' or 'Non-demented' categories and provides a comparative summary.
''')

st.subheader("Upload an MRI Scan")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption='Uploaded MRI Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions with each model
    results = []
    models_to_test = {"CNN": cnn_model, "ResNet18": resnet_model, "ViT": vit_model}

    for model_name, model_instance in models_to_test.items():
        if model_instance:
            predicted_class, confidence = predict_image(model_instance, image, transform, device, class_names)
            results.append({"Model": model_name, "Prediction": predicted_class, "Confidence": f"{confidence:.2f}"})
            st.write(f"**{model_name} Prediction:** {predicted_class} (Confidence: {confidence:.2f})")
        else:
            results.append({"Model": model_name, "Prediction": "Loading Failed", "Confidence": "N/A"})
            st.warning(f"Model {model_name} could not be loaded for prediction.")

    # Display comparative summary
    st.subheader("Comparative Summary")
    results_df = pd.DataFrame(results)
    st.table(results_df)

else:
    st.info("Please upload an image to get predictions.")

""")
