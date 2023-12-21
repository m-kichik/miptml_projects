from tqdm import tqdm
import random

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import wandb

from dataset import TrafficDataset
from model import MultiLabelCNN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def eval(model, val_loader, threshold=0.5, criterion=None, device=torch.device("cpu")):
    if criterion is None:
        raise RuntimeError("Criterion is None")

    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = (all_preds >= threshold).astype(int)

    f1 = f1_score(all_labels, all_preds, average='micro')

    return total_loss / len(val_loader), f1


def train(
    model,
    train_loader,
    val_loader,
    exp_name,
    num_epochs=100,
    criterion=None,
    optimizer=None,
    device=torch.device("cpu"),
):
    if criterion is None:
        raise RuntimeError("Criterion is None")
    if optimizer is None:
        raise RuntimeError("Optimizer is None")

    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_f1 = eval(
            model, val_loader, threshold=0.5, criterion=criterion, device=device
        )

        if val_f1 > best_f1:
            torch.save(model.state_dict(), exp_name + "_best.pth")
            best_f1 = val_f1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), exp_name + f"_{epoch + 1}_ep.pth")

        wandb.log({
            "train loss": train_loss,
            "validation loss": val_loss,
            "F1-score": val_f1,
        })

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val loss   : {val_loss:.4f}")
        print(f"F1 score   : {val_f1:.4f}")


def main():
    num_classes = 5
    nn_size = 1
    batch_size = 50
    num_epochs = 60
    in_size = 512
    device = torch.device("cuda:0")

    size_dict = {
        1: 'N',
        2: 'S',
        3: 'M',
        4: 'L',
        5: 'X',
    }

    experiment_name = f"multilabel_cnn_{size_dict[nn_size]}_normalize_and_no-fiasko"

    wandb.login(key='a110b325e0cff24ba829171ee36ae12d92ef3931')
    wandb.init(
        project='MIPT_ML_traffic_classification',
        name=experiment_name,
        config={
            "num_classes": num_classes,
            "nn_size": size_dict[nn_size],
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "input_size": in_size,
        }
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((in_size, in_size)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_val_transform = transforms.Compose(
        [
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = TrafficDataset(
        "traffic-detection-dataset/train", transform=train_transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = TrafficDataset(
        "traffic-detection-dataset/valid", transform=test_val_transform
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ds = TrafficDataset(
        "traffic-detection-dataset/test", transform=test_val_transform
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultiLabelCNN(
        num_classes=num_classes,
        input_size=in_size,
        nn_size=nn_size,
        device=device,
    )
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(
        model,
        train_loader,
        val_loader,
        experiment_name,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    test_loss, test_f1 = eval(
        model, test_loader, threshold=0.5, criterion=criterion, device=device
        )

    print(f"Final f1: {test_f1}")


if __name__ == "__main__":
    main()
