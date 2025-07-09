import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import BinaryImageDataset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np
from sklearn.metrics import recall_score
from tqdm import tqdm
import optuna


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(device):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model.to(device)


def get_transforms():
    return T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    model.train()
    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1} Training", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    return recall_score(all_labels, all_preds)


# --------------- OPTUNA OBJECTIVE ------------------
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    model = get_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_recall = 0.0
    for epoch in range(5):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        recall = evaluate(model, test_loader, loss_fn, device)
        best_recall = max(best_recall, recall)

    return best_recall


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    train_path = r'E:\Project\classification-ensemble\classification-ensemble\train'
    test_path = r'E:\Project\classification-ensemble\classification-ensemble\test'

    device = get_device()
    transform = get_transforms()

    train_dataset = BinaryImageDataset(train_path, transform=transform)
    test_dataset = BinaryImageDataset(test_path, transform=transform)

    # ------------------ OPTUNA SEARCH ---------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)

    # -------------- RETRAIN BEST MODEL -------------------
    print("Retraining best model and saving...")

    best_params = study.best_trial.params
    batch_size = best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = get_model(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=best_params["lr"],
                                 weight_decay=best_params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_recall = 0.0
    for epoch in range(10):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        recall = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1} - Recall: {recall:.4f}")
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved new best model at epoch {epoch+1}, recall={recall:.4f}")
