import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import BinaryImageDataset
import numpy as np
from sklearn.metrics import recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# def get_model(device):
#     from torchvision.models import resnet50, ResNet50_Weights
#     weights = ResNet50_Weights.DEFAULT
#     model = resnet50(weights=weights)
#     model.fc = nn.Linear(model.fc.in_features, 1)
#     return model.to(device)


def get_model(device):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    weights = ConvNeXt_Tiny_Weights.DEFAULT  # 载入预训练权重
    model = convnext_tiny(weights=weights)
    # 替换最后分类层，二分类输出1个神经元
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model.to(device)


def get_transforms():
    return T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_samples = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)

    for images, labels in train_loader_tqdm:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        train_loader_tqdm.set_postfix(loss=total_loss / total_samples)

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    recall = recall_score(all_labels, all_preds)

    return avg_loss, accuracy, recall


def plot_metrics(train_losses, test_losses, test_accuracies, test_recalls, save_path="training_metrics.png"):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(range(1, len(test_losses)+1), test_losses, 'm-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, 'g-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(range(1, len(test_recalls)+1), test_recalls, 'r-', label='Test Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Test Recall')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training metrics plot to {save_path}")
    plt.show()


def train(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs, save_path='best_model.pth'):
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_recalls = []

    best_recall = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        train_losses.append(train_loss)

        test_loss, accuracy, recall = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}: test_loss: {test_loss:.4f}, accuracy: {accuracy:.4f}, recall: {recall:.4f}")

        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
        test_recalls.append(recall)

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: New best recall {recall:.4f}, model saved to {save_path}")

    plot_metrics(train_losses, test_losses, test_accuracies, test_recalls)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Windows 多进程支持

    device = get_device()
    print(f"Using device: {device}")

    model = get_model(device)
    transform = get_transforms()

    train_image_paths = r'E:\Project\classification-ensemble\classification-ensemble\train'
    test_image_paths = r'E:\Project\classification-ensemble\classification-ensemble\test'

    train_dataset = BinaryImageDataset(train_image_paths, transform=transform)
    test_dataset = BinaryImageDataset(test_image_paths, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs=10)
