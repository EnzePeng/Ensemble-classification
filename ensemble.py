import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score

from dataset import BinaryImageDataset
import torchvision.transforms as T
import torchvision
from tqdm import tqdm
import os
import shutil
import torch.nn as nn


def load_model(model_type: str, weight_path: str, device='cpu'):
    if model_type == 'resnet':
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_type == 'efficientnet':
        model = torchvision.models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_type == 'convnext':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = None  # 训练好权重直接加载，不用预训练
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    else:
        raise ValueError("Unsupported model type. Choose 'resnet', 'efficientnet', or 'convnext'.")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference_ensemble(
    resnet_path,
    effnet_path,
    convnext_path,
    test_dir,
    batch_size=32,
    device='cpu',
    error_dir='recall_error',
    threshold=0.1,
    vote_required=2  # 1：至少一个模型判为1；2：至少两个模型判为1；3：三个都判为1
):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = BinaryImageDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model1 = load_model('resnet', resnet_path, device)
    model2 = load_model('efficientnet', effnet_path, device)
    model3 = load_model('convnext', convnext_path, device)

    os.makedirs(error_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    idx_start = 0
    error_count = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            labels_np = labels.numpy()

            logits1 = model1(images)
            logits2 = model2(images)
            logits3 = model3(images)

            probs1 = torch.sigmoid(logits1).cpu().numpy()
            probs2 = torch.sigmoid(logits2).cpu().numpy()
            probs3 = torch.sigmoid(logits3).cpu().numpy()

            preds1 = (probs1 > threshold).astype(int)
            preds2 = (probs2 > threshold).astype(int)
            preds3 = (probs3 > threshold).astype(int)

            # 三个模型投票
            final_preds = ((preds1 + preds2 + preds3) >= vote_required).astype(int)

            all_preds.extend(final_preds.flatten())
            all_labels.extend(labels_np.flatten())

            for i in range(images.size(0)):
                if labels_np[i] == 1 and final_preds[i] == 0:
                    src_path = test_dataset.image_paths[idx_start + i]
                    dst_path = os.path.join(error_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dst_path)
                    error_count += 1
                    print(f"Recall error - image: {src_path}, "
                          f"probs: resnet={probs1[i][0]:.4f}, effnet={probs2[i][0]:.4f}, convnext={probs3[i][0]:.4f}")

            idx_start += images.size(0)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n[Ensemble Voting Result]")
    print(f"Threshold: {threshold:.2f}, Vote Required: {vote_required}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall:   {recall:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"False Positive Rate (误检率): {fpr:.4f}")
    print(f"Recall error samples saved to: {error_dir}, count: {error_count}")

    # print(f"\n[Ensemble Voting Result]")
    # print(f"Threshold: {threshold:.2f}, Vote Required: {vote_required}")
    # print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"Test Recall:   {recall:.4f}")
    # print(f"Recall error samples saved to: {error_dir}, count: {error_count}")


if __name__ == "__main__":
    resnet_path = r'resNet.pth'
    effnet_path = r'effecientNet.pth'
    convnext_path = r'convNeXt2.pth'  # 你训练好的ConvNeXt模型权重路径
    test_dir = r'E:\Project\classification-ensemble\classification-ensemble\test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference_ensemble(
        resnet_path,
        effnet_path,
        convnext_path,
        test_dir,
        device=device,
        threshold=0.5,
        vote_required=1
    )
