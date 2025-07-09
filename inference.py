import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import recall_score, accuracy_score
from dataset import BinaryImageDataset
import torchvision.transforms as T
import torchvision
from tqdm import tqdm
import os
import shutil

def get_model(device):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    import torch.nn as nn
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model.to(device)

def inference(model_path, test_dir, batch_size=32, device='cpu', error_dir='recall_error'):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 加载测试集
    test_dataset = BinaryImageDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建错误保存文件夹
    os.makedirs(error_dir, exist_ok=True)

    # 模型构建
    # model = torchvision.models.efficientnet_b3(pretrained=False)
    # model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    # model = torchvision.models.resnet50(pretrained=False)
    # model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = get_model(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    model.eval()

    all_preds = []
    all_labels = []
    error_count = 0

    # 推理时需要获取对应图片路径，假设dataset有self.image_paths
    idx_start = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            labels_np = labels.numpy()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.1).astype(int)

            batch_size = images.size(0)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels_np.flatten())

            # 复制错误的样本图片到error_dir
            for i in range(batch_size):
                if labels_np[i] == 1 and preds[i] == 0:  # recall 错误，正样本被漏判
                    src_path = test_dataset.image_paths[idx_start + i]
                    dst_path = os.path.join(error_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dst_path)
                    error_count += 1
                    prob_val = probs[i]
                    if isinstance(prob_val, np.ndarray):
                        prob_val = prob_val.item()
                    print(f"Recall error - image: {src_path}, Probability(bad): {prob_val:.4f}")

            idx_start += batch_size

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Recall error samples saved to: {error_dir}, count: {error_count}")

if __name__ == "__main__":
    model_path = r'best_model.pth'
    test_dir = r'E:\Project\classification-ensemble\classification-ensemble\test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference(model_path, test_dir, device=device)
