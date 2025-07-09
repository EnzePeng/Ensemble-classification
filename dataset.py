import os
from PIL import Image
from torch.utils.data import Dataset
# import torchvision.transforms as T

class BinaryImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): root directory, should include 'good/' and 'bad/'
            transform (callable, optional): preprocessing
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        class_map = {"good": 0, "bad": 1}  #label

        for class_name, label in class_map.items():
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    fpath = os.path.join(class_dir, fname)
                    self.image_paths.append(fpath)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: failed to load image {path}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

        if self.transform:
            image = self.transform(image)
        return image, label

