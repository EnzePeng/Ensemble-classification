import os
import shutil
import random
from pathlib import Path

def sample_subfolders(src_root, dst_root, sample_ratio=0.2, seed=42, move=False):
    """
    From each subfolder in src_root, sample a portion of images and copy/move to dst_root preserving subfolder names.

    Args:
        src_root (str or Path): source root folder with subfolders (classes)
        dst_root (str or Path): destination root folder
        sample_ratio (float): fraction of images to sample in each subfolder
        seed (int): random seed for reproducibility
        move (bool): if True, move files; else copy files
    """
    random.seed(seed)
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    # Iterate all subfolders
    for subfolder in src_root.iterdir():
        print("Now process subfolder:", subfolder)
        if not subfolder.is_dir():
            continue

        images = [p for p in subfolder.iterdir() if p.suffix.lower() in valid_exts]
        sample_count = int(len(images) * sample_ratio)
        sampled = random.sample(images, sample_count) if sample_count > 0 else []

        dst_subfolder = dst_root / subfolder.name
        dst_subfolder.mkdir(parents=True, exist_ok=True)

        for img_path in sampled:
            if move:
                shutil.move(str(img_path), dst_subfolder / img_path.name)
            else:
                shutil.copy(str(img_path), dst_subfolder / img_path.name)

        print(f"[{subfolder.name}] Sampled {len(sampled)}/{len(images)} images to {dst_subfolder} (move={move})")

    print("\n✅ Sampling completed.")

# ====== Example usage ======

src_root = r'E:\Project\classification-ensemble\classification-ensemble\train'     # Replace with your source folder containing subfolders
dst_root = r'E:\Project\classification-ensemble\classification-ensemble\test'     # Replace with your destination folder
sample_ratio = 0.2               # Sample 30% from each subfolder
move_files = True               # False=copy, True=move

sample_subfolders(src_root, dst_root, sample_ratio, seed=666, move=move_files)

