import os
import torch
import pandas as pd
import tarfile
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

DEFAULT_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def transform_partial(
    keypoints: list[tuple[int, int, int, int]],
    orig_size: tuple[int, int] = (500, 357),
    resize_shorter: int = 224,
) -> list[tuple[int, int, int, int]]:
    orig_h, orig_w = orig_size

    scale_x = resize_shorter / orig_w
    scale_y = resize_shorter / orig_h

    transformed: list[tuple[int, int, int, int]] = []
    for part_id, x, y, vis in keypoints:
        # 3) resize
        x_r = x * scale_x
        y_r = y * scale_y
        transformed.append((part_id, x_r, y_r, vis))

    return transformed

class Cub2011(Dataset):
    def __init__(self, root = "./data", train = True, transform = None, download = False):
        # Hardcoded variables
        self.base_folder = 'CUB_200_2011/images'
        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.filename = 'CUB_200_2011.tgz'
        self.tgz_md5 = '97eceeb196236b17998738112f37df78'
        
        # Data variables
        self.root = os.path.expanduser(root)
        self.transform = transform if transform is not None else (
            DEFAULT_TRAIN_TRANSFORM 
            if train
            else DEFAULT_VAL_TRANSFORM
        )
        self.loader = default_loader
        self.train = train
        self.transform_partial = transform_partial

        # Download the data if neccessary
        if download:
            self._download()

        # Load the data
        self._load_metadata()

        # Check the integrity of the data
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        # Load and split the data
        images = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            sep=' ', names=['img_id', 'filepath']
        )
        labels = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target']
        )
        splits = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['img_id', 'is_training_img']
        )

        # Load part annotations: parts/part_locs.txt
        parts_file = os.path.join(self.root, 'CUB_200_2011', 'parts', 'part_locs.txt')
        self.part_dict = {}  # img_id -> list of (part_idx, x, y)
        with open(parts_file) as f:
            for line in f:
                img_id, part_id, x, y, vis = line.strip().split()
                img_id = int(img_id)
                part_id = int(part_id) - 1  # zero-based
                x = float(x)
                y = float(y)
                vis = int(vis)
                self.part_dict.setdefault(img_id, []).append((part_id, x, y, vis))

        # Merge the loaded data
        df = images.merge(labels, on='img_id').merge(splits, on='img_id')

        # Only save the data neccessary for training/evaluation
        flag = 1 if self.train else 0
        self.data = df[df.is_training_img == flag].reset_index(drop=True)

    def _check_integrity(self):
        # Check if the files are present
        print("Checking data integrity")

        if not hasattr(self, "data"):
            return False

        base = os.path.join(self.root, self.base_folder)
        for _, row in self.data.iterrows():
            path = os.path.join(base, row.filepath)
            if not os.path.isfile(path):
                print(f"Missing file: {path}")
                return False

        print("Finished data integrity checking")
        return True

    def _download(self):
        # Check if the data is already downloaded
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # Download the data
        print("Downloading data")
        download_url(self.url, self.root, self.filename, self.tgz_md5)

        archive_path = os.path.join(self.root, self.filename)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.root)

        # Clean up the downloaded archive
        try:
            os.remove(archive_path)
            print(f"Removed archive file: {archive_path}")
        except OSError as e:
            print(f"Warning: could not remove archive file {archive_path}: {e}")

        print("Finished data download")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the current batch
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        orig_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        # Get partial dictonary
        keypoints = self.part_dict.get(sample.img_id, [])
        keypoints = torch.tensor(self.transform_partial(keypoints, orig_size), dtype=torch.int)

        return img, target, keypoints