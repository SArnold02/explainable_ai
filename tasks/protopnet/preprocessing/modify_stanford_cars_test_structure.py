import os
import scipy.io
import shutil
from tqdm import tqdm

source_dir = r"D:\Facultate\Auto\explainable_ai\data\stanford_cars\cars_test"
annos = scipy.io.loadmat(r"D:\Facultate\Auto\explainable_ai\data\stanford_cars\cars_test_annos_withlabels.mat")["annotations"]
meta = scipy.io.loadmat(r"D:\Facultate\Auto\explainable_ai\data\stanford_cars\devkit\cars_meta.mat")["class_names"]
meta = meta[0]
target_dir = r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars\test_augmented"

os.makedirs(target_dir, exist_ok=True)

for anno in tqdm(annos[0], desc="Organizing test set by class"):
    class_idx = int(anno[4][0][0]) - 1
    filename = anno[5][0]
    class_name = meta[class_idx][0]

    class_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(class_dir, filename)

    shutil.copyfile(src_path, dst_path)
