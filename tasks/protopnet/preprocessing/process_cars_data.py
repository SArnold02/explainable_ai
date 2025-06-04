import os
import scipy.io
from PIL import Image
from tqdm import tqdm

def crop_and_save_stanford_cars(dataset_root, output_root):
    train_annos_path = os.path.join(dataset_root, 'devkit', 'cars_train_annos.mat')
    test_annos_path = os.path.join(dataset_root, 'cars_test_annos_withlabels.mat')
    meta_path = os.path.join(dataset_root, 'devkit', 'cars_meta.mat')

    train_images_dir = os.path.join(dataset_root, 'cars_train')
    test_images_dir = os.path.join(dataset_root, 'cars_test')

    train_output_dir = os.path.join(output_root, 'train_cropped')
    test_output_dir = os.path.join(output_root, 'test_cropped')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    meta = scipy.io.loadmat(meta_path)['class_names'][0]

    def crop_and_save(annos, image_dir, output_base_dir, split_name):
        for anno in tqdm(annos[0], desc=f'Cropping {split_name} images'):
            x1 = int(anno[0][0])
            y1 = int(anno[1][0])
            x2 = int(anno[2][0])
            y2 = int(anno[3][0])
            class_idx = int(anno[4][0][0]) - 1  # 1-indexed to 0-indexed
            filename = anno[5][0]
            class_name = meta[class_idx][0]

            src_path = os.path.join(image_dir, filename)
            out_dir = os.path.join(output_base_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)

            try:
                img = Image.open(src_path).convert("RGB")
                cropped = img.crop((x1, y1, x2, y2))
                cropped.save(out_path)
            except Exception as e:
                print(f"Failed to process {src_path}: {e}")

    # Load and process
    train_annos = scipy.io.loadmat(train_annos_path)['annotations']
    test_annos = scipy.io.loadmat(test_annos_path)['annotations']

    crop_and_save(train_annos, train_images_dir, train_output_dir, "train")
    crop_and_save(test_annos, test_images_dir, test_output_dir, "test")

crop_and_save_stanford_cars(
    dataset_root=r"D:\Facultate\Auto\explainable_ai\data\stanford_cars",
    output_root='../datasets/stanford_cars_cropped'
)
