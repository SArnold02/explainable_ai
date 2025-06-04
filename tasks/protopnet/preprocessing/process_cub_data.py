import os
import pandas as pd
from PIL import Image


def crop_and_save_cub_images(dataset_root, output_root):
    images_file = os.path.join(dataset_root, 'CUB_200_2011', 'images.txt')
    labels_file = os.path.join(dataset_root, 'CUB_200_2011', 'image_class_labels.txt')
    split_file = os.path.join(dataset_root, 'CUB_200_2011', 'train_test_split.txt')
    bbox_file = os.path.join(dataset_root, 'CUB_200_2011', 'bounding_boxes.txt')
    image_dir = os.path.join(dataset_root, 'CUB_200_2011', 'images')

    images = pd.read_csv(images_file, sep=' ', names=['img_id', 'filepath'])
    labels = pd.read_csv(labels_file, sep=' ', names=['img_id', 'target'])
    splits = pd.read_csv(split_file, sep=' ', names=['img_id', 'is_training_img'])
    bboxes = pd.read_csv(bbox_file, sep=' ', names=['img_id', 'x', 'y', 'width', 'height'])

    df = images.merge(labels, on='img_id') \
        .merge(splits, on='img_id') \
        .merge(bboxes, on='img_id')

    train_out = os.path.join(output_root, 'train_cropped')
    test_out = os.path.join(output_root, 'test_cropped')

    for _, row in df.iterrows():
        src_path = os.path.join(image_dir, row.filepath)
        out_dir = train_out if row.is_training_img == 1 else test_out
        out_path = os.path.join(out_dir, row.filepath)

        try:
            img = Image.open(src_path).convert('RGB')
            x, y, w, h = row.x, row.y, row.width, row.height
            cropped = img.crop((x, y, x + w, y + h))

            # Save cropped image
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cropped.save(out_path)
        except Exception as e:
            print(f"Failed processing {src_path}: {e}")

print(os.getcwd())
crop_and_save_cub_images(
    dataset_root="../../data",
    output_root="./datasets/cub200_cropped"
)