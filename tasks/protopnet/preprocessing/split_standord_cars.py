import os
import shutil
import random

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

source_dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars\train_augmented'

train_dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars\split\train'
test_dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars\split\test'

test_ratio = 0.2

class_folders = next(os.walk(source_dir))[1]

for class_name in class_folders:
    src_class_path = os.path.join(source_dir, class_name)
    train_class_path = os.path.join(train_dir, class_name)
    test_class_path = os.path.join(test_dir, class_name)

    makedir(train_class_path)
    makedir(test_class_path)

    images = os.listdir(src_class_path)
    random.shuffle(images)

    split_index = int(len(images) * (1 - test_ratio))
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_class_path, img), os.path.join(train_class_path, img))

    for img in test_images:
        shutil.copy(os.path.join(src_class_path, img), os.path.join(test_class_path, img))

    print(f"Class {class_name}: {len(train_images)} train, {len(test_images)} test")
