import Augmentor
import os
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

print("Choose dataset to augment:")
print("1. CUB-200")
print("2. Stanford Cars")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped\train_cropped'
    target_dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped\train_cropped_augmented'
elif choice == "2":
    dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars_cropped\train_cropped'
    target_dir = r'D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars_cropped\train_cropped_augmented'
else:
    print("Invalid choice. Exiting.")
    exit(1)

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):

    fd = folders[i]
    tfd = target_folders[i]
    print("Files in folder:", fd)
    print(os.listdir(fd))

    # rotation
    try:
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
    except Exception as e:
        print(f"Error during sampling in {fd}: {e}")