from __future__ import annotations
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tasks.protopnet.preprocessing.preprocess_protopnet import mean, std, preprocess_input_function
from tasks.protopnet.log import create_logger
from tasks.protopnet.helpers import makedir
from tasks.protopnet import protopnet as ppnet
import tasks.protopnet.train_and_test as tnt
import tasks.protopnet.save as save
import tasks.protopnet.push as push
import matplotlib.pyplot as plt

############################### CONSTANTS
img_size: int = 224
train_batch_size: int = 256
test_batch_size: int = 256
train_push_batch_size: int = 75
prototype_shape: tuple[int, int, int, int] = (2000, 128, 1, 1)
num_classes: int = 200
add_on_layers_type: str = "regular"
prototype_activation_function: str = "log"
joint_lr_step_size = 5
num_train_epochs = 8
num_warm_epochs = 2
push_start = 2
push_epochs = [i for i in range(num_train_epochs) if i % 3 == 0]
cub200_cropped_dataset_root = Path(r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped")
cars_cropped_dataset_root = Path(r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars_cropped")
joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}
last_layer_optimizer_lr = 1e-4
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}
#########################################

def run_protopnet(chosen_dataset):
    if chosen_dataset == "protopnet_cub200_cropped":
        run_protopnet_on_dataset(cub200_cropped_dataset_root, chosen_dataset)
    elif chosen_dataset == "protopnet_cars_cropped":
        run_protopnet_on_dataset(cars_cropped_dataset_root, chosen_dataset)

def run_protopnet_on_dataset(dataset_root, chosen_dataset) -> None:
    """Train & evaluate ProtoPNet on the cropped, augmented CUB‑200‑2011 set or on stanford cars dataset."""
    accuracies = []
    print("Using cropped and augmented " + chosen_dataset + " dataset.")

    train_augmented_dir = dataset_root / "train_cropped_augmented"
    train_cropped_dir = dataset_root / "train_cropped"
    test_cropped_dir = dataset_root / "test_cropped"


    model_dir = Path("./saved_models") / chosen_dataset / "resnet34" / "experiment_run"
    makedir(model_dir)
    log, logclose = create_logger(log_filename=model_dir / "train.log")

    # Datasets & loaders
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = datasets.ImageFolder(
        str(train_augmented_dir),
        transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    train_push_dataset = datasets.ImageFolder(
        str(train_cropped_dir),
        transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        ),
    )
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset,
        batch_size=train_push_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    test_dataset = datasets.ImageFolder(
        str(test_cropped_dir),
        transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    log(f"training set size: {len(train_loader.dataset)}")
    log(f"push set size: {len(train_push_loader.dataset)}")
    log(f"test set size: {len(test_loader.dataset)}")
    log(f"batch size: {train_batch_size}")

    # Build ProtoPNet
    protopnet = ppnet.construct_ProtoPNet(
        pretrained=True,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )
    protopnet = protopnet.cuda()

    class_specific = True

    joint_optimizer = torch.optim.Adam(
        [
            {
                "params": protopnet.features.parameters(),
                "lr": joint_optimizer_lrs["features"],
                "weight_decay": 1e-3,
            },
            {
                "params": protopnet.add_on_layers.parameters(),
                "lr": joint_optimizer_lrs["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": protopnet.prototype_vectors,
                "lr": joint_optimizer_lrs["prototype_vectors"],
            },
        ]
    )
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=joint_lr_step_size, gamma=0.1
    )

    warm_optimizer = torch.optim.Adam(
        [
            {
                "params": protopnet.add_on_layers.parameters(),
                "lr": warm_optimizer_lrs["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": protopnet.prototype_vectors,
                "lr": warm_optimizer_lrs["prototype_vectors"],
            },
        ]
    )

    log("start training")

    for epoch in range(num_train_epochs):
        log(f"epoch:\t{epoch}")

        if epoch < num_warm_epochs:
            tnt.warm_only(model=protopnet, log=log)
            _ = tnt.train(
                model=protopnet,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                class_specific=class_specific,
                coefs=coefs,
                log=log,
            )
        else:
            tnt.joint(model=protopnet, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(
                model=protopnet,
                dataloader=train_loader,
                optimizer=joint_optimizer,
                class_specific=class_specific,
                coefs=coefs,
                log=log,
            )

        accu = tnt.test(
            model=protopnet,
            dataloader=test_loader,
            class_specific=class_specific,
            log=log,
        )
        accuracies.append(accu)
        save.save_model_w_condition(
            model=protopnet,
            model_dir=model_dir,
            model_name=f"{epoch}nopush",
            accu=accu,
            target_accu=0.70,
            log=log,
        )

        # Prototype pushing
        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader,
                prototype_network_parallel=protopnet,
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes="tasks/protopnet/imgdir1",
                epoch_number=epoch,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                save_prototype_class_identity=True,
                log=log,
            )

            accu = tnt.test(
                model=protopnet,
                dataloader=test_loader,
                class_specific=class_specific,
                log=log,
            )
            save.save_model_w_condition(
                model=protopnet,
                model_dir=model_dir,
                model_name=f"{epoch}push",
                accu=accu,
                target_accu=0.70,
                log=log,
            )

    plt.figure()
    plt.plot(range(num_train_epochs), [a * 100 for a in accuracies], marker='o')
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(model_dir / "accuracy_plot.png")
    plt.close()

    logclose()