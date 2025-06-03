import kagglehub
import argparse
from data.dataset import Cub2011, DEFAULT_TRAIN_TRANSFORM, DEFAULT_VAL_TRANSFORM
from tasks import run_task, Task
from torchvision.datasets import StanfordCars


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Choose the task
    parser.add_argument("--task", type=Task, choices=list(Task), default=Task.BASELINE)
    parser.add_argument("--train_run", action="store_true")

    # Training and data arguments
    parser.add_argument("--dataset", type=str, default="cub")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resnet_checkpoint", type=str, default=None)

    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr_schedule", type=int, default=10)

    parser.add_argument("--image_size", type=tuple[int, int], default=(224, 224))
    parser.add_argument("--box_size", type=tuple[int, int], default=(78, 78))

    # ProtoTree specific options
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--W1", type=int, default=1)
    parser.add_argument("--H1", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--num_parts", type=int, default=15)
    parser.add_argument("--log_probabilities", action="store_true")
    parser.add_argument("--kont_algorithm", action="store_true")
    parser.add_argument("--lr_net", type=float, default=1e-5)
    parser.add_argument("--lr_block", type=float, default=0.001)
    parser.add_argument("--freeze_epoch", type=int, default=30)
    parser.add_argument("--upsample_threshold", type=float, default=0.98)

    return parser.parse_args()

python3 main.py --task prototree --num_workers 16 --print_every 100 --batch_size 32 --epoch 150 --train_run --depth=9 --resnet_checkpoint ./pre_trained/cub/model_checkpoint.pth --device cuda --lr_net 0.001 --patience 20 --freeze_epoch 30 --lr 0.1 --lr_block 0.1 --lr_net 1e-5 --kont_algorithm
python main.py --task prototree --num_workers 4 --print_every 100 --batch_size 32 --epoch 60 --depth=9 --kont_algorithm --checkpoint ./outputs/25-05-30-14-27-32/model_checkpoint.pth --device cuda --train_run --lr 0.1 --lr_block 0.1
def add_dataset_parameters(arguments):
    match arguments.dataset:
        case "cub":
            arguments.num_classes = 200
        case "cars":
            arguments.num_classes = 196
        case _:
            raise ValueError(f"Dataset {arguments.dataset} is not supported! ('cub' or 'cars')")


def main(arguments):
    # Initialize the datasets
    if arguments.dataset == "cub":
        train_dataset = Cub2011(train=True, download=arguments.download_data)
        val_dataset = Cub2011(train=False)
    else:
        if arguments.download_data:
            print("Downloading Stanford Cars dataset")
            kagglehub.dataset_download("rickyyyyyyy/torchvision-stanford-cars", "./data")

        train_dataset = StanfordCars(
            root="./data",
            split="train",
            transform=DEFAULT_TRAIN_TRANSFORM
        )
        val_dataset = StanfordCars(root="./data", split="test", transform=DEFAULT_VAL_TRANSFORM)

    # Choose the task and run it
    run_task(arguments, train_dataset, val_dataset)


if __name__ == "__main__":
    arguments = parse_arguments()
    add_dataset_parameters(arguments)
    main(arguments)