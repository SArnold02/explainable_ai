import argparse
from data.dataset import Cub2011
from tasks import run_task, Task


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Choose the task
    parser.add_argument("--task", type=Task, choices=list(Task), default=Task.BASELINE)

    # Training and data arguments
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr_schedule", type=int, default=10)

    return parser.parse_args()

def main(arguments):
    # Initialize the datasets
    train_dataset = Cub2011(train=True, download=arguments.download_data)
    val_dataset = Cub2011(train=False)
    
    # Choose the task and run it
    run_task(arguments, train_dataset, val_dataset)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)