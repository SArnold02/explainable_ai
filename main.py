import argparse
from data.dataset import Cub2011

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--download_data", action="store_true")

    return parser.parse_args()


def main(arguments):
    # Initialize the dataloader
    dataset = Cub2011("./data", download=arguments.download_data)
    

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)