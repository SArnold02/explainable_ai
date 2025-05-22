# Techincal report

## Setup

- Clone the repo
- Create a python environment, venv example:
    ``` sh
    python -m venv ./.venv
    ```
- Load the created environment, venv example:
    ``` sh
    sourche ./.venv/bin/activate
    ```
- Install the dependencies:
    ``` sh
    pip install -r requirements.txt
    ```


## Data

We use the CUB-200 dataset, in order to download it, use the --download_data flag, when running the main.py

## Getting the pre-trained ResNet

The checkpoint is uploaded with git lfs, to get it, follow the steps:

- sudo apt-get install git-lfs

- git lfs install

- git lfs fetch --all