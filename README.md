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

We use the CUB-200 and Stanford Cars dataset, in order to download it, use the --download_data flag, when running the main.py

## Getting the pre-trained ResNet

The checkpoint is uploaded with git lfs, to get it, follow the steps:

- sudo apt-get install git-lfs

- git lfs install

- git lfs fetch --all

If the git-lfs doesnt work, the checkpoints can be found here: https://drive.google.com/file/d/1qpS0POWhhvJICibQx-iNClToZsgpvoDn/view?usp=sharing