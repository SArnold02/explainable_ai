from enum import Enum
from tasks.baseline import run_baseline
from tasks.pipnet import run_pipnet
from tasks.protopnet import run_protopnet
from tasks.prototree import run_prototree
from tasks.tesnet import run_tesnet


class Task(Enum):
    BASELINE = "baseline"
    PROTONET = "protonet"
    PROTOTREE = "prototree"
    TESNET = "tesnet"
    PIPNET = "pipnet"

def run_task(arguments, train_dataset, val_dataset):
    # Choose the task you want to run
    match arguments.task:
        case Task.BASELINE:
            run_baseline(arguments, train_dataset, val_dataset)
        case Task.PROTONET:
            run_protopnet(arguments, train_dataset, val_dataset)
        case Task.PROTOTREE:
            run_prototree(arguments, train_dataset, val_dataset)
        case Task.TESNET:
            run_tesnet(arguments, train_dataset, val_dataset)
        case Task.PIPNET:
            run_pipnet(arguments, train_dataset, val_dataset)
        case _:
            raise ValueError(f"Ivalid task provided: {arguments.task}")