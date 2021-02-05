import os

import torch
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, islice
from oil.datasetup.datasets import split_dataset
from lie_conv.datasets import SpringDynamics

from forge import flags

flags.DEFINE_integer("n_train", 3000, "Number of training datapoints.")
flags.DEFINE_integer("n_test", 2000, "Number of testing datapoints.")
flags.DEFINE_integer("n_val", 2000, "Number of validation datapoints.")
flags.DEFINE_integer("n_systems", 10000, "Size of total dataset generated.")
flags.DEFINE_string(
    "data_path",
    "./datasets/ODEDynamics/SpringDynamics/",
    "Dataset is loaded from and/or downloaded to this path.",
)
flags.DEFINE_integer("sys_dim", 2, "[add description].")
flags.DEFINE_integer("space_dim", 2, "Dimension of particle system.")
flags.DEFINE_integer("data_seed", 0, "Data splits random seed.")
flags.DEFINE_integer("num_particles", 6, "Number of particles in system.")
flags.DEFINE_integer("chunk_len", 5, "Length of trajectories.")
flags.DEFINE_boolean(
    "load_preprocessed",
    False,
    "Load data already preprocessed to avoid RAM memory spike. Ensure data exists first for the chunk_lun required.",
)


def load(config):

    dataset = SpringDynamics(
        n_systems=config.n_systems,
        root_dir=config.data_path,
        space_dim=config.space_dim,
        num_particles=config.num_particles,
        chunk_len=config.chunk_len,
        load_preprocessed=config.load_preprocessed,
    )

    splits = {
        "train": config.n_train,
        "val": min(config.n_train, config.n_val),
        "test": config.n_test,
    }

    with FixedNumpySeed(config.data_seed):
        datasets = split_dataset(dataset, splits)

    dataloaders = {
        k: DataLoader(
            v,
            batch_size=min(config.batch_size, config.n_train),
            num_workers=0,
            shuffle=(k == "train"),
        )
        for k, v in datasets.items()
    }

    dataloaders["Train"] = islice(dataloaders["train"], len(dataloaders["val"]))

    return dataloaders, f"spring_dynamics"

