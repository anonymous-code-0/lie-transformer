import sys
from os import path as osp
import time
import json
import types
from math import sqrt
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
from collections import defaultdict
import deepdish as dd
from tqdm import tqdm
import matplotlib.pyplot as plt

from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    print_reports,
    log_reports,
    load_checkpoint,
    save_checkpoint,
    delete_checkpoint,
    ExponentialMovingAverage,
    get_component,
    nested_to,
    param_count,
    get_component,
    get_average_norm,
    param_count,
    parameter_analysis,
)

from eqv_transformer.molecule_predictor import MoleculePredictor
from eqv_transformer.multihead_neural import MultiheadLinear
from lie_conv.utils import Pass, Expression
from lie_conv.masked_batchnorm import MaskBatchNormNd
from oil.utils.utils import cosLr
from corm_data.collate import collate_fn

if torch.cuda.is_available():
    device = "cuda"
    # device = "cpu"
else:
    device = "cpu"

#####################################################################################################################
# Command line flags
#####################################################################################################################
# Directories
flags.DEFINE_string("data_dir", "data/", "Path to data directory")
flags.DEFINE_string(
    "results_dir", "checkpoints/", "Top directory for all experimental results."
)

# Configuration files to load
flags.DEFINE_string(
    "data_config", "configs/molecule/qm9_data.py", "Path to a data config file."
)
flags.DEFINE_string(
    "model_config",
    "configs/molecule/set_transformer.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string("run_name", "test", "Name of this job and name of results folder.")
flags.DEFINE_boolean("resume", False, "Tries to resume a job if True.")

# Logging
flags.DEFINE_integer(
    "report_loss_every", 500, "Number of iterations between reporting minibatch loss."
)
flags.DEFINE_integer(
    "evaluate_every", 10000, "Number of iterations between reporting validation loss."
)
flags.DEFINE_integer(
    "save_check_points",
    10,
    "frequency with which to save checkpoints, in number of epochs.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_float(
    "ema_alpha", 0.99, "Alpha coefficient for exponential moving average of train logs."
)

# Optimization
flags.DEFINE_integer("train_epochs", 500, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 90, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-5, "SGD learning rate.")
flags.DEFINE_float("beta1", 0.5, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.9, "Adam Beta 2 parameter")
flags.DEFINE_string(
    "lr_schedule",
    "none",
    "What learning rate schedule to use. Options: cosine, none",
)
flags.DEFINE_boolean(
    "parameter_count", False, "If True, print model parameter count and exit"
)
flags.DEFINE_boolean("debug", False, "Enable additional telemetry for debugging")
flags.DEFINE_boolean(
    "init_activations",
    False,
    "produce initialisation activation histograms the activations of specified modules through training",
)
flags.DEFINE_boolean("profile_model", False, "Run profiling code on model and exit")
flags.DEFINE_float(
    "lr_floor", 0, "minimum multiplicative factor of the learning rate in annealing"
)
flags.DEFINE_float(
    "warmup_length", 0.01, "fraction of the training time to use for warmup"
)
flags.DEFINE_bool(
    "find_spikes", False, "Find big spikes in validation loss and save checkpoints"
)
flags.DEFINE_boolean(
    "only_store_last_checkpoint",
    False,
    "If True, deletes last checkpoint when saving current checkpoint",
)
flags.DEFINE_boolean(
    "clip_grad",
    False,
    "If True, clip gradient L2-norms at 1.",
)


flags.DEFINE_string("load_path", None, "Path to load config, model checkpoints from")
flags.DEFINE_string("checkpoint", "model.ckpt-final", "Checkpoint file to load from")
flags.DEFINE_integer("samples", 20, "Number of samples to take to estimate loss")


def test_model(model, dataloader, seed=None):
    preds = torch.Tensor().to(device)

    with torch.no_grad():
        for data in dataloader:
            if seed is not None:
                torch.random.manual_seed(seed)
            data = {k: v.to(device) for k, v in data.items()}
            outputs = model(data, compute_loss=True)
            preds = torch.cat([preds, outputs.prediction_actual])

    return preds


def multisample_model(model, dataloader, actuals, samples, fix_seed=False):
    preds_all = []

    for i in range(samples):
        preds = test_model(model, dataloader, i if fix_seed else None)

        preds_all.append(preds.unsqueeze(-1))

        print(
            f"mae with {i+1} samples: {(torch.cat(preds_all, dim=-1).mean(dim=1) - actuals).abs().mean().item()}"
        )

    preds_all = torch.cat(preds_all, dim=-1)

    return preds_all


def main():
    config = forge.config()

    # print(config.__dict__["__flags"])

    with open(osp.join(config.load_path, "flags.json"), "r") as f:
        run_config = json.load(f)
    # print(run_config)

    config = types.SimpleNamespace(**{**config.__dict__["__flags"], **run_config})

    print(config)

    # Load data
    dataloaders, num_species, charge_scale, ds_stats, data_name = fet.load(
        config.data_config, config=config
    )

    test_dataloader = DataLoader(
        dataloaders["test"].dataset,
        batch_size=300,
        num_workers=0,
        shuffle=False,  # False,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    actuals = torch.Tensor().to(device)
    meadian, mad = ds_stats

    with torch.no_grad():
        for data in test_dataloader:
            data = {k: v.to(device) for k, v in data.items()}
            actuals = torch.cat([actuals, data[config.task]])

    config.num_species = num_species
    config.charge_scale = charge_scale
    config.ds_stats = ds_stats

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model.to(device)

    config.charge_scale = float(config.charge_scale.numpy())
    config.ds_stats = [float(stat.numpy()) for stat in config.ds_stats]

    load_checkpoint(osp.join(config.load_path, config.checkpoint), model)

    multisample = multisample_model(
        model, test_dataloader, actuals, config.samples, fix_seed=True
    )

    multisample_maes = (
        (multisample - actuals.unsqueeze(1)).abs().mean(dim=0).cpu().numpy()
    )

    multisample_average = multisample.mean(dim=1)
    multisample_average_mae = (multisample_average - actuals).abs().mean().cpu().numpy()

    other_samples = []
    for i in range(config.samples):
        with torch.no_grad():
            model.eval()
            test_mae = 0.0
            for data in test_dataloader:
                data = {k: v.to(device) for k, v in data.items()}
                outputs = model(data, compute_loss=True)
                test_mae = test_mae + outputs.mae

            other_samples.append(test_mae / len(test_dataloader))

    with open(
        osp.join(config.load_path, "sample_results_" + config.checkpoint + ".txt"), "w"
    ) as f:
        f.write("Sampled MAEs: " + str(multisample_maes) + "\r\n")
        f.write(
            "Spike:: "
            + "Min: "
            + str(multisample_maes.min())
            + ", Mean: "
            + str(multisample_maes.mean())
            + ", Max: "
            + str(multisample_maes.max())
            + "\r\n"
        )
        f.write("Averaged outputs MAE: " + str(multisample_average_mae) "\r\n")
        f.write("Other samples: " + str(other_samples))


if __name__ == "__main__":
    main()
