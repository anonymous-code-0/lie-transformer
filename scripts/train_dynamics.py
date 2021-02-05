from os import path as osp
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import transformers

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

# torch.autograd.set_detect_anomaly(True)

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
import deepdish as dd
from tqdm import tqdm
from itertools import chain
from lie_conv.hamiltonian import EuclideanK, SpringV, SpringH, HamiltonianDynamics
from types import SimpleNamespace


from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    parse_reports_cpu,
    print_reports,
    load_checkpoint,
    save_checkpoint,
    ExponentialMovingAverage,
    nested_to,
    param_count,
    get_component,
    get_average_norm,
)


# %%

#####################################################################################################################
# Command line flags
#####################################################################################################################
# Directories
flags.DEFINE_string(
    "results_dir", "checkpoints/", "Top directory for all experimental results."
)

# Configuration files to load
flags.DEFINE_string(
    "data_config",
    "configs/dynamics/spring_dynamics_data.py",
    "Path to a data config file.",
)
flags.DEFINE_string(
    "model_config",
    "configs/dynamics/eqv_transformer_model.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string(
    "run_name",
    "test",
    "Name of this job and name of results folder.",
)
flags.DEFINE_boolean("resume", False, "Tries to resume a job if True.")

# Logging
flags.DEFINE_integer(
    "report_loss_every", 10, "Number of iterations between reporting minibatch loss."
)
flags.DEFINE_integer(
    "evaluate_every", 10000, "Number of iterations between reporting validation loss."
)
flags.DEFINE_integer(
    "save_check_points",
    50,
    "frequency with which to save checkpoints, in number of epoches.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")

# Optimization
flags.DEFINE_integer("train_epochs", 200, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 100, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-3, "Adam learning rate.")
flags.DEFINE_float("beta1", 0.9, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.999, "Adam Beta 2 parameter")
flags.DEFINE_string(
    "lr_schedule", "cosine_annealing", "Learning rate schedule."
) 
flags.DEFINE_boolean(
    "clip_grad_norm", False, "Clip norm of the gradient at max_grad_norm."
)
flags.DEFINE_integer(
    "max_grad_norm", 100, "Maximum norm of gradient when clip_grad_norm is True."
)

# GPU device
flags.DEFINE_integer("device", 0, "GPU to use.")

# Debug mode tracks more stuff
flags.DEFINE_boolean("debug", False, "Track and show on tensorboard more metrics.")
flags.DEFINE_boolean(
    "kill_if_poor", False, "Kills run if loss is poor. Exact params to be set below."
)
flags.DEFINE_boolean(
    "save_test_predictions",
    False,
    "Makes and saves test predictions on one or more test sets (e.g. 5-step and 100-step predictions) at the end of training.",
)

#####################################################################################################################


# @forge.debug_on(Exception)
def main():
    # Parse flags
    config = forge.config()

    # Set device
    if torch.cuda.is_available():
        device = f"cuda:{config.device}"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    # Load data
    dataloaders, data_name = fet.load(config.data_config, config=config)

    train_loader = dataloaders["train"]
    test_loader = dataloaders["test"]
    val_loader = dataloaders["val"]

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model = model.to(device)
    print(model)

    # Prepare environment
    params_in_run_name = [
        ("batch_size", "bs"),
        ("learning_rate", "lr"),
        ("num_heads", "nheads"),
        ("num_layers", "nlayers"),
        ("dim_hidden", "hdim"),
        ("kernel_dim", "kdim"),
        ("location_attention", "locatt"),
        ("model_seed", "mseed"),
        ("lr_schedule", "lrsched"),
        ("layer_norm", "ln"),
        ("batch_norm", "bn"),
        ("channel_width", "width"),
        ("attention_fn", "attfn"),
        ("output_mlp_scale", "mlpscale"),
        ("train_epochs", "epochs"),
        ("block_norm", "block"),
        ("kernel_type", "ktype"),
        ("architecture", "arch"),
        ("activation_function", "act"),
        ("space_dim", "spacedim"),
        ("num_particles", "prtcls"),
        ("n_train", "ntrain"),
        ("group", "group"),
        ("lift_samples", "ls"),
    ]

    run_name = "" 
    for config_param in params_in_run_name:
        attr = config_param[0]
        abbrev = config_param[1]

        if hasattr(config, attr):
            run_name += abbrev
            run_name += str(getattr(config, attr))
            run_name += "_"

    if config.clip_grad_norm:
        run_name += "clipnorm" + str(config.max_grad_norm) + "_"

    results_folder_name = osp.join(
        data_name,
        model_name,
        config.run_name,
        run_name,
    )

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Print flags
    fet.print_flags()

    # Setup optimizer
    model_params = model.predictor.parameters()

    opt_learning_rate = config.learning_rate
    model_opt = torch.optim.Adam(
        model_params, lr=opt_learning_rate, betas=(config.beta1, config.beta2)
    )

    if config.lr_schedule == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_opt, config.train_epochs
        )
    elif config.lr_schedule == "cosine_annealing_warmup":
        num_warmup_epochs = int(0.05 * config.train_epochs)
        num_warmup_steps = len(train_loader) * num_warmup_epochs
        num_training_steps = len(train_loader) * config.train_epochs
        scheduler = transformers.get_cosine_schedule_with_warmup(
            model_opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # Try to restore model and optimizer from checkpoint
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, model_opt)
    else:
        start_epoch = 1

    train_iter = (start_epoch - 1) * (
        len(train_loader.dataset) // config.batch_size
    ) + 1

    print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    train_reports = []
    report_all = {}
    report_all_val = {}
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, loss=0.0)
    print("finished saving model at epoch 0 before training")

    # if (
    #     config.debug
    #     and config.model_config == "configs/dynamics/eqv_transformer_model.py"
    # ):
    #     model_components = (
    #         [(0, [], "embedding_layer")]
    #         + list(
    #             chain.from_iterable(
    #                 (
    #                     (k, [], f"ema_{k}"),
    #                     (
    #                         k,
    #                         ["ema", "kernel", "location_kernel"],
    #                         f"ema_{k}_location_kernel",
    #                     ),
    #                     (
    #                         k,
    #                         ["ema", "kernel", "feature_kernel"],
    #                         f"ema_{k}_feature_kernel",
    #                     ),
    #                 )
    #                 for k in range(1, config.num_layers + 1)
    #             )
    #         )
    #         + [(config.num_layers + 2, [], "output_mlp")]
    #     )  # components to track for debugging

    num_params = param_count(model)
    print(f"Number of model parameters: {num_params}")

    # Training
    start_t = time.time()

    total_train_iters = len(train_loader) * config.train_epochs
    iters_per_eval = int(total_train_iters / 100) # evaluate 100 times over the course of training

    assert (
        config.n_train % min(config.batch_size, config.n_train) == 0
    ), "Batch size doesn't divide dataset size. Can be problematic for loss computation (see below)."

    training_failed = False
    best_val_loss_so_far = 1e+7

    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(train_loader):
            data = nested_to(
                data, device, torch.float32
            )  # the format is ((z0, sys_params, ts), true_zs) for data
            outputs = model(data)

            if torch.isnan(outputs.loss):
                if not training_failed:
                    epoch_of_nan = epoch
                if (epoch > epoch_of_nan + 1) and training_failed:
                    raise ValueError("Loss Nan-ed.")
                training_failed = True

            model_opt.zero_grad()
            outputs.loss.backward(retain_graph=False)

            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm, norm_type=1
                )

            model_opt.step()

            train_reports.append(parse_reports_cpu(outputs.reports))

            if config.log_train_values:
                reports = parse_reports(outputs.reports)
                if batch_idx % config.report_loss_every == 0:
                    log_tensorboard(summary_writer, train_iter, reports, "train/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="train",
                    )
                    log_tensorboard(
                        summary_writer,
                        train_iter,
                        {"lr": model_opt.param_groups[0]["lr"]},
                        "hyperparams/",
                    )

            # Do learning rate schedule steps per STEP for cosine_annealing_warmup
            if config.lr_schedule == "cosine_annealing_warmup":
                scheduler.step()

            # Track stuff for debugging
            if config.debug:
                model.eval()
                with torch.no_grad():
                    curr_params = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }

                    # updates norm
                    if train_iter == 1:
                        update_norm = 0
                    else:
                        update_norms = []
                        for (k_prev, v_prev), (k_curr, v_curr) in zip(
                            prev_params.items(), curr_params.items()
                        ):
                            assert k_prev == k_curr
                            if (
                                "tracked" not in k_prev
                            ): 
                                update_norms.append((v_curr - v_prev).norm(1).item())
                        update_norm = sum(update_norms)

                    # gradient norm
                    grad_norm = 0
                    for p in model.parameters():
                        try:
                            grad_norm += p.grad.norm(1)
                        except AttributeError:
                            pass

                    # Average V size:
                    (z0, sys_params, ts), true_zs = data

                    z = z0
                    m = sys_params[..., 0]  # assume the first component encodes masses
                    D = z.shape[-1]  # of ODE dims, 2*num_particles*space_dim
                    q = z[:, : D // 2].reshape(*m.shape, -1)
                    k = sys_params[..., 1]

                    V = model.predictor.compute_V((q, sys_params))
                    V_true = SpringV(q, k)

                    log_tensorboard(
                        summary_writer,
                        train_iter,
                        {
                            "avg_update_norm1": update_norm / num_params,
                            "avg_grad_norm1": grad_norm / num_params,
                            "avg_predicted_potential_norm": V.norm(1) / V.numel(),
                            "avg_true_potential_norm": V_true.norm(1) / V_true.numel(),
                        },
                        "debug/",
                    )
                    prev_params = curr_params

                model.train()

            # Logging
            if train_iter % iters_per_eval == 0 or (
                train_iter == total_train_iters
            ):  # batch_idx % config.evaluate_every == 0:
                model.eval()
                with torch.no_grad():

                    reports = None
                    for data in test_loader:
                        data = nested_to(data, device, torch.float32)
                        outputs = model(data)

                        if reports is None:
                            reports = {
                                k: v.detach().clone().cpu()
                                for k, v in outputs.reports.items()
                            }
                        else:
                            for k, v in outputs.reports.items():
                                reports[k] += v.detach().clone().cpu()

                    for k, v in reports.items():
                        reports[k] = v / len(
                            test_loader
                        )  

                    reports = parse_reports(reports)
                    reports["time"] = time.time() - start_t
                    if report_all == {}:
                        report_all = deepcopy(reports)

                        for d in reports.keys():
                            report_all[d] = [report_all[d]]
                    else:
                        for d in reports.keys():
                            report_all[d].append(reports[d])

                    log_tensorboard(summary_writer, train_iter, reports, "test/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="test",
                    )

                    if config.kill_if_poor:
                        if epoch > config.train_epochs * 0.2:
                            if reports["mse"] > 0.01:
                                raise RuntimeError(
                                    f"Killed run due to poor performance."
                                )

                    # repeat for validation data
                    reports = None
                    for data in val_loader:
                        data = nested_to(data, device, torch.float32)
                        outputs = model(data)

                        if reports is None:
                            reports = {
                                k: v.detach().clone().cpu()
                                for k, v in outputs.reports.items()
                            }
                        else:
                            for k, v in outputs.reports.items():
                                reports[k] += v.detach().clone().cpu()

                    for k, v in reports.items():
                        reports[k] = v / len(
                            val_loader
                        )  

                    reports = parse_reports(reports)
                    reports["time"] = time.time() - start_t
                    if report_all_val == {}:
                        report_all_val = deepcopy(reports)

                        for d in reports.keys():
                            report_all_val[d] = [report_all_val[d]]
                    else:
                        for d in reports.keys():
                            report_all_val[d].append(reports[d])

                    log_tensorboard(summary_writer, train_iter, reports, "val/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="val",
                    )

                    if report_all_val['mse'][-1] < best_val_loss_so_far:
                        save_checkpoint(checkpoint_name, f"early_stop", model, model_opt, loss=outputs.loss)
                        best_val_loss_so_far = report_all_val['mse'][-1] 

                model.train()

            train_iter += 1

        if config.lr_schedule == "cosine_annealing":
            scheduler.step()

        if epoch % config.save_check_points == 0:
            save_checkpoint(
                checkpoint_name, train_iter, model, model_opt, loss=outputs.loss
            )

        dd.io.save(logdir + "/results_dict_train.h5", train_reports)
        dd.io.save(logdir + "/results_dict.h5", report_all)
        dd.io.save(logdir + "/results_dict_val.h5", report_all_val)

    # always save final model
    save_checkpoint(checkpoint_name, train_iter, model, model_opt, loss=outputs.loss)

    if config.save_test_predictions:
        print("Starting to make model predictions on test sets for *final model*.")
        for chunk_len in [5, 100]:
            start_t_preds = time.time()
            data_config = SimpleNamespace(
                **{**config.__dict__["__flags"], **{"chunk_len": chunk_len, "batch_size": 500}}
            )
            dataloaders, data_name = fet.load(config.data_config, config=data_config)
            test_loader_preds = dataloaders["test"]

            torch.cuda.empty_cache()
            with torch.no_grad():
                preds = []
                true = []
                num_datapoints = 0
                for idx, d in enumerate(test_loader_preds):
                    true.append(d[-1])
                    d = nested_to(d, device, torch.float32)
                    outputs = model(d)

                    pred_zs = outputs.prediction
                    preds.append(pred_zs)

                    num_datapoints += len(pred_zs)

                    if num_datapoints >= 2000:
                        break

                preds = torch.cat(preds, dim=0).cpu()
                true = torch.cat(true, dim=0).cpu()

                save_dir = osp.join(
                    logdir, f"traj_preds_{chunk_len}_steps_2k_test.pt"
                )
                torch.save(preds, save_dir)

                save_dir = osp.join(
                    logdir, f"traj_true_{chunk_len}_steps_2k_test.pt"
                )
                torch.save(true, save_dir)

                print(
                    f"Completed making test predictions for chunk_len = {chunk_len} in {time.time() - start_t_preds:.2f} seconds."
                )


if __name__ == "__main__":
    main()
