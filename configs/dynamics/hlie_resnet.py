import torch

from lie_conv.dynamicsTrainer import HLieResNet
from lie_conv.lieGroups import T, SE2, SE2_canonical, SO2
from eqv_transformer.dynamics_predictor import DynamicsPredictor

from forge import flags

flags.DEFINE_string("group", "T(2)", "Group to be invariant to.")
flags.DEFINE_integer("channel_width", 384, "Channel width for the network.")
flags.DEFINE_integer("num_layers", 4, "Number of layers.")
flags.DEFINE_integer("model_seed", 0, "Model rng seed")
flags.DEFINE_integer(
    "lift_samples",
    1,
    "Number of coset lift samples to use for non-trivial stabilisers.",
)

def load(config, **unused_kwargs):

    if config.group == "T(2)":
        group = T(2)
    elif config.group == "T(3)":
        group = T(3)
    elif config.group == "SE(2)":
        group = SE2()
    elif config.group == "SE(2)_canonical":
        group = SE2_canonical()
    elif config.group == "SO(2)":
        group = SO2()
    else:
        raise NotImplementedError(f"Group {config.group} is not implemented.")

    torch.manual_seed(config.model_seed)  # TODO: initialization seed
    network = HLieResNet(
        sys_dim=config.sys_dim,
        d=config.space_dim,
        group=group,
        k=config.channel_width,
        num_layers=config.num_layers,
        liftsamples=config.lift_samples,
    )

    if config.data_config == "configs/dynamics/nbody_dynamics_data.py":
        task = "nbody"
    elif config.data_config == "configs/dynamics/spring_dynamics_data.py":
        task = "spring"

    dynamics_predictor = DynamicsPredictor(network, debug=config.debug, task=task)

    return dynamics_predictor, "HLieResNet_Dynamics"
