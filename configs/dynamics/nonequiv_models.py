import torch

from lie_conv.dynamicsTrainer import FC, HFC
from lie_conv.graphnets import OGN, HOGN
from eqv_transformer.dynamics_predictor import DynamicsPredictor

from forge import flags

flags.DEFINE_integer("channel_width", 256, "Channel width for the network.")
flags.DEFINE_integer("num_layers", 4, "Number of layers.")
flags.DEFINE_integer("model_seed", 0, "Model rng seed")

flags.DEFINE_string(
    "network_type",
    "FC",
    "One of FC, HFC, OGN, HOGN.",
)

def load(config, **unused_kwargs):

    print(f"Using network: {config.network_type}.")

    torch.manual_seed(config.model_seed)  
    network = (eval(config.network_type))(
        sys_dim=config.sys_dim,
        d=config.space_dim,
        k=config.channel_width,
        num_layers=config.num_layers,
    )

    if config.data_config == "configs/dynamics/nbody_dynamics_data.py":
        task = "nbody"
    elif config.data_config == "configs/dynamics/spring_dynamics_data.py":
        task = "spring"

    dynamics_predictor = DynamicsPredictor(network, debug=config.debug, task=task) # not using config.debug since debugging involves H (not present for FC).

    return dynamics_predictor, f"{config.network_type}_Dynamics"
