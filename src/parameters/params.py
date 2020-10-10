import argparse

###############################################################################################
#                                                                                             #
# This python file captures all parameters used in the model.                                 #
# We use argparse library to get parameters values as command line options.                   #
# We also group parameters to print things out in a logical structure when users type --help. #
#                                                                                             #
###############################################################################################

parser = argparse.ArgumentParser(description="Recurrent Attention Model")

# glimpse network parameters.
glimpse_args = parser.add_argument_group("Glimpse Network")

glimpse_args.add_argument(
    "--patch_size", type=int, default=8, help="size of the high resolution patch in px."
)
glimpse_args.add_argument(
    "--patch_scale", type=int, default=3, help="scale of successive patches."
)
glimpse_args.add_argument(
    "--num_patches", type=int, default=2, help="number of downscaled patches"
)
glimpse_args.add_argument(
    "--loc_hidden", type=int, default=128, help="hidden size of loc fc"
)
glimpse_args.add_argument(
    "--glimpse_hidden", type=int, default=128, help="hidden size of glimpse fc"
)


# recurrent memory parameters.
core_args = parser.add_argument_group("Recurrent Network Params")
core_args.add_argument(
    "--num_glimpses", type=int, default=8, help="# of glimpses"
)
core_args.add_argument(
    "--hidden_size", type=int, default=256, help="hidden size of rnn"
)


# reinforcement learning (ppo) params
ppo_args = parser.add_argument_group("PPO Parameters")
ppo_args.add_argument(
    "--std", type=float, default=0.05, help="gaussian policy standard deviation"
)
ppo_args.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)


# data params
data_args = parser.add_argument_group("Data Params")
data_args.add_argument(
    "--valid_size", type=float, default=0.1,
    help="Proportion of training set used for validation",
)
data_args.add_argument(
    "--batch_size", type=int, default=128, help="number of images in each batch of data"
)
data_args.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of subprocesses to use for data loading",
)
data_args.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="Whether to shuffle the train and valid indices",
)
data_args.add_argument(
    "--show_sample",
    action="store_true",
    default=False,
    help="Whether to visualize a sample grid of the data",
)


# training params
train_arg = parser.add_argument_group("Training Params")
train_arg.add_argument(
    "--train", action="store_true", default=False, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=500, help="number of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=9e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=50,
    help="Number of epochs to wait before stopping train",
)


# other params
misc_arg = parser.add_argument_group("Misc. Parameters")
misc_arg.add_argument(
    "--use_gpu", action="store_true", default=False, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    action="store_true",
    default=False,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./data", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    action="store_true",
    default=False,
    help="Whether to use tensorboard for visualization",
)
misc_arg.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=5, help="How frequently to plot glimpses"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
