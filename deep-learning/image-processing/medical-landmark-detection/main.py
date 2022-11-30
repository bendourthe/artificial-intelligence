# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import argparse
import torch

from app.runner import Runner


# --------------------------------------------------------------------------- #
# METHOD DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("-r", "--run_name", type=str, required=True)
    parser.add_argument("-d", "--run_dir", type=str, required=True, default=".runs")
    parser.add_argument("-p", "--phase", choices=['train', 'validate', 'test'], required=True)
    # Optional arguments
    parser.add_argument("-C", "--config", default="config.yaml")
    parser.add_argument("-c", "--checkpoint", default="checkpoint path")
    parser.add_argument("-g", "--cuda_devices", default="0")
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-l", "--localNet", type=str)
    parser.add_argument("-ds", "--name_list", nargs="+", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-L", "--lr", type=float)
    parser.add_argument("-w", "--weight_decay", type=float)
    parser.add_argument("-s", "--sigma", type=int)
    parser.add_argument("-x", "--mix_step", type=int)
    parser.add_argument("-u", "--use_background_channel", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# JOB RUN
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = get_args()
    Runner(args).run()
