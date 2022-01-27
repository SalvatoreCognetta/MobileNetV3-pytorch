#! python3

import argparse
import importlib
import os

from src.logger.base_logger import logger

import src.utility.constants as CONST


def run(args):

    has_effect = False

    if args.split:
        try:
            mod_name = "src.{}".format(args.split)
            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)
            mod.run(args)

        except Exception as e:
            logger.exception("The script halted with an error")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test}")

def path(d):
    try:
        assert os.path.exists(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))


if __name__ == "__main__":
   
    #----------------Train Arguments----------------
    parser = argparse.ArgumentParser(description='MobileNet v3.')

    parser.add_argument('--split', nargs="?", choices=['train', 'test'], help='train the example or evaluate it')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=CONST.NET_MODES, default='small', help="Dimension of the network, options: small, large")
    parser.add_argument("--dataset", type=str, choices=CONST.AVAIL_DATASETS, default='cifar10', help=f"Dataset to use for training, options: {CONST.AVAIL_DATASETS}")
    parser.add_argument("--monitor", type=bool, default=False, help="Param to enable the wandb monitor, for cpu and gpu usage")
    parser.add_argument("--save_ckpt", type=bool, default=True, help="Set if the checkpoint must be saved or not, for testing.")

    # Collect arguments
    args = parser.parse_args()

    run(args)
