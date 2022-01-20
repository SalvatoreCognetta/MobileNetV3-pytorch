import os
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import configparser

from src.mobilenetv3 import MobileNetV3Module
from src.utility.utils import dotdict
import src.utility.constants as CONST
from src.data import setup_cifar10, setup_mnist


# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device: {}".format(DEVICE))

AVAIL_GPUS = max(1, torch.cuda.device_count())
# BATCH_SIZE = 512 if AVAIL_GPUS else 64
# NUM_WORKERS = int(os.cpu_count() / 2)

def run(args):
    #----------------Configuration----------------
    conf = configparser.ConfigParser()
    conf.read(os.path.dirname(__file__) + '/config.ini')
    
    # Constants
    SEED = int(conf['deterministic']['seed'])
    BATCH_SIZE  = int(conf['dataloader']['batch_size'])
    NUM_WORKERS = int(conf['dataloader']['num_workers'])
    NUM_GPUS  = int(conf['trainer']['num_gpus'])
    MAX_EPOCHS  = int(conf['hparams']['num_epochs'])

    if NUM_GPUS > AVAIL_GPUS:
        NUM_GPUS = AVAIL_GPUS
        print("WARNING: number of gpus greater than the available one. Using all the gpus. Modify the config.ini file.")

    # Deterministic
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    #----------------------------------------------

    #----------------Arguments----------------
    network_name = "mobilenetv3-"+args.mode+'-'+args.dataset
    #-----------------------------------------

    #----------------Dataset----------------
    rgb_img = True
    # Create the dataset
    if args.dataset == 'cifar10':
        rgb_img = True # 3 channels
        cifar_path = conf['paths']['cifar_path']
        train_loader, valid_loader = setup_cifar10(cifar_path, BATCH_SIZE, NUM_WORKERS)
        checkpoint_name = 'mobilenetv3-small-cifar10.ckpt'
    elif args.dataset == 'mnist':
        rgb_img = False # 1 channel
        mnist_path = conf['paths']['mnist_path']
        train_loader, valid_loader = setup_mnist(mnist_path, BATCH_SIZE, NUM_WORKERS)
        checkpoint_name = 'mobilenetv3-small-mnist.ckpt'
    #--------------------------------------

    model = MobileNetV3Module.load_from_checkpoint(checkpoint_path="../checkpoints/"+checkpoint_name)

    model(valid_loader)

    