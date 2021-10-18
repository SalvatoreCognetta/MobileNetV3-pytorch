import os
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import configparser
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mobilenetv3 import MobileNetV3Module
from utils import dotdict
from data import setup_cifar10, setup_mnist


# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device: {}".format(DEVICE))

AVAIL_GPUS = max(1, torch.cuda.device_count())
# BATCH_SIZE = 512 if AVAIL_GPUS else 64
# NUM_WORKERS = int(os.cpu_count() / 2)

AVAIL_DATASETS = ['cifar10', 'mnist']
NET_MODES = ['small', 'large']

if __name__ == "__main__":

    #----------------Configuration----------------
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    
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
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--mode", type=str, default='small', choices=NET_MODES, help="Dimension of the network, options: small, large")
    parser.add_argument("--dataset", type=str, default='cifar10', choices=AVAIL_DATASETS, help="Dataset to use for training, options: cifar10, mnist")
    parser.add_argument("-m", "--monitor", action="store_true", help="Param to enable the wandb monitor, for cpu and gpu usage")

    # Collect arguments
    args = parser.parse_args()

    assert args.dataset in AVAIL_DATASETS
    assert args.mode in NET_MODES
    network_name = "mobilenetv3-"+args.mode+'-'+args.dataset

    # Overrides the config.ini params
    if args.batch_size is not None:
        BATCH_SIZE = int(args.batch_size)
    if args.num_epochs is not None:
        MAX_EPOCHS = int(args.num_epochs)
    #-----------------------------------------

    #----------------WandB Monitor----------------
    if args.monitor:
        # Used to monitor CPU and GPU utils
        # Check: https://wandb.ai/site
        try:
            import wandb
            try:
                # 1. Start a new run
                wandb.init(project=network_name)
            except:
                print("wandb not initialized, start a local web server by running the command: $ wandb local")
                print("Running without wandb monitor...")
        except:
            print("wandb package not installed, install it via $ pip install wandb")
            print("Running without wandb monitor...")
    #---------------------------------------------


    #----------------Dataset----------------
    rgb_img = True
    # Create the dataset
    if args.dataset == 'cifar10':
        rgb_img = True # 3 channels
        cifar_path = conf['paths']['cifar_path']
        train_loader, valid_loader = setup_cifar10(cifar_path, BATCH_SIZE, NUM_WORKERS)
    elif args.dataset == 'mnist':
        rgb_img = False # 1 channel
        mnist_path = conf['paths']['mnist_path']
        train_loader, valid_loader = setup_mnist(mnist_path, BATCH_SIZE, NUM_WORKERS)
    #--------------------------------------


    #----------------Utilities----------------
    # Tensorboard Logger
    tb_logger = TensorBoardLogger("tb_logs", name=network_name)

    # Save best checkpoints at each epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val_Accuracy',
        filename=network_name,
        save_top_k=1,
        mode='max',
    )

    # Track training progresses and perform early stop
    early_stop_callback = EarlyStopping(
        monitor='val_Accuracy',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max'
    )
    #-----------------------------------------


    # Hyper-parameters
    hparams = dotdict({
        'num_classes': 10,
        'lr': 0.1,
        'momentum': 0.9,
        'dropout': 0.2,
        'weight_decay': 1e-5,
        'lr_decay':0.01,
        'step_size': 3,
        'batch_size': BATCH_SIZE,
        'epochs': MAX_EPOCHS
    })

    model = MobileNetV3Module(hparams, rgb_img, mode=args.mode)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=NUM_GPUS, accelerator=conf['trainer']['accelerator'], 
                        val_check_interval=1.0, max_epochs=MAX_EPOCHS, log_every_n_steps=20,\
                        progress_bar_refresh_rate=5, profiler='simple', \
                         logger=tb_logger, \
                        callbacks=[checkpoint_callback,early_stop_callback]) #precision=16,

    print("Precision: {}".format(trainer.precision))

    # Train the model ⚡⚡
    trainer.fit(model, train_loader, valid_loader)

    
    checkpoint_path = conf['paths']['checkpoint_path']
    # Saves only on the main process
    trainer.save_checkpoint(checkpoint_path+network_name)