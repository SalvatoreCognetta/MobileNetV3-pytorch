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

    assert NUM_GPUS <= AVAIL_GPUS

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
    parser.add_argument("--mode", type=str, default='small', help="Dimension of the network, options: small, large")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use for training, options: cifar10, mnist")

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


    #----------------Dataset----------------
    # Create the dataset
    if args.dataset == 'cifar10':
        cifar_path = conf['paths']['cifar_path']
        train_loader, valid_loader = setup_cifar10(cifar_path, BATCH_SIZE, NUM_WORKERS)
    elif args.dataset == 'mnist':
        mnist_path = conf['paths']['mnist_path']
        train_loader, valid_loader = setup_mnist(mnist_path, BATCH_SIZE, NUM_WORKERS)
    #--------------------------------------
        

    # Hyper-parameters
    hparams = dotdict({
        'num_classes': 10,
        'dropout': 0.2,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'batch_size': BATCH_SIZE,
        'momentum': 0.9,
        'epochs': MAX_EPOCHS
    })

    # Tensorboard Logger
    tb_logger = TensorBoardLogger("tb_logs", name=network_name)

    # Save best checkpoints at each epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename=network_name,
        save_top_k=1,
        mode='min',
    )

    # Track training progresses and perform early stop
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    model = MobileNetV3Module(hparams, mode=args.mode)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=NUM_GPUS, accelerator=conf['trainer']['accelerator'], 
                        val_check_interval=1.0, max_epochs=MAX_EPOCHS, log_every_n_steps=20,\
                        progress_bar_refresh_rate=5, profiler='simple', \
                        precision=16, logger=tb_logger, \
                        callbacks=[checkpoint_callback,early_stop_callback]) 

    print("Precision: {}".format(trainer.precision))

    # Train the model ⚡⚡
    trainer.fit(model, train_loader, valid_loader)