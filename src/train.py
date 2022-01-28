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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.mobilenetv3 import MobileNetV3Module
from src.utility.utils import dotdict
from src.data import setup_cifar10, setup_mnist

from src.logger.base_logger import logger

# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug("Detected device: {}".format(DEVICE))

AVAIL_CPUS = os.cpu_count() - 1 # Leave one for os
AVAIL_GPUS = max(1, torch.cuda.device_count())

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

    if NUM_WORKERS > AVAIL_CPUS: NUM_WORKERS = AVAIL_CPUS
    if NUM_GPUS > AVAIL_GPUS: NUM_GPUS = AVAIL_GPUS

    # Deterministic
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pl.seed_everything(SEED)

    torch.backends.cudnn.deterministic = True
    logger.info("CuDNN is enabled: {}".format(torch.backends.cudnn.enabled))
    #----------------------------------------------

    #----------------Arguments---------------------
    network_name = "mobilenetv3-"+args.mode+'-'+args.dataset

    # Overrides the config.ini params
    if args.batch_size is not None:
        BATCH_SIZE = int(args.batch_size)
    if args.num_epochs is not None:
        MAX_EPOCHS = int(args.num_epochs)
    #----------------------------------------------

    #----------------Dataset-----------------------
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
    #---------------------------------------------

    # #----------------WandB Monitor----------------
    # if args.monitor:
    #     # Used to monitor CPU and GPU utils
    #     # Check: https://wandb.ai/site
    #     try:
    #         import wandb

    #         try:
    #             # 1. Start a new run
    #             wandb.init(project=network_name)
    #         except:
    #             print("wandb not initialized, start a local web server by running the command: $ wandb local")
    #             print("Running without wandb monitor...")
    #     except:
    #         print("wandb package not installed, install it via $ pip install wandb")
    #         print("Running without wandb monitor...")
    # #---------------------------------------------
    
    
    # Hyper-parameters
    hparams = dotdict({
        'num_classes': 10,
        'lr': 0.1,
        'momentum': 0.9,
        'dropout': 0.8,
        'weight_decay': 1e-5,
        'lr_decay':0.01,
        'step_size': 3,
        'batch_size': BATCH_SIZE,
        'epochs': MAX_EPOCHS
    })
    # INPUT_SIZE = len(train_data[0])
    # hparams['input_dim'] = INPUT_SIZE

    # logger.debug(f"Input size: {INPUT_SIZE}")


    #----------------Utilities----------------
    loggers = [
        TensorBoardLogger(
            save_dir="tb_logs",
            name=network_name,
        ),
        WandbLogger(
            project=network_name,
            config=hparams,
            log_model=True,
        ) # TensorBoard Logger
    ]
    
    # if args.monitor:
    #     loggers.extend(WandbLogger(
    #         project=network_name,
    #         config=hparams,
    #         log_model=True,
    #     )) # Weight and Biases Logger

    callbacks = [
        # Save best checkpoints at each epoch
        ModelCheckpoint(
            monitor='train_Accuracy',
            filename=network_name,
            save_top_k=1,
            mode='max',
        ),

        # # Track training progresses and perform early stop
        # EarlyStopping(
        #     monitor='train_Accuracy',
        #     min_delta=0.001,
        #     patience=5,
        #     verbose=True,
        #     mode='max'
        # )
    ]
    #-----------------------------------------

    # For CIFAR10 the stride of the first convolutional must be set to 1
    model = MobileNetV3Module(hparams, rgb_img, mode=args.mode)

    # Initialize a trainer
    trainer = pl.Trainer(#gpus=NUM_GPUS, accelerator=conf['trainer']['accelerator'], 
                        val_check_interval=1.0, max_epochs=MAX_EPOCHS, log_every_n_steps=20,\
                        progress_bar_refresh_rate=5, profiler='simple', \
                        logger=loggers, callbacks=callbacks) #precision=16,

    logger.info("Precision: {}".format(trainer.precision))
    logger.info("Train the mobilenetv3 model.")


    # Train the model ⚡⚡
    trainer.fit(model, train_loader, valid_loader)
    
    if args.save_ckpt:
        checkpoint_path = conf['paths']['checkpoint_path']
        # Saves only on the main process
        trainer.save_checkpoint(checkpoint_path+network_name)