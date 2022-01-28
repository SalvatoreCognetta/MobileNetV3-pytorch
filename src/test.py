import os
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import configparser

from src.mobilenetv3 import MobileNetV3Module
from src.utility.utils import dotdict
import src.utility.constants as CONST
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
    
    if NUM_WORKERS > AVAIL_CPUS:
        NUM_WORKERS = AVAIL_CPUS
        logger.warning("WARNING: number of cpu cores greater than the available one. Using all the cores-1. Modify the config.ini file.")

    if NUM_GPUS > AVAIL_GPUS:
        NUM_GPUS = AVAIL_GPUS
        logger.warning("WARNING: number of gpus greater than the available one. Using all the gpus. Modify the config.ini file.")

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
        checkpoint_name = 'mobilenetv3-'+args.mode+'-cifar10.ckpt'
    elif args.dataset == 'mnist':
        rgb_img = False # 1 channel
        mnist_path = conf['paths']['mnist_path']
        train_loader, valid_loader = setup_mnist(mnist_path, BATCH_SIZE, NUM_WORKERS)
        checkpoint_name = 'mobilenetv3-'+args.mode+'-mnist.ckpt'
    #--------------------------------------
    logger.info(f"Loading checkpoint {checkpoint_name}")
    model = MobileNetV3Module.load_from_checkpoint(checkpoint_path="./checkpoints/"+checkpoint_name, rgb_img=rgb_img, mode=args.mode)
    # model.to(DEVICE)

    dataiter = iter(valid_loader)
    logger.info(f"Classes map: {valid_loader.dataset.class_to_idx}")
    logger.info(f"Classes: {valid_loader.dataset.classes}")

    # images, labels = dataiter.next()
    errors = 0
    num_of_items = 0
    y_true, y_pred = [], []

    for image, label in tqdm(dataiter):
        predictions =  torch.argmax(model(image), dim=1)
        errors += label.shape[0] - torch.sum(predictions  == label)
        num_of_items += label.shape[0]

        y_true += label.tolist()
        y_pred += predictions.tolist()

    logger.info(f"Number of errors: {errors}/{num_of_items}")

    #importing confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    confusion = confusion_matrix(y_true, y_pred)
    logger.info('Confusion Matrix\n')
    logger.info(confusion)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

    #importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    logger.info('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

    logger.info('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro')))
    logger.info('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro')))
    logger.info('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro')))

    logger.info('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
    logger.info('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
    logger.info('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

    logger.info('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted')))
    logger.info('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted')))
    logger.info('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    logger.info('\nClassification Report\n')
    logger.info(classification_report(y_true, y_pred, target_names=valid_loader.dataset.classes))

