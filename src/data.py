import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple

def setup_cifar10(cifar_path:str, batch_size:int, num_workers:int, download:bool=False) -> Tuple[DataLoader, DataLoader]:
    """ Create train and valid DataLoader for CIFAR10 Dataset
    """
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.247, 0.243, 0.261]
    )


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),  # Upsample
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample
        transforms.ToTensor(),
        normalizer,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        cifar_path, train=True, transform=transform_train, download=download
    )

    valid_test_dataset = torchvision.datasets.CIFAR10(
        cifar_path, train=False, transform=transform_valid, download=False
    )

    # Split valid_test_dataset into two
    dataset_size = len(valid_test_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # valid_indices = indices[:dataset_size//2]
    # test_indices = indices[dataset_size//2:]

    # valid_sampler = SubsetRandomSampler(valid_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_test_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, valid_loader





def setup_mnist(mnist_path:str, batch_size:int, num_workers:int, download:bool=False) -> Tuple[DataLoader, DataLoader]:
    """ Create train and valid DataLoader for MNIST Dataset
    """
    normalizer = transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),  # Upsample
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample
        transforms.ToTensor(),
        normalizer,
    ])

    train_dataset = torchvision.datasets.MNIST(
        mnist_path, train=True, transform=transform_train, download=download
    )

    valid_test_dataset = torchvision.datasets.MNIST(
        mnist_path, train=False, transform=transform_valid, download=False
    )

    # Split valid_test_dataset into two
    dataset_size = len(valid_test_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # valid_indices = indices[:dataset_size//2]
    # test_indices = indices[dataset_size//2:]

    # valid_sampler = SubsetRandomSampler(valid_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, valid_loader