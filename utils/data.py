import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def data_loader():
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader