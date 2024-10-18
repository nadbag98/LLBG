import torch
import torchvision
import hydra
from omegaconf import OmegaConf

from tqdm import tqdm
import datetime
import time
import logging

import src
from collections import OrderedDict
import argparse

import os
from src.cases.models.model_preparation import _construct_vision_model
from src.cases.models.vgg import VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, criterion, device, num_epochs=100):
    model.train()
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        curr_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if data.shape[0] == 0:
                continue
            data = data.to(device)
            target = target.to(device)
            # add new dimension for regression
            # target = target.unsqueeze(1)
            
            optimizer.zero_grad()
            # from ipdb import set_trace; set_trace()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            curr_loss += loss.item()
        curr_loss /= len(train_loader.dataset)
        train_losses.append(curr_loss)
        print(f"epoch: {epoch}, loss: {curr_loss}")
    # calculate final accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device).float()
            target = target.unsqueeze(1)
            
            output = model(data)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target.squeeze()).sum().item()
    train_acc = correct / len(train_loader.dataset)
    
    return train_losses, train_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            # target = target.unsqueeze(1)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            # round each output to 0 or 1
            # pred = torch.round(output)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    return test_loss, test_acc

def main():
    # assumes model being trained is a VGG model, as this is
    # the only cases needed for our experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="VGG11")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    args = parser.parse_args()

    model = VGG(
                    args.model_name,
                    in_channels=3,
                    num_classes=100,
                    norm="BatchNorm2d",
                    nonlin="ReLU",
                    head="CIFAR",
                    convolution_type="Standard",
                    drop_rate=0,
                    classical_weight_init=True,
                    use_bias=True,
                )

    path = f"{args.model_name.lower()}.pth"

    # try loading weights from pretrained model
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        model.load_state_dict(torch.load(path))
    except:
        # load cifar100 data into train_loader and test_loader, from torch
        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()
            ),
            batch_size=64,
            shuffle=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_losses, train_acc = train(model, train_loader, optimizer, criterion, device, num_epochs=args.num_epochs)
        print(f"train loss: {train_losses[-1]}, train acc: {train_acc}")
        # plot train losses and save plot to file
        import matplotlib.pyplot as plt
        plt.plot(train_losses)
        plt.savefig(f"train_losses_lr{args.lr}_wd{args.weight_decay}.png")

        # save model
        torch.save(model.state_dict(), path)

    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=64,
        shuffle=True
    )
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"test loss: {test_loss}, test acc: {test_acc}")

if __name__ == "__main__":
    main()