import torch
import torchvision
import hydra
from omegaconf import OmegaConf

from tqdm import tqdm
import datetime
import time
import logging

import breaching
from collections import OrderedDict
import argparse

import os
from breaching.cases.models.model_preparation import _construct_vision_model
from breaching.cases.models.vgg import VGG

# os.environ["HYDRA_FULL_ERROR"] = "0"
# log = logging.getLogger(__name__)

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
    # add argparser for num_epochs, learning rate, weight decay
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    args = parser.parse_args()


    # model = VGG(
    #                 "VGG11",
    #                 in_channels=3,
    #                 num_classes=100,
    #                 norm="BatchNorm2d",
    #                 nonlin="ReLU",
    #                 head="CIFAR",
    #                 convolution_type="Standard",
    #                 drop_rate=0,
    #                 classical_weight_init=True,
    #                 use_bias=True,
    #             )

    model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)),
                        ("relu0", torch.nn.LeakyReLU()),
                        ("conv2", torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                        ("relu1", torch.nn.LeakyReLU()),
                        ("conv3", torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                        ("relu2", torch.nn.LeakyReLU()),
                        ("conv4", torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                        ("relu3", torch.nn.LeakyReLU()),
                        ("flatt", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(16384, 12544)),
                        ("relu4", torch.nn.LeakyReLU()),
                        ("linear1", torch.nn.Linear(12544, 100, bias=True)),
                    ]
                )
            )
    path = f"cnn_beyond_bias_lr{args.lr}_wd{args.weight_decay}.pth"

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
        plt.savefig(f"train_losses_lr{args.lr}_wd{args.wd}.png")

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

# def train(cfg):
#     """This function controls the central routine."""
#     total_time = time.time()  # Rough time measurements here
#     setup = breaching.utils.system_startup(0, 1, cfg)
#     model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)
#     # load cifar100 data into train_loader and test_loader, from torch
#     train_loader = torch.utils.data.DataLoader(
#         dataset=torchvision.datasets.CIFAR100(
#             root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()
#         ),
#         batch_size=64,
#         shuffle=True
#     )


#     # train_loader, test_loader = breaching.cases.construct_dataloader(cfg.case, setup)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

#     #move model to device
#     model.to(device)

#     # train model
#     model.train()
#     for i in range(NUM_EPOCHS):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             # move data to device
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = loss_fn(output, target)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             if batch_idx % 100 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                    i, batch_idx * len(data), len(train_loader.dataset),
#                    100. * batch_idx / len(train_loader), loss.item()))
    
#     # save model
#     torch.save(model.state_dict(), 'model.pth')
#     return model

# def test(cfg, model):
#     """This function controls the central routine."""
#     total_time = time.time()  # Rough time measurements here
#     setup = breaching.utils.system_startup(0, 1, cfg)
#     model, loss_fn = _construct_vision_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)
#     # load test data into test_loader, from torch
#     test_loader = torch.utils.data.DataLoader(
#         dataset=torchvision.datasets.CIFAR100(
#             root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()
#         ),
#         batch_size=64,
#         shuffle=True
#     )
    

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

#     #move model to device
#     model.to(device)

#     # test model
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             # move data to device
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += loss_fn(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
    
#     test_loss /= len(test_loader.dataset)
#     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))
    

# @hydra.main(config_path="breaching/config", config_name="my_cfg", version_base="1.1")
# def main_launcher(cfg):
#     model = train(cfg)
#     test(cfg, model)


# if __name__ == "__main__":
#     main_launcher()