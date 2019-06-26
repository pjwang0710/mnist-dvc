from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
import conf
import sys
import os

INPUT = conf.source_image
train_csv = conf.train_csv
test_csv = conf.test_csv
output_model = conf.model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class mnist_dataset(Dataset):
    def __init__(self, csv_file, transform):
        df = pd.read_csv(csv_file)
        self.x_train = df['path']
        self.y_train = np.array(df['label'].astype(np.int32))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(INPUT, self.x_train[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train.index)


def main():
    use_cuda = False
    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('load data')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_datasets = mnist_dataset(train_csv, train_transform)
    test_datasets = mnist_dataset(test_csv, test_transform)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True, **kwargs)

    print('train size:{}'.format(len(train_datasets)))
    print('test size:{}'.format(len(test_datasets)))

    print('load model')
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2):
        train(model, criterion, device, train_loader, optimizer, epoch)
        test(model, criterion, device, test_loader)

    if (True):
        print('save model in {}'.format(output_model))
        torch.save(model, output_model)


if __name__ == '__main__':
    main()
