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
import os

INPUT = conf.source_image
output_model = conf.model
metrics_file = "data/eval.txt"
test_csv = conf.test_csv

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)


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
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_datasets = mnist_dataset(test_csv, test_transform)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True, **kwargs)

    model = torch.load(output_model)
    
    accuracy = test(model, device, test_loader)
    with open(metrics_file, 'w') as fd:
        fd.write('Accuracy: {:4f}\n'.format(accuracy))


if __name__ == '__main__':
    main()