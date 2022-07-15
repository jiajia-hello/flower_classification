import torch
from torch import nn
import argparse
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms,datasets
from matplotlib import pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device('cuda')

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x:[b,ch,h,w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        # short cut
        out = self.extra(x) + out
        return out

# flowerClassfication input[3,224,224]
class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        # 3,32,1,1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.blk1 = ResBlk(64, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=2)
        self.blk3 = ResBlk(128, 256, stride=2)
        self.blk4 = ResBlk(256, 512, stride=2)
        self.outlayer = nn.Linear(512, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x,[1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

def main():

    """
    train and test
    :return:
    """
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch",epoch,batchidx,"loss",loss.item())

        # test every epoch
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = pred.eq(label).sum().item()
                total_correct += correct
                total += x.size(0)

            acc = total_correct / total
            print('epoach', epoch, 'test acc:', acc)


