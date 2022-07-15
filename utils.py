import torch
from torch import nn
from matplotlib import pyplot as plt

def plot_curve(data,filedir):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.savefig(filedir+"/acc.jpg")
    plt.show()


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{} : {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

def evaluate(model, loader):
    correct=0
    total = len(loader.dataset)

    for x, y in enumerate(loader):
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def denormlize(self, x_hat):
#     x_hat = (x-mean)/std
#     x = x_hat*std+mean
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    x = x_hat*std +mean

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # prod 返回给定维度的积
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return  x.view(-1, shape)

