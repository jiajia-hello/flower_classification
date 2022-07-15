import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from flowerData import flower
from torch import optim
import torchvision
# from utils import evaluate
from resnet_torch import ResNet18
from utils import plot_curve
import os
import argparse

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--data_dir", default="/gemini/data-1")
parser.add_argument("--train_dir", default="/gemini/code/output")
args = parser.parse_args()

train_db = flower(os.path.join(args.data_dir,'flower_photos'), 224,'train')
val_db = flower(os.path.join(args.data_dir,'flower_photos'), 224,'val')
test_db = flower(os.path.join(args.data_dir,'flower_photos'), 224,'test')
train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True,num_workers=4)
val_loader = DataLoader(val_db, batch_size=args.batch_size, num_workers=2)
test_loader = DataLoader(test_db, batch_size=args.batch_size, num_workers=2)
epochs = args.num_epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    correct=0
    total = len(loader.dataset)
    model.eval()
    for x, y in loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def train():
    print(os.path.join(args.data_dir,'flower_photos'))
    model = ResNet18(5).to(device)
    acc_list=[]
    optimizer = optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=0.9)
    criteon = nn.CrossEntropyLoss()

    global_step = 0
    best_acc, best_epoch = 0, 0
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            model.train()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(step, "loss:", loss.item())

            global_step += 1

        val_acc = evaluate(model, val_loader)
        acc_list.append(val_acc)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), args.train_dir+"/flower.mdl")

        print('epochs', epoch,'best acc:', best_acc, 'best epoch', best_epoch)

    model.load_state_dict(torch.load(args.train_dir+"/flower.mdl"))
    model.eval()
    test_acc = evaluate(model, test_loader)
    plot_curve(acc_list,args.train_dir)
    print('test acc', test_acc)

if __name__ == '__main__':
    train()