import torch
import time

import torch
import os, glob
import random,csv

import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class flower(Dataset):
    def __init__(self, root, resize, mode):
        super(flower, self).__init__()

        self.resize = resize
        self.root = root

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir((os.path.join(root, name))):
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.images, self.labels = self.load_csv('flower.csv')

        if mode == 'train':
            self.images = self.images[0:int(0.6 * len(self.images))]
            self.labels = self.labels[0:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            image_files = []
            for name in self.name2label.keys():
                # image_files += glob.glob(os.path.join(self.root, name, '*.png'))
                image_files += glob.glob(os.path.join(self.root, name, '*.jpg'))
                # image_files += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # print(len(image_files), image_files)

            random.shuffle(image_files)
            with open(os.path.join("/gemini/code/code/", filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in image_files:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in image_files:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])

        image_files, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                image_files.append(img)
                labels.append(label)

        assert len(image_files) == len(labels)
        return image_files, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print('idx',idx)
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label


def count(root):
    num = 0
    for _ in os.listdir(root):
        num += 1
    print(num)
