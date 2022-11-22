import torch
import torchvision
import glob
import cv2
import PIL
import numpy as np

"""
using U-net

"""


class dataset_faces(torch.utils.data.Dataset):
    def __init__(self, fileName, transform=None):
        self.fileList = glob.glob(fileName)
        self.transform = transform

    def __len__(self):
        output = len(self.fileList)
        return output

    def __getitem__(self, idx):
        img = cv2.imread(self.fileList[idx])

        return self.transform(img), img


def cvfunc(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return PIL.Image.fromarray(img)


"""****000000****"""  # オーグメンテーションをいろいろ追加してみてもよいと思います！
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(cvfunc),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ]
)
