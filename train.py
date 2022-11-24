import torch
import torchvision
import glob
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import unet


class dataset_faces(torch.utils.data.Dataset):
    def __init__(self, fileName, transform_main=None):
        self.fileList = glob.glob(fileName + "*")
        self.transform_main = transform_main

    def __len__(self):
        output = len(self.fileList)
        return output

    def __getitem__(self, idx):
        img = cv2.imread(self.fileList[idx])
        img = self.transform_main(img)
        return img, self.linedraw(img)

    def linedraw(self, x):
        # 3x3カーネルで膨張1回（膨張はMaxPoolと同じ）
        dilated = torch.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # 膨張の前後でL1の差分を取る
        diff = torch.abs(x - dilated)
        # ネガポジ反転
        x = 1.0 - diff
        return x


def cvfunc(img):
    img = img[:, :, ::-1]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return PIL.Image.fromarray(img)


def show_tensor(input_image_tensor):
    img = input_image_tensor.to("cpu").detach().numpy().transpose(1, 2, 0)
    # img = img.astype(np.uint8)[0,0,:,:]
    plt.imshow(img)
    plt.show()


"""****000000****"""  # オーグメンテーションをいろいろ追加してみてもよいと思います！
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(cvfunc),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ]
)

linedataset = dataset_faces("image\\", transform)
show_tensor(linedataset[0][0] - linedataset[0][1])
# show_tensor()
