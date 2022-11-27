# import 類
import glob

import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from PIL import Image, ImageFile
from unet import UNet
import imutils

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用デバイス : {device}")

# モデル定義  timmを使ってるが自作する場合もある
# model=timm.create_model("efficientnet_b0",pretrained=False,num_classes=2)#channel変更 in_chans=1

model = UNet(3, 3)


class dataset_faces(torch.utils.data.Dataset):
    def __init__(self, fileName, transform_main=None):
        self.fileList = glob.glob(fileName + "//*.png")
        self.transform_main = transform_main

    def __len__(self):
        output = len(self.fileList)
        return output

    def __getitem__(self, idx):
        img = cv2.imread(self.fileList[idx])
        img = self.transform_main(img)

        return self.linedraw(img), img

    def linedraw(self, x):
        # 3x3カーネルで膨張1回（膨張はMaxPoolと同じ）
        dilated = torch.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # 膨張の前後でL1の差分を取る
        diff = torch.abs(x - dilated)
        # ネガポジ反転
        x = 1.0 - diff
        return x


target = input("データのあるフォルダ名>>")


def cvfunc(img):
    img = img[:, :, ::-1]
    img = resize_img(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return Image.fromarray(img)


def resize_img(img):
    """
    画像をpaddingしながら256x256にする
    """
    height, width, _ = img.shape  # 画像の縦横サイズを取得
    diffsize = abs(height - width)
    padding_half = int(diffsize / 2)

    # 縦長画像→幅を拡張する
    if height > width:
        padding_img = cv2.copyMakeBorder(
            img, 0, 0, padding_half, height - (width + padding_half), cv2.BORDER_CONSTANT, (0, 0, 0)
        )
    # 横長画像→高さを拡張する
    elif width > height:
        padding_img = cv2.copyMakeBorder(
            img, padding_half, width - (height + padding_half), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
        )
    else:
        padding_img = img
    # 最後にリサイズ
    return imutils.resize(padding_img, width=64)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(cvfunc),
        torchvision.transforms.ToTensor(),
    ]
)

myv_dataset = dataset_faces(target, transform)
myv_loader = torch.utils.data.DataLoader(myv_dataset, batch_size=1)  # , shuffle = True)


def show_tensor(input_image_tensor):
    img = input_image_tensor.to("cpu").detach().numpy().transpose(1, 2, 0)
    # img = img.astype(np.uint8)[0,0,:,:]
    plt.imshow(img)

    plt.show()


# モデル読み込み
# model.load_state_dict(torch.load("model.pth"))
model = torch.load("model.pth")
model.eval()  # 必須
model.to(device)


# get batch of images from the test DataLoader
a, b = next(iter(myv_loader))
# print(a.shape, b.shape)
# show_tensor(torchvision.utils.make_grid(torch.cat([a, b])))


with torch.no_grad():
    for i in myv_loader:
        images, filename = i
        images = images.to(device)
        filename = filename.to(device)
        outputs = model(images)
        # print(images)

        show_tensor(torchvision.utils.make_grid(torch.cat([filename, images, outputs])))
