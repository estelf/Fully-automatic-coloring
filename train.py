import torch
import torchvision
import glob
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import imutils
import tqdm


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

        return self.linedraw(img), img

    def linedraw(self, x):
        # 3x3カーネルで膨張1回（膨張はMaxPoolと同じ）
        dilated = torch.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # 膨張の前後でL1の差分を取る
        diff = torch.abs(x - dilated)
        # ネガポジ反転
        x = 1.0 - diff
        return x


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


def cvfunc(img):
    img = img[:, :, ::-1]
    img = resize_img(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return PIL.Image.fromarray(img)


def show_tensor(input_image_tensor, f):
    img = input_image_tensor.to("cpu").detach().numpy().transpose(1, 2, 0)
    # img = img.astype(np.uint8)[0,0,:,:]
    plt.imshow(img)
    plt.savefig(f"test_{f}.png")
    # plt.show()


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
train_loader = torch.utils.data.DataLoader(linedataset, batch_size=8, shuffle=True)

# モデル定義
model = UNet(3, 3)
# デバイスモデル設定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 損失関数,分類問題のためクロスエントロピー損失関数を利用
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

EPOCHS = 300
a, b = next(iter(train_loader))
# show_tensor(torchvision.utils.make_grid(torch.cat([a, b])))

for epoch in range(EPOCHS):
    model.train()  # モデルを学習モードにしてGPUに転送（重要）
    model.to(device)

    train_loss = 0
    total_t = 0

    for batch in train_loader:
        optimizer.zero_grad()  # 必須
        # image ,label = batch #(batch_size, channel, size, size)
        image, label = batch

        image = image.to(device)
        label = label.to(device)  # dtype=torch.long

        preds = model(image)  # (batch_size, num_class)
        # print(preds.dtype)
        loss = criterion(preds, label)  # 必須
        loss.backward()  # 必須
        optimizer.step()  # 必須

        train_loss += loss.item()

    model.eval()  # 評価モードにする

    with torch.no_grad():  # 必須
        for batch in tqdm.tqdm(train_loader):
            image, label = batch  # (batch_size, channel, size, size)

            image = image.to(device)
            label = label.to(device)
            preds = model(image)
        show_tensor(torchvision.utils.make_grid(torch.cat([image, label, preds])), epoch)
