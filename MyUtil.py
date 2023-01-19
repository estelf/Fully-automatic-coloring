import datetime
import io
import random
import sys

import albumentations as A
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import sklearn.metrics
import torch


# アルバメンテーション用リストをtorchV.lamda用に機能拡張
class AlbTransToTVLambda:
    def __init__(self, list) -> None:
        self.list = list
        self.tem = A.Compose(list)

    def __call__(self, img):
        return PIL.Image.fromarray(self.tem(image=img)["image"])

    def __getitem__(self, i):
        return self.list[i]


# 学習自動停止機構
class early_stopping:
    def __init__(self, user_count):
        self.pre_loss = sys.maxsize
        self.user_count = user_count
        self.conter = 0

    def check(self, loss):
        if self.pre_loss > loss:
            self.pre_loss = loss
            self.conter = 0
        else:
            self.conter = self.conter + 1

        if self.conter >= self.user_count:
            print(f"{self.user_count}回更新しなかったため停止しました")
            raise KeyboardInterrupt


def make_confusion_matrix_img(y_test, pred):
    mem = io.BytesIO()
    cm = sklearn.metrics.confusion_matrix(y_test, pred)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap="Blues")
    plt.savefig(mem, format="png")
    mem.seek(0)
    img = cv2.imdecode(np.asarray(bytearray(mem.read())), cv2.IMREAD_COLOR)
    return img


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
            img, 0, 0, padding_half, height - (width + padding_half), cv2.BORDER_CONSTANT, (255, 255, 255)
        )
    # 横長画像→高さを拡張する
    elif width > height:
        padding_img = cv2.copyMakeBorder(
            img, padding_half, width - (height + padding_half), 0, 0, cv2.BORDER_CONSTANT, (255, 255, 255)
        )
    else:
        padding_img = img
    # 最後にリサイズ
    return imutils.resize(padding_img, width=64)


def save_tensor(input_image_tensor, f):
    img = input_image_tensor.to("cpu").detach().numpy().transpose(1, 2, 0)
    cv2.imwrite(f"res\\test_{f}.png", img[:, :, ::-1])


def show_tensor(input_image_tensor):
    # writer.add_image(tag="img",img_tensor=input_image_tensor,global_step=f)
    img = input_image_tensor.to("cpu").detach().numpy().transpose(1, 2, 0)

    plt.imshow((img * 255).astype(np.uint8))
    # plt.savefig(f"res\\test_{f}.png")
    # plt.show()


def torch_fix_seed(seed):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class Report:
    def __init__(self, params) -> None:
        device = params["device"]
        model = params["model"]
        criterion = params["criterion"]
        optimizer = params["optimizer"]
        scheduler = params["scheduler"]
        EPOCHS = params["EPOCHS"]
        writer = params["writer"]
        split_rate = params["split_rate"]
        auglist = params["auglist"]
        linedataset = params["linedataset"]
        early_stopping_count = params["early_stopping_count"]

        now = datetime.datetime.now()
        today = now.strftime("%Y年%m月%d日(%A) %H:%M:%S")
        aug_table = ""
        aug_table_md = ""

        for i in auglist:
            aug_table = (
                aug_table
                + f"""<tr><td colspan="2" style="border: solid 1px #adb3c1;">
                <center>{i.__class__.__name__}</center>
                </td></tr>"""
            )
        for i in auglist:
            aug_table_md = aug_table_md + f"""|{i.__class__.__name__}|  \n"""
        self.RowHTML = f"""
    <h1><center>～機械学習 学習前レポート～</center></h1>
        <h3><center>{today}</center></h3>
        <div style="display: flex;
        flex-wrap: wrap;
        row-gap: 2em;
        column-gap: 10px;">
        <div >
        <table style="border-collapse: collapse; background-color: #f0f2f7; color: #333;">
        <tbody>
            <tr>
    <td colspan="2" style="border: solid 1px #adb3c1;background-color: #d0ebfd;"><center><b>学習全体レポート</b></center></td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">デバイス名</td>
                <td style="border: solid 1px #adb3c1;">{device}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">学習ネットワーク名</td>
                <td style="border: solid 1px #adb3c1;">{model.__class__.__name__}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">損失関数</td>
                <td style="border: solid 1px #adb3c1;">{criterion.__class__.__name__}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">最適化関数</td>
                <td style="border: solid 1px #adb3c1;">{optimizer.__class__.__name__}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">スケジューラ</td>
                <td style="border: solid 1px #adb3c1;">{scheduler.__class__.__name__}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">総エポック数</td>
                <td style="border: solid 1px #adb3c1;">{EPOCHS}</td>
            </tr>

            <tr>
                <td style="border: solid 1px #adb3c1;">早期打ち止め回数</td>
                <td style="border: solid 1px #adb3c1;">{early_stopping_count}</td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">tensorBoardパス</td>
                <td style="border: solid 1px #adb3c1;">{writer.log_dir}</td>
            </tr>
        </tbody>
    </table>
    </div>

        <div >
        <table style="border-collapse: collapse; background-color: #f0f2f7; color: #333;">
        <tbody>
            <tr>
    <td colspan="2" style="border: solid 1px #adb3c1;background-color: #d0ebfd;"><center><b>最適化関数レポート</b></center></td>
    </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">最適化関数</td>
                <td style="border: solid 1px #adb3c1;">{optimizer.__class__.__name__} </td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;"> 学習率</td>
                <td style="border: solid 1px #adb3c1;">{optimizer.defaults["lr"]}</td>
            </tr>
        </tbody>
    </table></div>

        <div >
        <table style="border-collapse: collapse; background-color: #f0f2f7; color: #333;">
        <tbody>
            <tr>
<td colspan="2" style="border: solid 1px #adb3c1;background-color: #d0ebfd;"><center><b>データセットレポート</b></center></td>
    </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">データ総数</td>
                <td style="border: solid 1px #adb3c1;">{len(linedataset)} </td>
            </tr>
            <tr>
                <td style="border: solid 1px #adb3c1;">データパス</td>
                <td style="border: solid 1px #adb3c1;">{linedataset.fileName}</td>
            </tr>

            <tr>
                <td style="border: solid 1px #adb3c1;">分割率</td>
                <td style="border: solid 1px #adb3c1;">{split_rate}</td>
            </tr>

        </tbody>
    </table>
    </div>


        <div >
        <table style="border-collapse: collapse; background-color: #f0f2f7; color: #333;">
        <tbody>
            <tr>
    <td colspan="2" style="border: solid 1px #adb3c1;background-color: #d0ebfd;"><center><b>登録データ拡張設定</b></center></td>
    </tr>
    {aug_table}
        </tbody>
    </table>
    </div>
    </div>
        """

        self.RowMD = f"""
# ～機械学習 学習前レポート～
### {today}
|**学習全体レポート**||
|:----:||
|デバイス名|{device}|
|学習ネットワーク名|{model.__class__.__name__}|
|損失関数|{criterion.__class__.__name__}|
|最適化関数|{optimizer.__class__.__name__}|
|スケジューラ|{scheduler.__class__.__name__}|
|総エポック数|{EPOCHS}|
|早期打ち止め回数|{early_stopping_count}|
|tensorBoardパス|{writer.log_dir}|

|**最適化関数レポート**||
|:----:||
|最適化関数|{optimizer.__class__.__name__}|
| 学習率|{optimizer.defaults["lr"]}|

|**データセットレポート**||
|:----:||
|データ総数|{len(linedataset)}|
|データパス|{linedataset.fileName}|

|**登録データ拡張設定**|
|:----|
{aug_table_md}
        """

    def display(self):
        from IPython.display import HTML, display

        display(HTML(self.RowHTML))
