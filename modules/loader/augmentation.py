import random
from pathlib import Path

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class SafeRotate(A.ImageOnlyTransform):
    """枠内に収まるように画像を回転させる処理"""
    
    def __init__(
        self, 
        angle: tuple[int, int] = (-45, 45),  
        border_value: tuple[int, int, int] = (0, 0, 0), 
        p: float = 0.5
    ) -> None:
        super(SafeRotate, self).__init__(p=p)
        self.angle = angle
        self.border_value = border_value
        
    def apply(
        self, 
        img: np.ndarray, 
        **params
    ) -> np.ndarray:
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, np.random.uniform(*self.angle), 1.0)
        
        # 新しい画像サイズを計算
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 移動補正
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 回転
        return cv2.warpAffine(
            img, 
            M, (new_w, new_h), borderValue=self.border_value
        )


class RandomPad(A.ImageOnlyTransform):
    """指定した範囲でランダムにパディングを適用する処理"""
    
    def __init__(
        self, 
        min_pad: int = 0, 
        max_pad: int = 10, 
        border_value: tuple[int, int, int] = (0, 0, 0), 
        p: float = 0.5
    ) -> None:
        super(RandomPad, self).__init__(p=p)
        self.min_pad = min_pad
        self.max_pad = max_pad
        self.border_value = border_value

    def apply(
        self, 
        img: np.ndarray, 
        **params
    ) -> np.ndarray:
        pad_top = random.randint(self.min_pad, self.max_pad)
        pad_bottom = random.randint(self.min_pad, self.max_pad)
        pad_left = random.randint(self.min_pad, self.max_pad)
        pad_right = random.randint(self.min_pad, self.max_pad)
        return cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=self.border_value
        )


class ResizeForTrOCRAutoOrient(A.ImageOnlyTransform):
    """高さ64, 幅3072のキャンバスに収める前処理"""
    
    def __init__(
        self, 
        height: int = 64,
        max_width: int = 3072,
        p: float = 1.0
    ) -> None:
        super(ResizeForTrOCRAutoOrient, self).__init__(p=p)
        self.height = height
        self.max_width = max_width

    def apply(
        self, 
        img: np.ndarray, 
        **params
    ) -> np.ndarray:
        h, w = img.shape[:2]

        # 縦長の場合は90度回転して横長に
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = img.shape[:2]

        new_w = int(w * (self.height / h))
        new_w = min(new_w, self.max_width)

        resized = cv2.resize(img, (new_w, self.height), interpolation=cv2.INTER_AREA)

        canvas = np.ones((self.height, self.max_width, 3), dtype=np.uint8) * 255
        canvas[:, :new_w] = resized
        
        return canvas


def get_transform(
    phase: str
) -> A.Compose:
    """DataAugmentaionの設定
    """
    
    if phase == "train":
        aug_list = [
            A.OneOf([
                A.HueSaturationValue(),
                A.RGBShift(),
                A.InvertImg()
            ]),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(p=0.2),
            A.ImageCompression(p=0.3),
            SafeRotate(),
            RandomPad(),
            A.PadIfNeeded(224, 224, fill=0),
            A.Resize(224, 224),
            # A.Normalize(mean=0.5, std=0.5),
            # ToTensorV2()
        ]
    else:
        aug_list = [
            A.PadIfNeeded(224, 224, fill=0),
            A.Resize(224, 224),
            # A.Normalize(mean=0.5, std=0.5),
            # ToTensorV2()
        ]
        
    return A.Compose(aug_list)
    


if __name__ == "__main__":
    trans = get_transform("train")
    
    img_path_list = list(Path("../../dataset/").glob("*.png"))
    random.shuffle(img_path_list)
    for i, p in enumerate(img_path_list):
        if i > 10:
            break
        
        img = Image.open(p).convert("RGB")
        img = np.array(img)
        
        img = trans(image=img)['image']
        
        plt.imshow(img)
        plt.show()

