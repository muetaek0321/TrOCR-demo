from pathlib import Path

import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor

from .augmentation import get_transform


__all__ = ["OCRDataset"]


class OCRDataset(Dataset):
    """画像分類用データセットクラス"""
    
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        dataset_dir: Path | str,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        phase: str = "test"
    ) -> None:
        self.dataset_df = dataset_df
        self.dataset_dir = Path(dataset_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.phase = phase
        
        # DataAugmentationの準備
        self.tramsform = get_transform(phase)
        
        # 学習時のみデータセット追加
        if phase == "train":
            self.load_additional_dataset()
        
    def load_additional_dataset(
        self
    ) -> None:
        """データセットの追加読み込み
        """
        add_df = pd.read_csv("./add_dataset/annotations.csv", encoding="utf-8-sig")
        self.dataset_df = pd.concat([self.dataset_df, add_df])
        
    def __len__(
        self
    ) -> int:
        """Datasetの長さを返す"""
        return len(self.dataset_df)
    
    def __getitem__(
        self, 
        index: int
    ) -> tuple[np.ndarray, dict]:
        """index指定でデータを返す
        
        Args:
            index (int): Datasetのインデックス
            
        Returns:
            np.ndarray: 画像
            list: アノテーション
        """
        # 画像とラベルを取得
        data = self.dataset_df.iloc[index, :]
        img_name, text = data["img"], data["text"]
        
        # 画像読み込み
        img = Image.open(self.dataset_dir.joinpath(img_name)).convert("RGB")
        
        # DataAugmentationの適用
        img = self.tramsform(image=np.array(img))["image"]
            
        # 前処理の適用
        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values
        labels = self.processor.tokenizer(
            text, padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        processed_data = {
            "pixel_values": pixel_values.squeeze(), 
            "labels": torch.tensor(labels)
        }
        
        return processed_data
        
        
        

