from pathlib import Path

import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from datasets import Dataset as HFDataset

from .augmentation import get_transform


__all__ = ["OCRDataset"]


class OCRDataset(Dataset):
    """画像分類用データセットクラス"""
    
    def __init__(
        self,
        dataset: HFDataset,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        phase: str = "test"
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.phase = phase
        
        # DataAugmentationの準備
        self.tramsform = get_transform(phase)
        
    def __len__(
        self
    ) -> int:
        """Datasetの長さを返す"""
        return len(self.dataset)
    
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
        img, text = self.dataset[index]
        
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
        
        
        

