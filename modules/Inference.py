import os
os.environ["HF_HOME"] = "./.cache/huggingface"
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

from modules.utils import imread_jpn


class Inference:
    """推論を実行するクラス"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.output_path = Path(output_path).joinpath("inference")
        
        # 学習の準備
        self.model.to(device)
        self.model.eval()
        
        # 結果保存
        self.img_name_list = []
        self.pred_lbls = []
        
    def __call__(
        self,
        img_path: Path | str
    ) -> str:
        """1画像で推論の処理を実行
        """
        # 画像の読み込み
        img = Image.open(img_path).convert("RGB")
        
        # 画像の前処理
        input_img = self.processor(images=img, return_tensors="pt").pixel_values[0]
        input_img = input_img.to(self.device)
        input_img = input_img.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model.generate(input_img)
        
        pred_str = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        # print("result:", pred_str)
        
        return pred_str
    
