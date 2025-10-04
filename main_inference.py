import os
os.environ["HF_HOME"] = "./.cache/huggingface"
import json
from pathlib import Path

import torch
import toml
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoImageProcessor, AutoTokenizer, 
    TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
)

from modules.Inference import Inference
from modules.utils import fix_seeds


# 定数
CONFIG_PATH = "./config/inference_config.toml"

   
def inferece():
    # 乱数の固定
    fix_seeds()
    
    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
    
    ## 入出力パス
    result_path = Path(cfg["train_result_path"])
    input_path = Path(cfg["input_path"])
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")
    
    # 学習時のconfigから必要なパラメータを取得
    with open(result_path.joinpath("train_config.toml"), mode="r", encoding="utf-8") as f:
        cfg_t = toml.load(f)
    encoder_name = cfg_t["model"]["encoder_name"]
    decoder_name = cfg_t["model"]["decoder_name"]
    
    # 前処理
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessor(image_processor, tokenizer)
    
    # モデルの定義
    # モデルの定義
    with open(result_path.joinpath("config.json"), mode="r", encoding="utf-8") as f:
        model_config_dict = json.load(f)
        model_config = VisionEncoderDecoderConfig(**model_config_dict)
    model = VisionEncoderDecoderModel(model_config)
    # モデルのコンフィグの設定
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # 作成したモデルの読み込み
    checkpoint = torch.load(result_path.joinpath("model_latest.pth"), map_location=device)
    # checkpoint = torch.load(result_path.joinpath("model_best.pth"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # 推論クラスの定義
    infer = Inference(
        model=model,
        processor=processor,
        device=device,
        output_path=result_path
    )
    
    # 推論するデータの準備
    test_df = pd.read_csv(input_path, encoding="cp932")
    
    # 画像を1枚ずつ推論
    preds = []
    for img_name in tqdm(test_df["img"], desc="inference"):
        # 推論
        pred_str = infer(input_path.parent.joinpath(img_name))
        preds.append(pred_str)
    
    # 検出結果を追加して結果出力
    test_df["pred"] = preds
    test_df.to_csv(result_path.joinpath("inference_result.csv"), encoding="cp932", index=False)
        

if __name__ == "__main__":
    inferece()