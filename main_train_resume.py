import os
os.environ["HF_HOME"] = "./.cache/huggingface"
import json
from pathlib import Path
from argparse import ArgumentParser

import torch
# from torch.optim import AdamW
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader
import toml
import pandas as pd
from transformers import (
    AutoImageProcessor, AutoTokenizer, 
    TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
)

from modules.utils import fix_seeds, ProcessTimeManager
from modules.loader import OCRDataset
from modules.trainer import Trainer


# 定数
DATASET_DIR = Path("./dataset")

    
def train(
    resume: str
) -> None:
    # 乱数の固定
    fix_seeds()
    
    # 出力パス
    output_path = Path(resume)
    
    # 途中再開の設定
    config_path = output_path.joinpath("train_config.toml")
    
    # 設定ファイルの読み込み
    with open(config_path, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
    
    ## モデル名
    encoder_name = cfg["model"]["encoder_name"]
    decoder_name = cfg["model"]["decoder_name"]
    
    ## 各種パラメータ
    num_epoches = cfg["parameters"]["num_epoches"]
    batch_size = cfg["parameters"]["batch_size"]
    lr = cfg["optimizer"]["lr"]
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス: {device}")
    
    # Datasetの読み込み
    dataset_df = pd.read_csv(DATASET_DIR.joinpath("annotations.csv"), encoding="cp932")
    dataset_df = dataset_df.sample(frac=1, random_state=42, ignore_index=True)
    
    # 前処理
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessor(image_processor, tokenizer)
    
    # Datasetの作成
    train_dataset = OCRDataset(
        dataset_df.iloc[:-50, :], DATASET_DIR,
        processor, phase="train"
    )
    val_dataset = OCRDataset(
        dataset_df.iloc[-50:, :], DATASET_DIR,
        processor, phase="val"
    )
    print(f"データ分割 train:val = {len(train_dataset)}:{len(val_dataset)}")
    
    # DataLoaderの作成
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                num_workers=0, pin_memory=True)
    
    # 途中保存した重みの読み込み
    checkpoint = torch.load(output_path.joinpath("model_latest.pth"), map_location="cpu")
    
    # モデルの定義
    with open(output_path.joinpath("config.json"), mode="r", encoding="utf-8") as f:
        model_config_dict = json.load(f)
        model_config = VisionEncoderDecoderConfig(**model_config_dict)
    model = VisionEncoderDecoderModel(model_config)
    model.load_state_dict(checkpoint["model"])
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

    # optimizerの定義
    optimizer = RAdamScheduleFree(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Trainerの定義
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        processor=processor,
        device=device,
        output_path=output_path,
        resume_checkpoint=checkpoint
    )
    
    # 学習ループを実行
    start_epoch = checkpoint["epoch"]
    for i in range(start_epoch, num_epoches):
        epoch = i + 1
        
        # 訓練
        train_loss = trainer.train(epoch)
        # 検証
        val_loss, val_cer = trainer.validation(epoch)
        
        # ログの標準出力
        print(f"Epoch:{epoch}")
        print(f"  Train Loss:{train_loss:.4f}")
        print(f"  Validation Loss:{val_loss:.4f}")
        print(f"  Score(Character Error Rate):{val_cer:.4f}")
        
        # 学習の進捗を出力
        trainer.output_learning_curve()
        
        # モデルの重みを保存
        trainer.save_weight()
        # 学習ログを保存
        trainer.output_log()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("resume", type=str)
    args = parser.parse_args()  
    
    with ProcessTimeManager(is_print=True) as pt:
        train(resume=args.resume)
