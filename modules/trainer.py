from pathlib import Path

import torch
from torch.optim import Optimizer
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# エラー対処
matplotlib.use('Agg') 


class Trainer:
    """訓練を実行するクラス"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        optimizer: Optimizer | RAdamScheduleFree,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        processor: TrOCRProcessor,
        device: torch.device | str,
        output_path: str | Path,
        resume_checkpoint: dict | None = None
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.processor = processor
        self.device = device
        self.output_path = Path(output_path)
        
        # 学習の準備
        self.model.to(device)
        
        # 評価指標の準備
        self.cer_metric = evaluate.load("cer")
        
        if resume_checkpoint is None:
            # 新規モデル作成
            self.log = {"epoch": [], "train_loss": [], "val_loss": [], "val_cer": []}
            self.best_metric = np.inf
        else:
            # 途中再開
            log_df = pd.read_csv(self.output_path.joinpath("training_log.csv"))
            self.log = log_df.to_dict(orient="list")
            self.best_metric = log_df["val_loss"].min()
            
            # 読み込んだoptimizerのstateをGPUに渡す
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    
    def compute_cer(
        self,
        pred_ids, 
        label_ids
    ) -> float:
        """Character Error Rateを計算
        """
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        try:
            cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        except:
            cer = 0

        return cer
    
    def train(
        self,
        epoch: int
    ) -> float:
        """訓練のループを実行
        """
        self.model.train()
        self.optimizer.train()
        iter_train_loss = []
        
        for batch in tqdm(self.train_dataloader, desc="train"):                   
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            
            self.optimizer.zero_grad()

            outputs = self.model(**batch)

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            
            iter_train_loss.append(loss.item())
            
        # 1epochの平均lossを計算
        epoch_train_loss = np.mean(iter_train_loss)
        self.log["train_loss"].append(epoch_train_loss)
        self.log["epoch"].append(epoch)
            
        return epoch_train_loss
    
    def validation(
        self,
        epoch: int
    ) -> float:
        """検証のループを実行
        """
        self.model.eval()
        self.optimizer.eval()
        iter_val_loss = []
        iter_val_cer = []
        
        for batch in tqdm(self.val_dataloader, desc="val"):
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
                gen_outputs = self.model.generate(batch["pixel_values"])
            
            iter_val_loss.append(outputs.loss.item())
            cer = self.compute_cer(pred_ids=gen_outputs, label_ids=batch["labels"])
            iter_val_cer.append(cer)
        
        # 1epochの平均lossを計算
        epoch_val_loss = np.mean(iter_val_loss)
        self.log["val_loss"].append(epoch_val_loss)
        # 1epochの平均Character Error Rateを計算
        epoch_val_cer = np.mean(iter_val_cer)
        self.log["val_cer"].append(epoch_val_cer)
        
        # 最良のLossを判定
        if self.best_metric > epoch_val_loss:
            self.best_metric = epoch_val_loss
            self.save_weight(model_name=f"model_best.pth")
            
        return epoch_val_loss, epoch_val_cer, 
        
    def save_weight(
        self,
        model_name: str = f"model_latest.pth"
    ) -> None:
        """モデルの重みを保存
        """        
        checkpoint = {
            "epoch": len(self.log["epoch"]),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.output_path.joinpath(model_name))
        print(f"model saved: {model_name}")
    
    def output_learning_curve(
        self,
    ) -> None:
        """学習曲線の出力
        """
        epoch = len(self.log["epoch"]) # 現在までのエポック数を取得
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(f"Loss (Epoch:{epoch})")
        ax1.plot(self.log["epoch"], self.log["train_loss"], c='red', label='train')
        ax1.plot(self.log["epoch"], self.log["val_loss"], c='blue', label='val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"Character Error Rate (Epoch:{epoch})")
        ax2.plot(self.log["epoch"], self.log["val_cer"], c='blue', label='val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CER')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("learning_curve.png"))
        
        plt.close()
        
    def output_log(
        self,
    ) -> None:
        """ログファイルの出力
        """
        # DataFrameに変換してcsvで出力
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(self.output_path.joinpath("training_log.csv"),
                      encoding='utf-8-sig', index=False)
    
