# TrOCR

## ※Github Copilotにより作成

## プロジェクト概要
このプロジェクトは、MicrosoftのTrOCRモデルを使用してOCR（光学文字認識）タスクを実行するためのものです。以下の機能を提供します：
- モデルのトレーニング
- 推論（Inference）

## ディレクトリ構造
```
LICENSE                 # ライセンスファイル
main_inference.py       # 推論スクリプト
main_train_resume.py    # トレーニング再開スクリプト
main_train.py           # トレーニングスクリプト
pyproject.toml          # プロジェクト設定ファイル
README.md               # プロジェクト概要（このファイル）

add_dataset/            # 追加データセット
  annotations.csv       # アノテーションファイル

config/                 # 設定ファイル
  inference_config.toml # 推論用設定ファイル
  train_config.toml     # トレーニング用設定ファイル

dataset/                # データセット
  annotations.csv       # アノテーションファイル

modules/                # モジュール
  inference.py          # 推論ロジック
  trainer.py            # トレーニングロジック
  loader/               # データローダー
    augmentation.py     # データ拡張
    dataset.py          # データセットロジック
  utils/                # ユーティリティ
    cv2_japanese.py     # 日本語対応のOpenCVユーティリティ
    date_str.py         # 日付文字列処理
    fix_seed.py         # 乱数シード固定
    process_time.py     # 処理時間計測

results/                # 結果保存ディレクトリ
  TrOCR_finetune/       # ファインチューニング結果
    config.json         # 設定ファイル
    inference_result.csv # 推論結果
    model_best.pth      # 最良モデル
    model_latest.pth    # 最新モデル
    train_config.toml   # トレーニング設定
    training_log.csv    # トレーニングログ
    予測結果_保管/      # 推論結果のバックアップ

test_data/              # テストデータ
  annotations.csv       # アノテーションファイル
```

## 必要な依存関係
このプロジェクトでは以下のPythonライブラリを使用しています：
- albumentations>=2.0.8
- datasets>=4.1.1
- evaluate>=0.4.6
- jiwer>=4.0.0
- matplotlib>=3.10.6
- numpy>=2.3.3
- opencv-python>=4.11.0.86
- protobuf>=6.32.1
- schedulefree>=1.4.1
- scikit-learn>=1.7.2
- sentencepiece>=0.2.1
- toml>=0.10.2
- torch==2.8.0
- transformers>=4.56.2

## 使用方法

### 1. モデルのトレーニング
`main_train.py`を使用してモデルをトレーニングします。
```bash
python main_train.py
```

### 2. 推論
`main_inference.py`を使用して推論を実行します。
```bash
python main_inference.py
```

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。