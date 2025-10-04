from pathlib import Path

import numpy as np
import cv2


__all__ = ["imread_jpn", "imwrite_jpn"]


def imread_jpn(
    img_path: Path | str
) -> np.ndarray:
    """日本語を含むパスの画像を読み込み
    
    Args:
        img_path (Path, str): 画像ファイルパス
    
    Returns:
        np.ndarray: 画像データ   
    """
    # NumPyで画像ファイルを開く
    buf = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    return img


def imwrite_jpn(
    output_path: Path | str,
    img: np.ndarray
) -> None:
    """日本語を含むパスの画像を保存
    
    Args:
        output_path (Path, str): 出力先パス
        img (np.ndarray): 画像データ
    """
    # 画像データを変換
    ext = Path(output_path).suffix
    result, n = cv2.imencode(ext, img)

    # 保存
    if result:
        with open(output_path, mode='w+b') as f:
            n.tofile(f)
