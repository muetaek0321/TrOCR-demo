import random

import numpy as np
import torch


__all__ = ["fix_seeds"]


def fix_seeds(
    seed: int = 0
) -> None:
    """使用する可能性のあるライブラリの乱数を固定
    
    Args:
        seed (int): 乱数のSEED値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
