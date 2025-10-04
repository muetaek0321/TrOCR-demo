import datetime


__all__ = ["now_date_str"]


def now_date_str(
    fmt: str = "%Y-%m-%d_%H-%M-%S"
) -> str:
    """現在時刻の文字列を作成
    
    Args:
        fmt (str): 時刻表示のフォーマット指定
    """
    t_delta = datetime.timedelta(hours=9)
    jst = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(jst)
    
    return now.strftime(fmt)
