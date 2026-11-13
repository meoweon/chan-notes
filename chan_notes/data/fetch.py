"""baostock 数据抓取（单连接、限速）。

baostock 官方规则：禁止多连接（多线程/多进程）访问。
本模块使用全局会话，进程内单例。
"""
import baostock as bs
import pandas as pd

_session = None


def _bs():
    """获取 baostock 会话。"""
    return bs


def _drain(rs) -> list[list[str]]:
    """将结果集drain为列表。"""
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return rows


def login():
    """登录 baostock。"""
    global _session
    if _session is None:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
        _session = lg


def logout():
    """登出 baostock。"""
    global _session
    if _session is not None:
        bs.logout()
        _session = None


def fetch_30m(code: str, start: str, end: str) -> pd.DataFrame:
    """抓取 30 分钟线（前复权）。

    Args:
        code: 股票代码，如 sh.600000
        start: 开始日期，YYYY-MM-DD
        end: 结束日期，YYYY-MM-DD

    Returns:
        DataFrame with columns: code, time, open, high, low, close, volume, amount, adjustflag
    """
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency="30",
        adjustflag="2",  # 前复权
    )
    rows = _drain(rs)
    if not rows:
        return pd.DataFrame(columns=["code", "time", "open", "high", "low", "close", "volume", "amount", "adjustflag"])

    df = pd.DataFrame(rows, columns=fields.split(","))
    # 合并 date + time 为完整时间戳（baostock 30m 的 time 格式：YYYYMMDDHHmmssfff）
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
    df = df.drop(columns=["date"])
    # 转换数值列
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    """抓取日线（前复权）。

    Args:
        code: 股票代码，如 sh.600000
        start: 开始日期，YYYY-MM-DD
        end: 结束日期，YYYY-MM-DD

    Returns:
        DataFrame with columns: code, time, open, high, low, close, volume, amount, adjustflag, turn, pctChg
    """
    fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="2",  # 前复权
    )
    rows = _drain(rs)
    if not rows:
        return pd.DataFrame(columns=["code", "time", "open", "high", "low", "close", "volume", "amount", "adjustflag", "turn", "pctChg"])

    df = pd.DataFrame(rows, columns=fields.split(","))
    df = df.rename(columns={"date": "time"})
    df["time"] = pd.to_datetime(df["time"])
    # 转换数值列
    for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df
