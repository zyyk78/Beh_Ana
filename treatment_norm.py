"""Treatment 统一映射
表：treatment_map.csv（raw -> standard），笔误自动映射。
用法:
    from treatment_norm import load_map, norm_treatment
    TMAP = load_map()                    # 或 load_map("treatment_map.csv")
    df["Treatment"] = df["Train"].map(norm_treatment)
未登录的新值: 原样返回并打印一次提醒（不会静默丢数据）。
"""
import csv, os
import pandas as pd

_MAP = None
_WARNED = set()
_DEFAULT = "00"

def load_map(path=None):
    global _MAP
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treatment_map.csv")
    m = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = (row.get("raw") or "").strip()
            std = (row.get("standard") or "").strip()
            if raw:
                m[raw] = std
    _MAP = m
    return m

def norm_treatment(name):
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return _DEFAULT
    name = str(name).strip()
    if _MAP is None:
        load_map()
    if name in _MAP:
        return _MAP[name]
    if name not in _WARNED:
        print(f"⚠️ 未登记的 treatment: {name!r}（按原样保留，请补进 treatment_map.csv）")
        _WARNED.add(name)
    return name
