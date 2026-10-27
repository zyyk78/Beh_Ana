"""
行为 pkl 健康检查（坏段判定）
================================
管线上游（fixing.py / ParseEvent.py）偶尔会产出「时间轴坏掉」的 pkl：
  - 事件时间戳里出现 NaN（Mod 引导信号整段为 NaN）
  - 所有 trial 事件的时间戳是同一个常数（事件被压到同一时刻）
  - 顶层不是 dict、或某个事件 key 不是 dict（Miniscope 钙成像结果 pkl 也属于这类）
这类 pkl 的 trial 数可能看起来不小（例如 1028 个），
但一旦并进同鼠同天的事件字典，就会把 trial 序列/排序彻底带偏，
在 sort_events / transform_sequence / learning_auc 处炸掉。

本模块给出统一判据，管线（2-analysis.ipynb cell 7）与 merge_pkls.py 共用同一套规则。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

TRIAL_KEYS = ["SlnL", "SlnR", "ROmL", "ROmR"]
MOD_KEYS = ["ModL", "ModR"]
NEED_KEYS = ["ModL", "ModR", "LickL", "LickR", "SlnL", "SlnR", "ROmL", "ROmR"]
EMPTY_THRESH = 10        # 与 2-analysis.ipynb 的空段阈值一致

OK, WARN, EMPTY, BROKEN = "ok", "warn", "empty", "broken"


def _ts(v):
    """把 Time_S / Time_E 转成 float 数组（NaN 保留，便于统计）。"""
    if v is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(v, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)


def check_event(Event, empty_threshold=EMPTY_THRESH, need_keys=NEED_KEYS):
    """检查一个已加载的 Event（行为事件字典）。

    Returns
    -------
    dict: {level: 'ok'|'empty'|'broken', reason: str, detail: dict}
    """
    detail = {}
    if not isinstance(Event, dict):
        return {"level": BROKEN, "reason": f"顶层不是 dict（{type(Event).__name__}），非行为数据",
                "detail": detail}

    # 1) 事件 key 必须是 dict
    bad_keys = [k for k in need_keys
                if Event.get(k) is not None and not isinstance(Event[k], dict)]
    if bad_keys:
        detail["bad_keys"] = {k: type(Event[k]).__name__ for k in bad_keys}
        return {"level": BROKEN, "reason": f"事件 key 不是 dict：{bad_keys}", "detail": detail}

    # 2) 时间戳健康度：NaN / inf
    nan_keys, total_ts = [], 0
    trial_ts = []
    for k, v in Event.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        for fld in ("Time_S", "Time_E"):
            a = _ts(v.get(fld))
            if a.size:
                total_ts += a.size
                if not np.all(np.isfinite(a)):
                    nan_keys.append(f"{k}.{fld}")
        if k in TRIAL_KEYS:
            trial_ts.append(_ts(v.get("Time_S")))

    detail["nan_keys"] = nan_keys
    detail["n_timestamps"] = total_ts
    if nan_keys:
        n_bad = 0
        for k, v in Event.items():
            if isinstance(k, str) and isinstance(v, dict):
                for fld in ("Time_S", "Time_E"):
                    a = _ts(v.get(fld))
                    if a.size:
                        n_bad += int(a.size - np.isfinite(a).sum())
        detail["n_nan"] = n_bad
        return {"level": BROKEN,
                "reason": f"时间戳含 NaN/inf（{n_bad} 个，键 {nan_keys[:6]}）",
                "detail": detail}

    # 3) trial 时间轴：所有 trial 事件被压到同一时刻
    tr = np.concatenate(trial_ts) if trial_ts else np.array([], dtype=float)
    tr = tr[np.isfinite(tr)]
    detail["n_trial_events"] = int(tr.size)
    detail["n_unique_trial_ts"] = int(np.unique(tr).size) if tr.size else 0
    # 只有「事件数够多、却全挤在同一时刻」才是时间轴被压扁；
    # 事件本来就只有一两个的小文件属于空段，交给 EMPTY 规则，不要误判成坏段。
    if tr.size >= empty_threshold and detail["n_unique_trial_ts"] <= 1:
        detail["const_ts"] = float(tr[0])
        return {"level": BROKEN,
                "reason": f"所有 trial 时间戳相同（{tr[0]:.0f}），时间轴被压扁",
                "detail": detail}

    # 4) trial 引导信号（Mod）缺失 —— 只告警，不自动剔除
    #    （旧管线把这类 session 保留下来：transform_sequence 会把正确侧默认成 L，
    #      cell 7 里再把 accuracy 置 0。自动剔除会改变数据集，所以这里只提示。）
    n_mod = 0
    for k in MOD_KEYS:
        v = Event.get(k)
        if isinstance(v, dict):
            n_mod += _ts(v.get("Time_S")).size
    detail["n_mod"] = int(n_mod)
    if tr.size and n_mod == 0:
        return {"level": WARN,
                "reason": "无 Mod 引导事件（trial 正确侧将默认按 L 处理，accuracy 会置 0）",
                "detail": detail}

    # 5) 空段：trial / reward 太少
    n_rew = 0
    for k in ("SlnL", "SlnR"):
        v = Event.get(k)
        if isinstance(v, dict):
            n_rew += _ts(v.get("Time_S")).size
    detail["n_reward"] = int(n_rew)
    if tr.size <= empty_threshold or n_rew <= empty_threshold:
        return {"level": EMPTY,
                "reason": f"trial={int(tr.size)} reward={int(n_rew)}（≤ 空段阈值 {empty_threshold}）",
                "detail": detail}

    return {"level": OK, "reason": "正常", "detail": detail}


def check_file(path, empty_threshold=EMPTY_THRESH):
    """检查单个 pkl 文件（读盘 + check_event）。"""
    p = Path(path)
    out = {"file": p.name, "path": str(p), "MB": p.stat().st_size / 1e6}
    try:
        with open(p, "rb") as f:
            Event = pickle.load(f)
    except Exception as e:
        out.update(level=BROKEN, reason=f"读取失败：{type(e).__name__}: {e}")
        return out
    res = check_event(Event, empty_threshold=empty_threshold)
    out.update(level=res["level"], reason=res["reason"], detail=res["detail"])
    return out


def main():
    ap = argparse.ArgumentParser(description="扫描行为 pkl，报告坏段/空段")
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    roots = a.root or ["/mnt/HDD1/zyyk78", "/home/zyyk78/HDD/zyyk78/Stage4"]

    rows, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            if "miniscope" in str(p).lower():
                continue
            st = p.stat()
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            rows.append(check_file(p))

    bad = [r for r in rows if r["level"] == BROKEN]
    emp = [r for r in rows if r["level"] == EMPTY]
    print(f"扫描 {len(rows)} 个 pkl：坏段 {len(bad)}，空段 {len(emp)}，正常 {len(rows) - len(bad) - len(emp)}")
    for r in bad + emp:
        print(f"  [{r['level']:6s}] {r['file']:32s} {r['MB']:7.2f}MB  {r['reason']}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["level", "file", "path", "MB", "reason"])
            for r in rows:
                w.writerow([r["level"], r["file"], r["path"], f"{r['MB']:.3f}", r["reason"]])
        print(f"CSV: {a.csv}")


if __name__ == "__main__":
    main()
