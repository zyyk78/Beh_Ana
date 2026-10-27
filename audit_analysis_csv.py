"""
analysis_results CSV 体检
==========================
对 2-analysis.ipynb 产出的 analysis_results_*.csv 做全面体检，
并把每个问题归因到：
   [log] = 硬件 log(llom_log) 缺失 / 当天没做 -> 数据本来就少，属上游事实
   [bug] = 管线代码漏洞（统计口径、过滤逻辑、字段缺失、路径解析等）
   [data]= 数据本身缺陷（坏段/空段/缺 Mod），已被管线过滤或已标注

用法：python audit_analysis_csv.py [csv路径]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("/mnt/HDD1/zyyk78")
LOG_DIR = DATA_ROOT / "llom_log"


def log_dates():
    out = set()
    if LOG_DIR.exists():
        for p in LOG_DIR.glob("log*.txt"):
            m = re.search(r"log(\d{8})", p.name)
            if m:
                out.add(m.group(1))
    return out


def available_groups():
    """扫描根目录下所有行为 pkl，按(鼠目录, 日期)分组。"""
    groups = defaultdict(list)
    n_mini = 0
    seen = set()
    for p in DATA_ROOT.rglob("*.pkl"):
        st = p.stat()
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        if "miniscope" in str(p).lower():
            n_mini += 1
            continue
        m = re.search(r"(\d{8})", p.name)
        if not m:
            continue
        groups[(p.parent.name, m.group(1))].append(p.name)
    return groups, n_mini


def mouse_of(path):
    return Path(path).parent.name


def date_of(path):
    m = re.search(r"(\d{8})", Path(path).name)
    return m.group(1) if m else ""


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "analysis_results_20261027_165201.csv"
    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path}   行数={len(df)}")

    logs = log_dates()
    groups, n_mini = available_groups()
    print(f"扫描 {DATA_ROOT}: 行为鼠×天组={len(groups)} (另有 {n_mini} 个 miniscope pkl, 管线会跳过)")
    print(f"llom_log 日志数={len(logs)}")

    df["_mouse"] = df["Train_file"].map(mouse_of)
    df["_date"] = df["Train_file"].map(date_of)
    df["_multi"] = df["Train_file"].astype(str).str.contains(",")

    print("\n" + "=" * 100)
    print("【问题 1】缺失的 session（磁盘上有 pkl，但 CSV 没这一行）")
    csv_keys = set(zip(df["_mouse"], df["_date"]))
    missing = {k: v for k, v in groups.items() if k not in csv_keys}
    print("  无" if not missing else "")
    for (mouse, date), files in sorted(missing.items()):
        print(f"  {mouse:6s} {date}: 磁盘有 {files}")

    print("\n" + "=" * 100)
    print("【问题 2】重复行（同一鼠×天出现多行）")
    dup = df.groupby(["_mouse", "_date"]).size()
    dup = dup[dup > 1]
    print("  无" if dup.empty else dup.to_string())

    print("\n" + "=" * 100)
    print("【问题 3】逐行指标异常统计")
    checks = [
        ("Accuracy == 0", df["Accuracy"] == 0),
        ("Learning_auc == 0", df["Learning_auc"] == 0),
        ("avg_licks_after_reward == 0", df["avg_licks_after_reward"] == 0),
        ("avg_licks_after_noreward == 0", df["avg_licks_after_noreward"] == 0),
        ("Switching Latency == 0", df["Switching Latency"] == 0),
        ("Incorrect Switch == 0", df["Incorrect Switch"] == 0),
        ("Trial_Gap == 0", df["Trial_Gap"] == 0),
        ("Reward Ratio == 1.0", df["Reward Ratio"] == 1.0),
        ("Reward Ratio == 0.0", df["Reward Ratio"] == 0.0),
    ]
    for name, mask in checks:
        print(f"  {name:34s}: {int(mask.sum())} 行")

    nomod = df[(df["Accuracy"] == 0) & (df["Reward Ratio"] > 0)]
    print(f"\n  → 『有 reward 但 Accuracy=0』(缺 Mod 引导, 正确侧被默认成 L): {len(nomod)} 行")
    print("     涉及鼠:", sorted(nomod["_mouse"].unique()))
    if len(nomod):
        has_log = sum(1 for d in nomod["_date"] if d in logs)
        print(f"     当天有 llom_log: {has_log} 行 / 无 log: {len(nomod) - has_log} 行")
        print("     无 log 的日期:", sorted({d for d in nomod["_date"] if d not in logs}))

    print("\n" + "=" * 100)
    print("【问题 4】log 缺失 vs 指标异常")
    df["_has_log"] = df["_date"].isin(logs)
    sub = df.groupby("_has_log").agg(
        n=("Accuracy", "size"),
        acc0_ratio=("Accuracy", lambda s: float((s == 0).mean())),
        lick0_ratio=("avg_licks_after_reward", lambda s: float((s == 0).mean())),
        sw0_ratio=("Switching Latency", lambda s: float((s == 0).mean())),
        rew_mean=("Reward Ratio", "mean"),
        gap_mean=("Trial_Gap", "mean"),
        acc_mean=("Accuracy", "mean"),
    )
    print(sub.to_string())
    no_log_days = sorted({d for d in df["_date"] if d not in logs})
    print(f"\n  CSV 涉及但 llom_log 里没有的日期: {len(no_log_days)} 天")
    print("  ", no_log_days[:50])

    print("\n" + "=" * 100)
    print("【问题 5】字段/口径类问题（与 log 无关）")
    bad_mouse = sorted({m for m in df["_mouse"] if not re.match(r"^[A-Z]\d{2}$", m)})
    print(f"  5.1 鼠目录名不符合 [A-Z]\\d{{2}} (下游 Mouse_ID 正则会解析失败): {bad_mouse}")
    for m in bad_mouse:
        print(f"        {m}: {int((df['_mouse'] == m).sum())} 行受影响")
    print(f"  5.2 合并组行数 (Train_file 含逗号): {int(df['_multi'].sum())}")
    mism = df[df["Date"].astype(str) != df["_date"]]
    print(f"  5.3 Date 列与文件日期不一致的行数: {len(mism)}")
    print(f"  5.4 CSV 列 (无独立 Mouse_ID 列): {list(df.columns[:12])}")
    for c in ["Reward Ratio", "Accuracy"]:
        bad = df[(df[c] < 0) | (df[c] > 1)]
        print(f"  5.5 {c} 越界(不在[0,1]): {len(bad)} 行")
    print("  5.6 Trial_Gap 定义: (最后 lick/reward 事件 - 最前) / (lick 事件数 x 50000)，单位秒")

    print("\n" + "=" * 100)
    print("【问题 6】与历史 CSV 的对比")
    for o in sorted(Path(".").glob("analysis_results_*.csv")):
        if o.name == csv_path:
            continue
        try:
            od = pd.read_csv(o)
        except Exception:
            continue
        if len(od) < 100:
            continue
        pcol = "Train_file" if "Train_file" in od.columns else ("Train" if "Train" in od.columns else None)
        if pcol is None:
            continue
        ok = set(zip(od[pcol].map(mouse_of), od[pcol].map(date_of)))
        nk = set(zip(df["_mouse"], df["_date"]))
        print(f"  vs {o.name}: 行数 {len(od)} -> {len(df)}; 旧有新无 {len(ok - nk)}; 新有旧无 {len(nk - ok)}")
        for k in sorted(nk - ok):
            print(f"        +{k}")
        for k in sorted(ok - nk):
            print(f"        -{k}")


if __name__ == "__main__":
    main()
