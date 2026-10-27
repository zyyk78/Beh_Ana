"""
同鼠同天多 pkl 物理合并工具
================================
把同一只鼠同一天的多个有效 pkl 合并成一个独立的 pkl 文件，
偏移约定与 2-analysis.ipynb 的 _merge_events() 完全一致：

    offset = 0
    for 每段（按 seq 排序）:
        该段所有时间戳 += offset
        offset += 该段最大时间戳 + MERGE_GAP      # MERGE_GAP = 50000 (=1s @50kHz)

安全策略（默认只打印计划，加 --apply 才真正写盘）：
  * 只合并 pkl_health 判定为 ok 的段；坏段/空段自动排除并报告
  * 两段若存在重复（共有 trial 时间戳 / 间隔序列高度相关）→ 判为重复导出，只保留大的
  * 原文件永不改动；合并结果写到独立目录 <out>/<鼠名>/，并附 .merge_meta.json 溯源

用法：
    python merge_pkls.py --root /mnt/HDD1/zyyk78 --out ./merged_pkls            # 计划（dry-run）
    python merge_pkls.py --root /mnt/HDD1/zyyk78 --out ./merged_pkls --apply    # 真正写出
"""
import argparse
import hashlib
import json
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from pkl_health import OK, WARN, check_file

MERGE_GAP = 50000          # 段间偏移量（1s @50kHz），与管线保持一致
TRIAL_KEYS = ["SlnL", "SlnR", "ROmL", "ROmR"]


def sha1_of(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seq_key(p):
    m = re.search(r"_(\d{2})\.pkl$", p.name)
    return m.group(1) if m else "00"


def seg_max_ts(Event):
    """该段所有时间戳的最大值（Time_S / Time_E 一起算）。"""
    mx = 0
    for k, v in Event.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        for fld in ("Time_S", "Time_E"):
            a = v.get(fld)
            if a is not None and len(a):
                mx = max(mx, max(a))
    return mx


def trial_ts_of(Event):
    ts = []
    for k in TRIAL_KEYS:
        v = Event.get(k)
        if isinstance(v, dict):
            ts += [float(x) for x in (v.get("Time_S") or [])]
    return np.array(sorted(ts))


def merge_events(events):
    """与 2-analysis.ipynb _merge_events() 同款偏移合并。"""
    merged, offset = {}, 0
    for Ev in events:
        for k, v in Ev.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            if k not in merged:
                merged[k] = {"Time_S": [], "Time_E": []}
            ts = v.get("Time_S") or []
            te = v.get("Time_E") or []
            merged[k]["Time_S"].extend([t + offset for t in ts])
            merged[k]["Time_E"].extend([t + offset for t in te])
        offset += seg_max_ts(Ev) + MERGE_GAP
    return merged


def duplicate_check(ta, tb):
    """判断两段是否互为重复导出：共有 trial 时间戳数 / 间隔序列相关。"""
    shared = int(np.intersect1d(ta, tb).size) if ta.size and tb.size else 0
    ga, gb = np.diff(ta), np.diff(tb)
    corr = 0.0
    if ga.size > 20 and gb.size > 20:
        n = min(ga.size, gb.size)
        c = float(np.corrcoef(ga[:n], gb[:n])[0, 1])
        corr = 0.0 if np.isnan(c) else c
    return shared, corr


def group_pkls(roots):
    groups = defaultdict(list)
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            if "miniscope" in str(p).lower():
                continue
            m = re.search(r"(\d{8})", p.name)
            if not m:
                continue
            st = p.stat()
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            groups[(p.parent, p.parent.name, m.group(1))].append(p)
    return groups


def plan(roots, out_dir):
    groups = group_pkls(roots)
    results = []
    for (parent, mouse, date), files in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if len(files) < 2:
            continue
        files = sorted(files, key=seq_key)
        checks, parts, excluded = [], [], []
        for f in files:
            c = check_file(f)
            checks.append(c)
            if c["level"] in (OK, WARN):   # 可疑段（缺 Mod 等）仍可合并，但会在输出里标注
                parts.append(f)
            else:
                excluded.append(c)

        loaded = {}
        for f in parts:
            with open(f, "rb") as fh:
                loaded[f] = pickle.load(fh)

        dup_pairs, drop = [], set()
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                fa, fb = parts[i], parts[j]
                shared, corr = duplicate_check(trial_ts_of(loaded[fa]), trial_ts_of(loaded[fb]))
                if shared > 0 or corr > 0.9:
                    keep, drop_one = (fa, fb) if fa.stat().st_size >= fb.stat().st_size else (fb, fa)
                    dup_pairs.append((fa.name, fb.name, shared, corr))
                    drop.add(drop_one)
        if drop:
            parts = [f for f in parts if f not in drop]

        results.append({
            "mouse": mouse, "date": date, "dir": str(parent),
            "files": files, "checks": checks, "excluded": excluded,
            "parts": parts, "dup_pairs": dup_pairs,
            "out": Path(out_dir) / mouse / (f"{date}_{'_'.join(seq_key(p) for p in parts)}_merged.pkl"),
        })
    return results


def execute(item, apply=False):
    parts = item["parts"]
    if len(parts) < 2:
        return None, None
    events = []
    for f in parts:
        with open(f, "rb") as fh:
            events.append(pickle.load(fh))
    merged = merge_events(events)

    meta = {
        "mouse": item["mouse"],
        "date": item["date"],
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "merge_gap": MERGE_GAP,
        "source_dir": item["dir"],
        "parts": [],
    }
    offset = 0
    for f, Ev in zip(parts, events):
        mt = seg_max_ts(Ev)
        tt = trial_ts_of(Ev)
        meta["parts"].append({
            "file": f.name,
            "path": str(f),
            "sha1": sha1_of(f),
            "MB": round(f.stat().st_size / 1e6, 4),
            "offset": offset,
            "seg_max_ts": mt,
            "n_trial_events": int(tt.size),
            "t_min": float(tt.min()) if tt.size else None,
            "t_max": float(tt.max()) if tt.size else None,
        })
        offset += mt + MERGE_GAP

    if apply:
        item["out"].parent.mkdir(parents=True, exist_ok=True)
        with open(item["out"], "wb") as fh:
            pickle.dump(merged, fh, protocol=4)
        with open(str(item["out"]) + ".merge_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    return merged, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--out", default="merged_pkls")
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认只打印计划）")
    a = ap.parse_args()
    roots = a.root or ["/mnt/HDD1/zyyk78", "/home/zyyk78/HDD/zyyk78/Stage4"]

    items = plan(roots, a.out)
    print(f"扫描 {len(items)} 个「同鼠同天多 pkl」组\n")
    n_merged = 0
    for it in items:
        print("=" * 90)
        print(f"[{it['mouse']}] {it['date']}   {it['dir']}")
        for c in it["checks"]:
            flag = "合并" if c["level"] in (OK, WARN) else "排除"
            print(f"  {c['file']:32s} {c['MB']:7.2f}MB  [{c['level']:6s}|{flag}] {c['reason']}")
        for a1, b1, sh, co in it["dup_pairs"]:
            print(f"  ! 疑似重复导出：{a1} <-> {b1}（共有 trial 时间戳 {sh}，间隔序列相关 {co:.2f}）→ 只保留大的")
        if len(it["parts"]) >= 2:
            merged, meta = execute(it, apply=a.apply)
            n_merged += 1
            tail = "" if a.apply else "（dry-run，未写盘）"
            print(f"  => 合并 {len(it['parts'])} 段 -> {it['out']}{tail}")
            for p in meta["parts"]:
                print(f"      + {p['file']}  offset={p['offset']}  seg_max_ts={p['seg_max_ts']}  "
                      f"trial={p['n_trial_events']}")
        elif len(it["parts"]) == 1:
            print(f"  => 仅 1 个有效段，无需合并：{it['parts'][0].name}")
        else:
            print("  => 无有效段")
    print("=" * 90)
    print(f"需合并组: {n_merged} / {len(items)}；写盘模式: {'apply' if a.apply else 'dry-run'}")


if __name__ == "__main__":
    main()
