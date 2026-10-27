"""
同鼠同天多 pkl 体检报告
=========================
目的：回答两件事
  1) 这些 pkl 各自有多大、装了什么（trial/lick/miniscope 事件数、时间跨度）
  2) 它们彼此是什么关系（内容完全相同？部分重叠？互不重叠的连续两段？时间轴损坏？）
     —— 从而决定「删掉某个」还是「把某两个合并」。

用法：
    python pkl_group_report.py                      # 默认扫描常用根目录
    python pkl_group_report.py --root /mnt/HDD1/zyyk78
    python pkl_group_report.py --out pkl_report.csv # 同时导出 CSV
    python pkl_group_report.py --all                # 连单 pkl 的鼠日也列出来
"""
import argparse
import hashlib
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

FS = 50000.0            # 采样率(Hz)：时间戳 / 50000 = 秒
TRIAL_KEYS = ["SlnL", "SlnR", "ROmL", "ROmR"]
BLOCK_GAP_S = 1.0       # 事件间隔超过该值即认为是两个 block
EMPTY_THRESH = 10       # 与 2-analysis.ipynb 的空段阈值保持一致

DEFAULT_ROOTS = [
    "/mnt/HDD1/zyyk78",
    "/home/zyyk78/HDD/zyyk78/Stage4",
]


def _as_floats(v):
    """把 Time_S 变成 float 数组，NaN 保留（后面统计有效比例用）。"""
    if v is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(v, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)


def profile(pkl_path):
    """统计单个 pkl 的体检信息。"""
    p = Path(pkl_path)
    d = {
        "file": p.name,
        "path": str(p),
        "MB": p.stat().st_size / 1e6,
        "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "status": "OK",
        "reason": "",
    }
    try:
        with open(p, "rb") as f:
            Ev = pickle.load(f)
    except Exception as e:
        d.update(status="DELETE", reason=f"无法加载({type(e).__name__})")
        return d

    if not isinstance(Ev, dict):
        d.update(status="DELETE", reason=f"顶层不是 dict（{type(Ev).__name__}），非行为数据")
        return d

    counts, finite, all_ts, nan_ts = {}, 0, [], 0
    trial_ts = []
    for k, v in Ev.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        ts = _as_floats(v.get("Time_S"))
        counts[k] = int(ts.size)
        good = ts[np.isfinite(ts)]
        finite += good.size
        nan_ts += int(ts.size - good.size)
        if good.size:
            all_ts.append(good)
        if k in TRIAL_KEYS:
            trial_ts.append(good)

    total_ts = sum(counts.values())
    d["events"] = total_ts
    d["nan_ratio"] = (nan_ts / total_ts) if total_ts else 0.0
    d["trials"] = sum(counts.get(k, 0) for k in TRIAL_KEYS)
    d["mods"] = counts.get("ModL", 0) + counts.get("ModR", 0)
    d["licks"] = counts.get("LickL", 0) + counts.get("LickR", 0)
    d["minS"] = counts.get("MinS", 0)
    d["SlnL"], d["SlnR"] = counts.get("SlnL", 0), counts.get("SlnR", 0)
    d["ROmL"], d["ROmR"] = counts.get("ROmL", 0), counts.get("ROmR", 0)

    flat = np.concatenate(all_ts) if all_ts else np.array([], dtype=float)
    d["t_min"] = float(flat.min()) if flat.size else None
    d["t_max"] = float(flat.max()) if flat.size else None
    d["span_s"] = (d["t_max"] - d["t_min"]) / FS if flat.size else 0.0

    tr = np.concatenate(trial_ts) if trial_ts else np.array([], dtype=float)
    tr = np.sort(tr)
    d["uniq_trial_ts"] = int(np.unique(tr).size) if tr.size else 0
    d["_trial_ts"] = tr          # 保留给两两关系判定用
    d["_gaps"] = np.diff(tr) if tr.size > 1 else np.array([], dtype=float)
    blocks = 1 if tr.size else 0
    if tr.size > 1:
        blocks += int(((np.diff(tr) / FS) > BLOCK_GAP_S).sum())
    d["blocks"] = blocks

    # 内容指纹：同一事件内容的 pkl 指纹相同 → 完全重复
    h = hashlib.md5()
    for k in sorted(counts):
        v = Ev[k]
        ts = _as_floats(v.get("Time_S"))
        ts = np.sort(ts[np.isfinite(ts)])
        h.update(k.encode())
        h.update(ts.astype(np.int64).tobytes())
    d["hash"] = h.hexdigest()[:10]

    # ---- 判定该 pkl 是废段还是有效段 ----
    if d["trials"] == 0:
        d.update(status="DELETE", reason="无 trial 事件（空段）")
    elif d["uniq_trial_ts"] == 1:
        d.update(status="DELETE", reason=f"所有 trial 时间戳相同({d['t_min']:.0f})，时间轴损坏")
    elif d["nan_ratio"] > 0.5:
        d.update(status="DELETE", reason=f"{d['nan_ratio']:.0%} 时间戳为 NaN")
    elif d["trials"] <= EMPTY_THRESH:
        d.update(status="DELETE", reason=f"trial 仅 {d['trials']} 个（≤空段阈值 {EMPTY_THRESH}）")
    elif d["mods"] == 0:
        d.update(status="DELETE", reason="无 Mod 事件（无 trial 引导信号）")
    else:
        d["status"] = "KEEP"
        d["reason"] = "有效段"
    return d


def pairwise(a, b):
    """两个 pkl 的关系：
       - same_content : 事件集合完全相同（同一份数据被存了两遍）
       - shared_ts    : 共有 trial 时间戳个数（>0 说明同一时钟下的重复导出）
       - dup_corr     : 试次间隔序列相似度（≈1 说明是同一段数据的错位副本）
       - overlap_s    : 时间区间重叠秒数（各文件时钟独立，单独看会误导，仅作参考）
    """
    out = {"overlap_s": 0.0, "same_content": a.get("hash") == b.get("hash"),
           "shared_ts": 0, "dup_corr": 0.0}
    if a.get("t_min") is not None and b.get("t_min") is not None:
        lo = max(a["t_min"], b["t_min"])
        hi = min(a["t_max"], b["t_max"])
        out["overlap_s"] = max(0.0, (hi - lo) / FS)
    ta, tb = a.get("_trial_ts"), b.get("_trial_ts")
    if ta is not None and tb is not None and ta.size and tb.size:
        out["shared_ts"] = int(np.intersect1d(ta, tb).size)
        ga, gb = a.get("_gaps"), b.get("_gaps")
        if ga is not None and gb is not None and ga.size > 20 and gb.size > 20:
            n = min(ga.size, gb.size)
            c = float(np.corrcoef(ga[:n], gb[:n])[0, 1])
            out["dup_corr"] = 0.0 if np.isnan(c) else c
    return out


def group_verdict(items):
    """给整组（鼠×天）一句话建议。"""
    keep = [i for i in items if i["status"] == "KEEP"]
    dele = [i for i in items if i["status"] == "DELETE"]
    if not keep:
        return "整组皆废段，可全部删除"
    if dele:
        names = ", ".join(i["file"] for i in dele)
        return f"删除 {names}；其余 {len(keep)} 段合并"
    if len(keep) == 1:
        return "仅一个有效段，无需合并"
    if all(i.get("hash") == keep[0].get("hash") for i in keep):
        return "内容完全重复，只保留体积最大的一个"
    rels = [pairwise(keep[i], keep[j]) for i in range(len(keep)) for j in range(i + 1, len(keep))]
    for rel in rels:
        if rel["same_content"]:
            return "重复拷贝，删其一"
        if rel["dup_corr"] > 0.9:
            return f"间隔序列相似度 {rel['dup_corr']:.2f}，同一段数据的副本，删其一"
        if rel["shared_ts"] > 0.5 * min(len(i.get("_trial_ts", [])) for i in keep):
            return f"共有 {rel['shared_ts']} 个 trial 时间戳，同一时钟下的重复导出，删其一"
    return f"{len(keep)} 个互不重叠的有效段，按 seq 顺序合并"


def _seq_key(p):
    m = re.search(r"_(\d{2})\.pkl$", p.name)
    return m.group(1) if m else "00"


def collect(roots, only_multi=True, include_miniscope=False):
    groups = defaultdict(list)
    seen = set()          # (st_dev, st_ino) 去重：/home/zyyk78/HDD 是 /mnt/HDD1 的软链，同一文件会被扫到两次
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            m = re.search(r"(\d{8})", p.name)
            if not m:
                continue
            if not include_miniscope and "miniscope" in str(p).lower():
                continue          # 钙成像结果 pkl，不是行为数据
            try:
                st = p.stat()
                key = (st.st_dev, st.st_ino)
            except Exception:
                key = None
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            groups[(p.parent, p.parent.name, m.group(1))].append(p)
    out = []
    for (parent, mouse, date), files in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if only_multi and len(files) < 2:
            continue
        files = sorted(set(files), key=_seq_key)
        items = [profile(f) for f in files]
        out.append({"mouse": mouse, "date": date, "dir": str(parent),
                    "items": items, "verdict": group_verdict(items)})
    return out


def print_report(groups):
    for g in groups:
        print("=" * 100)
        print(f"[{g['mouse']}] {g['date']}   {g['dir']}")
        print(f"  建议：{g['verdict']}")
        print(f"  {'file':26s}{'MB':>7s}{'mtime':>18s}{'trials':>8s}{'Sln':>7s}{'ROm':>7s}"
              f"{'Mod':>6s}{'lick':>8s}{'MinS':>8s}{'span_s':>10s}{'blocks':>8s}  status/reason")
        for i in g["items"]:
            print(f"  {i['file']:26s}{i['MB']:7.2f}{i['mtime']:>18s}{i.get('trials', 0):8d}"
                  f"{i.get('SlnL', 0) + i.get('SlnR', 0):7d}{i.get('ROmL', 0) + i.get('ROmR', 0):7d}"
                  f"{i.get('mods', 0):6d}{i.get('licks', 0):8d}{i.get('minS', 0):8d}"
                  f"{i.get('span_s', 0):10.1f}{i.get('blocks', 0):8d}  "
                  f"{'x DELETE' if i['status'] == 'DELETE' else '+ KEEP  '}  {i['reason']}")
        for a in range(len(g["items"])):
            for b in range(a + 1, len(g["items"])):
                x, y = g["items"][a], g["items"][b]
                if "hash" not in x or "hash" not in y:
                    continue
                rel = pairwise(x, y)
                print(f"    关系 {x['file']} <-> {y['file']}: "
                      f"时间轴重叠 {rel['overlap_s']:.1f}s / "
                      f"{'内容完全相同' if rel['same_content'] else '内容不同'} / "
                      f"共有trial时间戳 {rel['shared_ts']} / 间隔序列相似度 {rel['dup_corr']:.2f}")


def export_csv(groups, path):
    import csv
    cols = ["mouse", "date", "dir", "file", "MB", "mtime", "status", "reason",
            "trials", "SlnL", "SlnR", "ROmL", "ROmR", "mods", "licks", "minS",
            "span_s", "blocks", "nan_ratio", "hash"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in groups:
            for i in g["items"]:
                row = {c: i.get(c, "") for c in cols}
                row.update(mouse=g["mouse"], date=g["date"], dir=g["dir"])
                w.writerow(row)
    print(f"\nCSV 已导出: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None, help="扫描根目录，可重复")
    ap.add_argument("--out", default=None, help="导出 CSV 路径")
    ap.add_argument("--all", action="store_true", help="也列出单 pkl 的鼠日")
    ap.add_argument("--include-miniscope", action="store_true", help="也扫描 Miniscope 钙成像结果")
    a = ap.parse_args()

    roots = a.root or DEFAULT_ROOTS
    groups = collect(roots, only_multi=not a.all, include_miniscope=a.include_miniscope)
    print(f"扫描根目录: {roots}")
    print(f"同鼠同天多 pkl 组: {len(groups)}\n")
    print_report(groups)
    if a.out:
        export_csv(groups, a.out)


if __name__ == "__main__":
    main()
