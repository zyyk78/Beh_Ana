"""
每只鼠的 pkl 现状总览
=========================
按「鼠」（= 所在目录名）汇总当前 pkl 状态：
  正常 / 可疑(缺Mod) / 坏段(NaN、常数时间轴) / 空段 / 已合并(已改名为 .pkl.bak)
并给出每只鼠的实验天数、时间跨度、剩余的同鼠同天多 pkl 组。

用法：
    python mouse_pkl_status.py
    python mouse_pkl_status.py --csv mouse_pkl_status.csv
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

from pkl_health import OK, WARN, EMPTY, BROKEN, check_file

DATA_ROOTS = ["/mnt/HDD1/zyyk78"]
MERGED_DIR = "/home/jyy/Documents/Beh_Ana_J/merged_pkls"


def scan():
    """返回 {mouse: {...}} 汇总 + 每鼠同天多 pkl 组列表。"""
    per_mouse = defaultdict(lambda: {
        "pkl": 0, "bak": 0, "dates": set(), "ok": 0, "warn": 0,
        "broken": 0, "empty": 0, "dirs": set(),
    })
    day_groups = defaultdict(list)
    seen = set()

    for root in DATA_ROOTS:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.pkl*"):
            name = p.name
            if "miniscope" in str(p).lower():
                continue
            if name.endswith(".pkl.bak"):
                st = p.stat()
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                mouse = p.parent.name
                per_mouse[mouse]["bak"] += 1
                m = re.search(r"(\d{8})", name)
                if m:
                    per_mouse[mouse]["dates"].add(m.group(1))
                continue
            if not name.endswith(".pkl"):
                continue
            st = p.stat()
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)

            mouse = p.parent.name
            info = per_mouse[mouse]
            info["pkl"] += 1
            info["dirs"].add(str(p.parent))
            m = re.search(r"(\d{8})", name)
            if m:
                info["dates"].add(m.group(1))
                day_groups[(mouse, m.group(1))].append(str(p))

            r = check_file(p)
            lvl = r["level"]
            info[{"ok": "ok", "warn": "warn", "broken": "broken", "empty": "empty"}[lvl]] += 1
            info.setdefault("bad_list", []).append((name, lvl, r["reason"]))

    # 已合并目录
    merged_dir = Path(MERGED_DIR)
    merged_by_mouse = defaultdict(list)
    if merged_dir.exists():
        for p in merged_dir.rglob("*_merged.pkl"):
            mouse = p.parent.name
            m = re.search(r"(\d{8})", p.name)
            merged_by_mouse[mouse].append(p.name)
            if m:
                per_mouse[mouse]["dates"].add(m.group(1))
                per_mouse[mouse]["dirs"].add(str(p.parent))

    multi = {k: v for k, v in day_groups.items() if len(v) > 1}
    return per_mouse, multi, merged_by_mouse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    per_mouse, multi, merged_by_mouse = scan()
    print(f"{'鼠':6s}{'目录':22s}{'pkl':>5s}{'已合并(.bak)':>13s}{'合并产物':>9s}"
          f"{'天数':>5s}{'起':>9s}{'止':>9s}{'正常':>5s}{'可疑':>5s}{'坏段':>5s}{'空段':>5s}{'仍多pkl':>8s}")
    print("-" * 118)
    tot = defaultdict(int)
    for mouse in sorted(per_mouse):
        i = per_mouse[mouse]
        ds = sorted(i["dates"])
        nm = len([1 for k in multi if k[0] == mouse])
        # 数据根目录优先显示，merged_pkls/ 附在后面
        dirs = sorted(i["dirs"], key=lambda d: (d.startswith("/home/jyy"), d))
        dshort = dirs[0].replace("/mnt/HDD1/zyyk78/", "") if dirs else ""
        if len(dirs) > 1:
            dshort += f" +{len(dirs)-1}"
        print(f"{mouse:6s}{dshort:22s}{i['pkl']:5d}{i['bak']:13d}{len(merged_by_mouse.get(mouse, [])):9d}"
              f"{len(ds):5d}{(ds[0] if ds else '-'):>9s}{(ds[-1] if ds else '-'):>9s}"
              f"{i['ok']:5d}{i['warn']:5d}{i['broken']:5d}{i['empty']:5d}{nm:8d}")
        for k in ("pkl", "bak", "ok", "warn", "broken", "empty"):
            tot[k] += i[k]
    print("-" * 118)
    print(f"{'合计':6s}{'':22s}{tot['pkl']:5d}{tot['bak']:13d}{sum(len(v) for v in merged_by_mouse.values()):9d}"
          f"{'':5s}{'':>9s}{'':>9s}{tot['ok']:5d}{tot['warn']:5d}{tot['broken']:5d}{tot['empty']:5d}"
          f"{len(multi):8d}")

    print("\n【坏段 / 空段 明细】")
    for mouse in sorted(per_mouse):
        for name, lvl, reason in per_mouse[mouse].get("bad_list", []):
            if lvl in (BROKEN, EMPTY):
                print(f"  [{lvl:6s}] {mouse:6s} {name:28s} {reason}")

    print("\n【仍然存在的同鼠同天多 pkl 组】")
    for (mouse, date), files in sorted(multi.items()):
        print(f"  {mouse:6s} {date}: {[Path(f).name for f in files]}")

    if merged_by_mouse:
        print("\n【已合并产物（merged_pkls/）】")
        for mouse in sorted(merged_by_mouse):
            print(f"  {mouse:6s}: {merged_by_mouse[mouse]}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["mouse", "pkl", "bak_merged", "merged_out", "n_days",
                        "first_day", "last_day", "ok", "warn", "broken", "empty", "multi_groups"])
            for mouse in sorted(per_mouse):
                i = per_mouse[mouse]
                ds = sorted(i["dates"])
                w.writerow([mouse, i["pkl"], i["bak"], len(merged_by_mouse.get(mouse, [])), len(ds),
                            ds[0] if ds else "", ds[-1] if ds else "",
                            i["ok"], i["warn"], i["broken"], i["empty"],
                            len([1 for k in multi if k[0] == mouse])])
        print(f"\nCSV: {a.csv}")


if __name__ == "__main__":
    main()
