import sys, os, re, pickle
sys.path.insert(0, "/home/jyy/Documents/Beh_Ana_J")
import behave_ana
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ====== 复刻 notebook cell[7] 完整加载逻辑（口径一致）======
def _time_s(Event, key):
    v = Event.get(key)
    if isinstance(v, dict):
        ts = v.get('Time_S')
        return list(ts) if ts is not None else []
    return []

MERGE_GAP = 50000
def _all_ts_max(Event):
    mx = 0
    for k in Event:
        if isinstance(k, str) and isinstance(Event[k], dict):
            for fld in ('Time_S', 'Time_E'):
                arr = Event[k].get(fld)
                if arr: mx = max(mx, max(arr))
    return mx
def _merge_events(events):
    merged = {}; offset = 0
    for Ev in events:
        seg_max = _all_ts_max(Ev)
        for k, v in Ev.items():
            if not isinstance(k, str) or not isinstance(v, dict): continue
            if k not in merged: merged[k] = {'Time_S': [], 'Time_E': []}
            ts = v.get('Time_S') or []; te = v.get('Time_E') or []
            merged[k]['Time_S'].extend([t + offset for t in ts])
            merged[k]['Time_E'].extend([t + offset for t in te])
        offset += seg_max + MERGE_GAP
    return merged

EMPTY_THRESH = 10
REWARD_CAP = 420
long_event=['MinT','Blackhole']
ignore_list=['MinS','CamT','LickL','LickR','tMod']

def prob_from_token(basename):
    m = re.match(r'^(\d{8})_(.+)_(\d{2})\.pkl$', basename)
    if not m: return None
    token = m.group(2)
    if '7' in token and '9' not in token: return '7:3'
    if '9' in token and '7' not in token: return '9:1'
    return None

# ====== 按 鼠×天 分组（复刻 cell[6]）======
root = Path("/mnt/HDD1/zyyk78/")
_grp = defaultdict(list)
for f in root.rglob("*.pkl"):
    m = re.search(r'(\d{8})', f.name)
    if not m: continue
    date = m.group(1)
    mouse = f.parent.name
    seq = re.search(r'_(\d{2})\.pkl$', f.name)
    seq = seq.group(1) if seq else '00'
    _grp[(mouse, date)].append((seq, str(f)))
matches = []
for (mouse, date), items in _grp.items():
    items.sort(key=lambda x: x[0])
    matches.append([p for _, p in items])

rows = []
n_sess = 0; n_skip_nomod = 0; n_skip_prob = 0; n_skip_dirty = 0; n_skip_empty = 0
mouse_set = set(); prob_counts = {}

for group_paths in matches:
    # 概率：用组内第一个文件名（同鼠同天同条件）
    prob = prob_from_token(os.path.basename(group_paths[0]))
    if prob is None:
        n_skip_prob += 1
        continue
    # 加载+脏检
    segments = []
    dirty = False
    for gp in group_paths:
        try:
            with open(gp,"rb") as pf: Event = pickle.load(pf)
        except: continue
        if not isinstance(Event, dict): continue
        need_keys = ['ModL','ModR','LickL','LickR','SlnL','SlnR','ROmL','ROmR']
        bad = [k for k in need_keys if Event.get(k) is not None and not isinstance(Event[k], dict)]
        if bad: dirty = True; continue
        segments.append((Event, gp))
    if dirty: n_skip_dirty += 1
    if not segments: continue
    # 空段过滤
    kept = []
    for Ev, gp in segments:
        n_rew = len(_time_s(Ev,'SlnL')) + len(_time_s(Ev,'SlnR'))
        eve_tmp = behave_ana.sort_events(Ev, ignore_list=ignore_list, long_event=long_event)
        n_trial = len(behave_ana.transform_sequence(eve_tmp))
        if n_trial <= EMPTY_THRESH or n_rew <= EMPTY_THRESH: continue
        kept.append((Ev, gp))
    if not kept: n_skip_empty += 1; continue
    # 合并
    Event = _merge_events([Ev for Ev,_ in kept]) if len(kept)>1 else kept[0][0]
    eve_list = behave_ana.sort_events(Event, ignore_list=ignore_list, long_event=long_event)
    # 420 截断
    reward_eves = [(t,e) for t,e in eve_list if e in ('SlnL','SlnR')]
    if len(reward_eves) > REWARD_CAP:
        cap_t = reward_eves[REWARD_CAP-1][0]
        eve_list = [(t,e) for t,e in eve_list if t <= cap_t]
    # 无 Mod → 跳过（无法定义 correct_side）
    if len(_time_s(Event,'ModL'))==0 and len(_time_s(Event,'ModR'))==0:
        n_skip_nomod += 1; continue
    n_sess += 1
    prob_counts[prob] = prob_counts.get(prob,0)+1
    mouse = os.path.basename(os.path.dirname(group_paths[0]))
    mouse_set.add(mouse)
    # transform → trial
    ts = behave_ana.transform_sequence(eve_list)
    ts = [t for t in ts if t[1] != 'E']  # 剔 Blackhole 无效
    for i in range(1, len(ts)):
        is_correct, correct_side, reward_or_not = ts[i]
        rows.append({
            'MouseID': mouse,
            'Trial': i+1,
            'ProbabilityCondition': prob,
            'ChoseHighProb': int(is_correct),
            'PreviousOutcome': int(ts[i-1][2]),
        })

df = pd.DataFrame(rows)
print(f"组(鼠×天)总数: {len(matches)}")
print(f"有效 session: {n_sess}  (7:3={prob_counts.get('7:3',0)}, 9:1={prob_counts.get('9:1',0)})")
print(f"剔除: 概率歧义={n_skip_prob} 脏={n_skip_dirty} 空段={n_skip_empty} 无Mod={n_skip_nomod}")
print(f"鼠数: {len(mouse_set)}  {sorted(mouse_set)}")
print(f"长表行数: {len(df)}")
print(f"\n=== 按条件 ===")
print(df.groupby('ProbabilityCondition').agg(trials=('ChoseHighProb','count'), chose_high_rate=('ChoseHighProb','mean'), prev_rew=('PreviousOutcome','mean')).round(3))
print(f"\n=== 按鼠×条件 chose_high_rate ===")
print(df.groupby(['MouseID','ProbabilityCondition'])['ChoseHighProb'].mean().round(3).unstack())
df.to_csv("/tmp/glme_trial_data.csv", index=False)
print("\n存 /tmp/glme_trial_data.csv")
