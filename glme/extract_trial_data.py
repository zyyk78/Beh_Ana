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

# ====== 概率映射（高概率边 reward%）======
# 单数字 token: 7->7:3, 9->9:1, 8->8:2, 6->6:4, 5->5:5(无高低,剔除)
# 含 'p' = 可变范式(session 内比例变),剔除; 老格式无 token 剔除; 多数字歧义剔除
_PROB_MAP = {'7':'7:3', '9':'9:1', '8':'8:2', '6':'6:4'}

# ====== Treatment 派生(复刻 3-main BehaviorDataAnalyzer.extract_treatment) ======
# basename parts[1] = treatment token; Alm*/Arm* 合并成 m*, Arc/Alc→Ac
_ARM_ALM_MERGE = {
    'Alm50p9F':'m50p9F','Arm50p9F':'m50p9F', 'Alm50p7F':'m50p7F','Arm50p7F':'m50p7F',
    'Alm25p9F':'m25p9F','Arm25p9F':'m25p9F', 'Alm25p7F':'m25p7F','Arm25p7F':'m25p7F',
    'Alm12p9F':'m12p9F','Arm12p9F':'m12p9F', 'Alm12p7F':'m12p7F','Arm12p7F':'m12p7F',
    'Alm0p9F':'m0p9F','Arm0p9F':'m0p9F',     'Alm0p7F':'m0p7F','Arm0p7F':'m0p7F',
    'Arc7F':'Ac7F','Alc7F':'Ac7F',
}
def treat_from_token(basename):
    parts = basename.replace('.pkl','').split('_')
    if len(parts) < 3: return '00'
    t = parts[1]
    return _ARM_ALM_MERGE.get(t, t)

def prob_from_token(basename):
    m = re.match(r'^(\d{8})_(.+)_(\d{2})\.pkl$', basename)
    if not m: return None
    token = m.group(2)
    if 'p' in token: return None          # 可变范式
    nums = set(re.findall(r'\d', token))
    if len(nums) != 1: return None        # 多数字/无数字
    d = next(iter(nums))
    if d == '5': return None              # 5:5 无高低概率边
    return _PROB_MAP.get(d)               # 7/9/8/6

# 限食/限水：token 末字符 F=限食(Food) W=限水(Water)；老格式无 token 返回 None
def restriction_from_token(basename):
    m = re.match(r'^(\d{8})_(.+)_(\d{2})\.pkl$', basename)
    if not m: return None
    token = m.group(2)
    if 'p' in token: return None          # 可变范式，后缀语义不明
    suf = token[-1]
    return suf if suf in ('F','W') else None

# ====== 按 鼠×天 分组(复刻 cell[6])======
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

# ====== 第一遍: 收集 sessions ======
sessions = []   # (mouse, date, prob, treat, trial_rows)
n_skip_nomod = 0; n_skip_prob = 0; n_skip_dirty = 0; n_skip_empty = 0
prob_counts = {}

for group_paths in matches:
    prob = prob_from_token(os.path.basename(group_paths[0]))
    if prob is None:
        n_skip_prob += 1
        continue
    segments = []; dirty = False
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
    kept = []
    for Ev, gp in segments:
        n_rew = len(_time_s(Ev,'SlnL')) + len(_time_s(Ev,'SlnR'))
        eve_tmp = behave_ana.sort_events(Ev, ignore_list=ignore_list, long_event=long_event)
        n_trial = len(behave_ana.transform_sequence(eve_tmp))
        if n_trial <= EMPTY_THRESH or n_rew <= EMPTY_THRESH: continue
        kept.append((Ev, gp))
    if not kept: n_skip_empty += 1; continue
    Event = _merge_events([Ev for Ev,_ in kept]) if len(kept)>1 else kept[0][0]
    eve_list = behave_ana.sort_events(Event, ignore_list=ignore_list, long_event=long_event)
    reward_eves = [(t,e) for t,e in eve_list if e in ('SlnL','SlnR')]
    if len(reward_eves) > REWARD_CAP:
        cap_t = reward_eves[REWARD_CAP-1][0]
        eve_list = [(t,e) for t,e in eve_list if t <= cap_t]
    if len(_time_s(Event,'ModL'))==0 and len(_time_s(Event,'ModR'))==0:
        n_skip_nomod += 1; continue
    prob_counts[prob] = prob_counts.get(prob,0)+1
    mouse = os.path.basename(os.path.dirname(group_paths[0]))
    bname = os.path.basename(group_paths[0])
    date = re.search(r'(\d{8})', bname).group(1)
    treat = treat_from_token(bname)
    restr = restriction_from_token(bname)
    ts = behave_ana.transform_sequence(eve_list)
    ts = [t for t in ts if t[1] != 'E']   # 剔 Blackhole 无效
    trial_rows = []
    for i in range(1, len(ts)):           # 每 session 第1 trial 无前轮,剔除
        is_correct, correct_side, reward_or_not = ts[i]
        trial_rows.append({
            'ChoseHighProb': int(is_correct),
            'PreviousOutcome': int(ts[i-1][2]),
            'TrialInSession': i+1,
        })
    sessions.append((mouse, date, prob, treat, restr, trial_rows))

# ====== 第二遍: 按 mouse,date 排序, 分配全局累计 Trial + Session序号 ======
sessions.sort(key=lambda s: (s[0], s[1]))   # date=YYYYMMDD 字典序=时间序
rows = []
mouse_counter = defaultdict(int)
mouse_sess_counter = defaultdict(int)   # 每鼠有效训练日序号(第几次训练)
for mouse, date, prob, treat, restr, trial_rows in sessions:
    mouse_sess_counter[mouse] += 1
    sess_num = mouse_sess_counter[mouse]
    for r in trial_rows:
        mouse_counter[mouse] += 1
        tg = mouse_counter[mouse]
        r['MouseID'] = mouse
        r['Date'] = date
        r['Treatment'] = treat
        r['Restriction'] = restr              # F=限食 W=限水
        r['Trial'] = tg                      # 全局累计序号(跨 session, 按日历)
        r['TrialBin'] = (tg - 1) // 50       # 每50 trial 一箱
        r['SessionNum'] = sess_num           # 该鼠第几次训练(within-mouse训练经验)
        r['ProbabilityCondition'] = prob
        rows.append(r)

df = pd.DataFrame(rows)
# 合并 Age/Sex/Strain（mouse_inform.csv；缺失=NaN，complete-case 由 R 脚本过滤）
INFO_CSV = "/home/jyy/Documents/Beh_Ana_J/glme/mouse_inform.csv"
if os.path.exists(INFO_CSV):
    info = pd.read_csv(INFO_CSV)
    df = df.merge(info[['MouseID','Age','Sex','Strain']], on='MouseID', how='left')
    cc = df.dropna(subset=['Age','Sex','Strain'])
    print(f"合并 mouse_inform: {len(info)} 鼠，complete-case(Age+Sex+Strain非空) {cc.MouseID.nunique()} 鼠")
    if cc.MouseID.nunique() > 0:
        print(f"  complete-case 品系分布: {cc.groupby('MouseID')['Strain'].first().value_counts().to_dict()}")
else:
    print(f"⚠ 未找到 {INFO_CSV}，Age/Sex/Strain 列缺失（探索性分析无法跑）")
    df['Age'] = np.nan; df['Sex'] = np.nan; df['Strain'] = np.nan
mouse_set = sorted(df['MouseID'].unique())
print(f"组(鼠×天)总数: {len(matches)}")
print(f"有效 session: {len(sessions)}  概率分布: {prob_counts}")
print(f"剔除: 概率歧义/无token/可变/5:5={n_skip_prob} 脏={n_skip_dirty} 空段={n_skip_empty} 无Mod={n_skip_nomod}")
print(f"鼠数: {len(mouse_set)}  {mouse_set}")
print(f"长表行数: {len(df)}")
print(f"\n=== 按条件 ===")
print(df.groupby('ProbabilityCondition').agg(trials=('ChoseHighProb','count'), chose_high_rate=('ChoseHighProb','mean'), prev_rew=('PreviousOutcome','mean')).round(3))
print(f"\n=== 按 Restriction(限食F/限水W) ===")
print(df.groupby('Restriction').agg(trials=('ChoseHighProb','count'), chose_high_rate=('ChoseHighProb','mean')).round(3))
print(f"\n=== 按 SessionNum(第几次训练) 分箱 ===")
print(df.assign(SessBin=df.SessionNum//5).groupby('SessBin').agg(trials=('ChoseHighProb','count'), chose_high_rate=('ChoseHighProb','mean'), mean_sess=('SessionNum','mean')).round(3))
print(f"\n=== 按 Strain(品系) ===")
print(df.dropna(subset=['Strain']).groupby('Strain').agg(trials=('ChoseHighProb','count'), mice=('MouseID','nunique'), chose_high_rate=('ChoseHighProb','mean')).round(3))
print(f"\n=== 按 Treatment ===")
print(df.groupby(['Treatment','Restriction']).agg(trials=('ChoseHighProb','count')).to_string())
print(f"\n=== 按鼠×条件 chose_high_rate ===")
print(df.groupby(['MouseID','ProbabilityCondition'])['ChoseHighProb'].mean().round(3).unstack())
print(f"\n=== Trial 全局范围 ===")
print(df.groupby('MouseID')['Trial'].agg(['min','max','count']).head(10))

OUT = "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv"
df.to_csv(OUT, index=False)
print(f"\n存 {OUT}")
