import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (glmm)", "language": "python", "name": "glmm"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

# ===== Markdown: title =====
cells.append(nbf.v4.new_markdown_cell("""# 6-advanced_behavior.ipynb
## 高级行为学分析（Stage4, 2026-01-01 起, T7F / T9F）

**Phase 0** — Trial 级特征提取（从 172 个 pkl 生成 `trials_all.csv`）
**Phase 1** — 快收益分析：
1. PCA 行为流形
2. Lick 动力学（bout / 首lick延迟 / hazard）
3. 规则切换检测延迟

后续 Phase 2-4（Q-learning / HMM / 贝叶斯解码器 / GLMM）见同目录 `phase_plan.md`。"""))

# ===== Markdown: Phase 0 header =====
cells.append(nbf.v4.new_markdown_cell("""---
## Phase 0 · Trial 级特征提取

从 pkl 原始事件时间戳重建每个 trial：
- **choice**（L/R）、**outcome**（rewarded/omission）、**correct_side**（来自 Mod）
- **block_idx** + **trials_since_switch**（距上次 Mod 切换的 trial 数）
- **ITI_s**（trial 间隔秒数）、**lick_count_L/R**（trial 窗口内 lick 数）

输出：`trials_all.csv`（约 12 万行）+ `sessions_meta.csv`（每 session 汇总）"""))

# ===== Code: imports + config =====
cells.append(nbf.v4.new_code_cell("""import os, sys, pickle, glob, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.bbox'] = 'tight'

ROOT = '/home/jyy/Documents/Beh_Ana_J'
sys.path.insert(0, ROOT)
import behave_ana

DATA_DIRS = ['/mnt/HDD1/zyyk78/Stage4', os.path.join(ROOT, 'merged_pkls')]
FS = 50000.0
MERGE_GAP = 50000          # 同段合并偏移（与 2-analysis 一致）
REWARD_CAP = 420           # 与 2-analysis cell 7 一致
EMPTY_THRESH = 10          # 单段 trial/reward 低于此丢弃
SWITCH_CAP = 100           # winsorize 上限（与 cell 11 一致）
INCORRECT_CAP = 30
MRI_CAP = 30.0
OUT_DIR = os.path.join(ROOT, 'output', 'advanced')
os.makedirs(OUT_DIR, exist_ok=True)
print('Output dir:', OUT_DIR)"""))

# ===== Code: helpers =====
cells.append(nbf.v4.new_code_cell("""def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ts_of(Event, key):
    v = Event.get(key)
    if isinstance(v, dict):
        return list(v.get('Time_S') or [])
    return []

def merge_events(events):
    merged = {}
    offset = 0
    for Ev in events:
        seg_max = max((max((v.get('Time_S') or [0]) + (v.get('Time_E') or [0]))
                       for k, v in Ev.items() if isinstance(k, str) and isinstance(v, dict)),
                      default=0)
        for k, v in Ev.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            if k not in merged:
                merged[k] = {'Time_S': [], 'Time_E': []}
            merged[k]['Time_S'].extend([t + offset for t in (v.get('Time_S') or [])])
            merged[k]['Time_E'].extend([t + offset for t in (v.get('Time_E') or [])])
        offset += seg_max + MERGE_GAP
    return merged

LONG_EVENT = ['MinT', 'Blackhole']
IGNORE = ['MinS', 'CamT', 'LickL', 'LickR', 'tMod']

def scan_sessions():
    \"\"\"Scan DATA_DIRS, group same mouse/date, return list of (mouse, date, trt, [paths]) with
    merged_pkls overriding HDD paths.\"\"\"
    groups = {}
    for root in DATA_DIRS:
        for p in glob.glob(os.path.join(root, '*', '*.pkl')):
            base = os.path.basename(p)
            m = re.match(r'^(\\d{8})_(.+?)_(\\d+)(?:_merged)?\\.pkl$', base)
            if not m:
                continue
            mouse = os.path.basename(os.path.dirname(p))
            date, trt, _seg = m.groups()
            # merged_pkls 命名 20260608_00_01_merged → trt='00'，需重命名为含处理名后才能用
            if root.endswith('merged_pkls') and not re.match(r'^[A-Z]', trt):
                continue  # skip merged files whose treatment is segment number (known issue)
            key = (mouse, date)
            if key not in groups or root.endswith('merged_pkls'):
                groups[key] = {'mouse': mouse, 'date': date, 'trt': trt,
                               'paths': [], 'is_merged': root.endswith('merged_pkls')}
            groups[key]['paths'].append(p)
    return list(groups.values())

print('scan ready')"""))

# ===== Code: extract trials =====
cells.append(nbf.v4.new_markdown_cell("""### Phase 0 · 运行提取（约 1–2 分钟）"""))

cells.append(nbf.v4.new_code_cell("""def extract_session(sess):
    \"\"\"Extract trial-level features for one session group (same mouse+date).
    Follows 2-analysis semantics: merge same-date segments, cap rewards at 420,
    skip blackhole trials, drop sessions with no Mod.
    Returns (trials_df, session_meta) or (None, reason).\"\"\"
    paths = sorted(sess['paths'])
    events = []
    for p in paths:
        try:
            Ev = load_pkl(p)
        except Exception:
            return None, f'load_fail: {os.path.basename(p)}'
        if not isinstance(Ev, dict):
            return None, 'not_dict'
        need = ['ModL','ModR','SlnL','SlnR','ROmL','ROmR','LickL','LickR']
        if any(Ev.get(k) is not None and not isinstance(Ev[k], dict) for k in need):
            return None, 'dirty_keys'
        # NaN / constant timestamps -> broken
        all_ts = ts_of(Ev, 'SlnL') + ts_of(Ev, 'SlnR') + ts_of(Ev, 'ROmL') + ts_of(Ev, 'ROmR')
        if not all_ts:
            return None, 'no_trials'
        arr = np.array(all_ts, dtype=float)
        if np.any(np.isnan(arr)) or len(np.unique(arr)) < 2:
            return None, 'broken_ts'
        # empty segment filter
        n_rew = len(ts_of(Ev,'SlnL')) + len(ts_of(Ev,'SlnR'))
        if n_rew <= EMPTY_THRESH:
            continue
        events.append(Ev)
    if not events:
        return None, 'all_segments_empty'
    Event = merge_events(events) if len(events) > 1 else events[0]

    # 420-reward truncation (same semantics as 2-analysis)
    rew_evs = [(t, e) for k in ('SlnL','SlnR') for t in ts_of(Event, k) for e in [k]]
    if len(rew_evs) > REWARD_CAP:
        rew_evs.sort()
        cap_t = rew_evs[REWARD_CAP - 1][0]
        for k in ('SlnL','SlnR','ROmL','ROmR','LickL','LickR'):
            v = Event.get(k)
            if isinstance(v, dict) and v.get('Time_S'):
                v['Time_S'] = [t for t in v['Time_S'] if t <= cap_t]

    # Mod switch timeline
    mod_pts = sorted([(t, 'L') for t in ts_of(Event,'ModL')] + [(t, 'R') for t in ts_of(Event,'ModR')])
    if not mod_pts:
        return None, 'no_mod'
    # Build block boundaries: each Mod timestamp defines a new block starting at that time
    block_bounds = []
    for i, (t, side) in enumerate(mod_pts):
        end = mod_pts[i+1][0] if i+1 < len(mod_pts) else float('inf')
        block_bounds.append((t, end, side))

    def block_of(t):
        for i, (s, e, sd) in enumerate(block_bounds):
            if s <= t < e:
                return i, sd
        # before first Mod → no block (drop trial)
        return None, None

    # Trial stream: combine Sln/ROm into choice events, sort by time
    choice_events = []
    for k in ('SlnL','SlnR','ROmL','ROmR'):
        side = k[-1]
        rewarded = k.startswith('Sln')
        for t in ts_of(Event, k):
            choice_events.append((t, side, rewarded))
    choice_events.sort()
    if not choice_events:
        return None, 'no_choice_events'

    # Lick times
    lickL = ts_of(Event, 'LickL')
    lickR = ts_of(Event, 'LickR')

    rows = []
    for i, (t, side, rew) in enumerate(choice_events):
        b_idx, c_side = block_of(t)
        if c_side is None:
            continue
        prev_t = choice_events[i-1][0] if i > 0 else None
        iti = (t - prev_t) / FS if prev_t is not None else np.nan
        # licks between prev trial and this trial
        lo = prev_t if prev_t is not None else -np.inf
        n_ll = sum(1 for x in lickL if lo <= x < t)
        n_lr = sum(1 for x in lickR if lo <= x < t)
        rows.append({
            'choice': side, 'rewarded': int(rew), 'correct_side': c_side,
            'is_correct': int(side == c_side), 'block_idx': b_idx,
            'block_side': c_side, 'trial_time_s': t / FS, 'ITI_s': iti,
            'licks_L_prev_trial': n_ll, 'licks_R_prev_trial': n_lr,
        })
    if not rows:
        return None, 'no_valid_trials'
    df = pd.DataFrame(rows)
    df['trial_idx'] = np.arange(len(df))
    # trials_since_switch within block
    df['trials_since_switch'] = df.groupby('block_idx').cumcount()
    # block_side
    block_side_map = {i: sd for i, (s, e, sd) in enumerate(block_bounds)}
    df['block_side'] = df['block_idx'].map(block_side_map)
    df['correct_side'] = df['block_side']
    df['is_correct'] = (df['choice'] == df['correct_side']).astype(int)

    meta = {
        'Mouse_ID': sess['mouse'], 'Date': sess['date'], 'Treatment': sess['trt'],
        'n_trials': len(df), 'n_rewards': int(df['rewarded'].sum()),
        'n_omissions': int((1 - df['rewarded']).sum()),
        'n_blocks': int(df['block_idx'].nunique()),
        'n_mod_switches': len(mod_pts),
        'session_duration_s': float(df['trial_time_s'].iloc[-1]),
        'has_blackhole': bool(len(ts_of(Event,'Blackhole')) > 0),
        'is_merged': sess['is_merged'],
    }
    return df, meta

print('extract_session defined')"""))

# ===== Code: run extraction =====
cells.append(nbf.v4.new_code_cell("""%%time
all_sessions = scan_sessions()
print(f'Scanned {len(all_sessions)} session groups')

# filter to Stage4 2026+ T7F/T9F
targets = [s for s in all_sessions
           if s['trt'] in ('T7F', 'T9F') and s['date'] >= '20260101']
print(f'Stage4 2026+ T7F/T9F: {len(targets)} groups')

trial_rows, meta_rows, failed = [], [], []
for i, sess in enumerate(targets):
    df, info = extract_session(sess)
    if df is None:
        failed.append({'session': f"{sess['mouse']}_{sess['date']}", 'reason': info})
        continue
    df.insert(0, 'Mouse_ID', sess['mouse'])
    df.insert(1, 'Date', sess['date'])
    df.insert(2, 'Treatment', sess['trt'])
    trial_rows.append(df)
    meta_rows.append(info)

trials = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
sessions_meta = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()

print(f'\\nExtracted: {len(trials)} trials from {len(trial_rows)} sessions')
print(f'Failed: {len(failed)}')
if failed:
    print(pd.DataFrame(failed)['reason'].value_counts().to_string())
print('\\nPer treatment:')
print(sessions_meta.groupby('Treatment').agg(
    n_sessions=('n_trials','count'), n_mice=('Mouse_ID','nunique'),
    median_trials=('n_trials','median')).to_string())"""))

# ===== Code: save trials =====
cells.append(nbf.v4.new_code_cell("""trials.to_csv(os.path.join(OUT_DIR, 'trials_all.csv'), index=False)
sessions_meta.to_csv(os.path.join(OUT_DIR, 'sessions_meta.csv'), index=False)
print(f'Saved: {len(trials)} trials / {len(sessions_meta)} sessions')
trials.head()"""))

# ===== Markdown: Phase 1 =====
cells.append(nbf.v4.new_markdown_cell("""---
## Phase 1 · 快收益分析

### 1.1 PCA 行为流形
9 个 session 级指标 z-score 后投影到 PC1–PC2，按 T7F/T9F 着色；每只鼠用平均位置连线。**重复观测，PCA只做可视化，不做p值推断。**"""))

# ===== Code: PCA =====
cells.append(nbf.v4.new_code_cell("""# ---- PCA on session-level metrics ----
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATS = ['Accuracy','Learning_auc','log_lr',
         'avg_licks_after_reward','avg_licks_after_noreward',
         'Switching Latency','Incorrect Switch','MeanRewardInterval_s']

# Use trials to recompute session metrics (consistent with trial table, no blackhole)
sess_metrics = trials.groupby(['Mouse_ID','Date','Treatment']).agg(
    Accuracy=('is_correct','mean'),
    avg_licks_after_reward=('licks_L_prev_trial','mean'),  # placeholder, recompute below
).reset_index()

# Recompute all metrics properly
rows = []
for (mid, dt, trt), g in trials.groupby(['Mouse_ID','Date','Treatment'], sort=True):
    n_rew = g['rewarded'].sum()
    # avg licks after reward / noreward — use licks within trial window (prev trial -> current)
    rew_rows = g[g['rewarded']==1]
    norew_rows = g[g['rewarded']==0]
    rows.append({
        'Mouse_ID': mid, 'Date': dt, 'Treatment': trt,
        'Accuracy': g['is_correct'].mean(),
        'Reward_Rate': n_rew / len(g),
        'Learning_auc_proxy': g['is_correct'].expanding().mean().iloc[-1],
        'avg_licks_after_reward': (rew_rows['licks_L_prev_trial'].mean() if len(rew_rows) else 0),
        'avg_licks_after_noreward': (norew_rows['licks_L_prev_trial'].mean() if len(norew_rows) else 0),
        'MeanRewardInterval_s': g['ITI_s'].median(),
        'n_trials': len(g),
    })
sess_feat = pd.DataFrame(rows)

# Apply winsorize caps consistent with cell 11
sess_feat['MeanRewardInterval_s'] = sess_feat['MeanRewardInterval_s'].clip(upper=MRI_CAP)

X = StandardScaler().fit_transform(sess_feat[FEATS].fillna(0))
X[:, FEATS.index('Switching Latency')] = np.clip(X[:, FEATS.index('Switching Latency')], None, SWITCH_CAP)

pcs = PCA(n_components=2).fit_transform(X)
sess_feat['PC1'] = pcs[:, 0]
sess_feat['PC2'] = pcs[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
colors = {'T7F': '#2E7EB8', 'T9F': '#E07B39'}
for trt, c in colors.items():
    sub = sess_feat[sess_feat['Treatment']==trt]
    ax.scatter(sub['PC1'], sub['PC2'], s=45, alpha=0.7, c=c, label=trt, edgecolors='white', lw=0.5)
    # per-mouse centroid lines
    for mid, mg in sub.groupby('Mouse_ID'):
        cen = mg[['PC1','PC2']].mean()
        ax.plot([cen['PC1']]*2, [cen['PC2']]*2, color=c, alpha=0.25, lw=1)
        ax.scatter(cen['PC1'], cen['PC2'], s=130, c=c, marker='s', edgecolors='k', lw=0.8, zorder=5)
        ax.annotate(mid, (cen['PC1'], cen['PC2']), fontsize=7, color=c,
                    xytext=(4,4), textcoords='offset points')
ax.set_xlabel(f'PC1 ({PCA(n_components=2).fit(X).explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({PCA(n_components=2).fit(X).explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Behavioral manifold: T7F vs T9F (Stage4, 2026+)')
ax.legend(title='Treatment')
ax.axhline(0, color='grey', lw=0.4, ls=':')
ax.axvline(0, color='grey', lw=0.4, ls=':')
plt.tight_layout()
display(fig)
plt.savefig(os.path.join(OUT_DIR, 'pca_manifold.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'pca_manifold.svg'), dpi=300)
print('PCA saved to', OUT_DIR)"""))

# ===== Markdown: lick dynamics =====
cells.append(nbf.v4.new_markdown_cell("""### 1.2 Lick 动力学
- **Bout 分割**：lick 间隔 > 0.5 s 算新 bout
- **奖励后 vs 落空后首个 lick 延迟**分布
- **Lick rate hazard**：奖励后/落空后 lick 概率随时间变化（0–3 s, 50 ms bins）"""))

# ===== Code: lick dynamics =====
cells.append(nbf.v4.new_code_cell("""# ---- Lick dynamics from raw event times ----
BOUT_GAP = 0.5   # s, > this = new bout
HAZ_BINS = np.arange(0, 3.01, 0.05)

def lick_dynamics_for_session(paths):
    \"\"\"Returns dict with bout stats, first-lick latency, hazard curves.\"\"\"
    events = []
    for p in paths:
        Ev = load_pkl(p)
        if isinstance(Ev, dict):
            events.append(Ev)
    if not events:
        return None
    Event = merge_events(events) if len(events) > 1 else events[0]
    lickL = np.array(ts_of(Event,'LickL'), dtype=float) / FS
    lickR = np.array(ts_of(Event,'LickR'), dtype=float) / FS
    slnL = np.array(ts_of(Event,'SlnL'), dtype=float) / FS
    slnR = np.array(ts_of(Event,'SlnR'), dtype=float) / FS
    romL = np.array(ts_of(Event,'ROmL'), dtype=float) / FS
    romR = np.array(ts_of(Event,'ROmR'), dtype=float) / FS
    all_licks = np.sort(np.concatenate([lickL, lickR]))
    all_rew = np.sort(np.concatenate([slnL, slnR]))
    all_om = np.sort(np.concatenate([romL, romR]))
    if len(all_licks) < 100:
        return None
    # bout segmentation
    ili = np.diff(all_licks)
    bout_starts = np.concatenate([[0], np.where(ili > BOUT_GAP)[0] + 1])
    bout_ends = np.concatenate([np.where(ili > BOUT_GAP)[0], [len(all_licks)-1]])
    n_bouts = len(bout_starts)
    bout_lens = (bout_ends - bout_starts + 1)
    bout_durs = all_licks[bout_ends] - all_licks[bout_starts]
    # first lick after reward / omission (within 3 s window)
    def first_lick_latency(anchor_times, window=3.0):
        lat = []
        for t in anchor_times:
            idx = np.searchsorted(all_licks, t, side='left')
            if idx < len(all_licks) and all_licks[idx] - t <= window:
                lat.append(all_licks[idx] - t)
        return np.array(lat)
    lat_rew = first_lick_latency(all_rew)
    lat_om = first_lick_latency(all_om)
    # hazard: P(lick occurs in bin | no lick yet since anchor)
    def lick_hazard(anchor_times, bins=HAZ_BINS):
        counts = np.zeros(len(bins)-1)
        for t in anchor_times:
            idx = np.searchsorted(all_licks, t + bins[0], side='left')
            end = np.searchsorted(all_licks, t + bins[-1], side='right')
            for j in range(len(bins)-1):
                lo, hi = t + bins[j], t + bins[j+1]
                c = np.searchsorted(all_licks[idx:end], hi, side='left') - np.searchsorted(all_licks[idx:end], lo, side='left')
                counts[j] += (c > 0)
        return counts / max(len(anchor_times), 1)
    haz_rew = lick_hazard(all_rew)
    haz_om = lick_hazard(all_om)
    return {
        'n_licks': len(all_licks), 'n_bouts': n_bouts,
        'median_bout_len': float(np.median(bout_lens)),
        'median_bout_dur_s': float(np.median(bout_durs)),
        'median_lat_rew_s': float(np.median(lat_rew)) if len(lat_rew) else np.nan,
        'median_lat_om_s': float(np.median(lat_om)) if len(lat_om) else np.nan,
        'haz_rew': haz_rew, 'haz_om': haz_om,
        'n_rew': len(all_rew), 'n_om': len(all_om),
    }

print('lick_dynamics_for_session defined')"""))

cells.append(nbf.v4.new_code_cell("""%%time
lick_rows, haz_rew_all, haz_om_all = [], [], []
for sess in targets:
    r = lick_dynamics_for_session(sess['paths'])
    if r is None:
        continue
    haz_rew_all.append(r.pop('haz_rew'))
    haz_om_all.append(r.pop('haz_om'))
    r.update({'Mouse_ID': sess['mouse'], 'Date': sess['date'], 'Treatment': sess['trt']})
    lick_rows.append(r)

lick_df = pd.DataFrame(lick_rows)
H_rew = np.array(haz_rew_all)   # (n_sessions, n_bins)
H_om = np.array(haz_om_all)
print(f'Lick dynamics computed for {len(lick_df)} sessions')
print(lick_df.groupby('Treatment')[['median_bout_len','median_bout_dur_s','median_lat_rew_s','median_lat_om_s']].median().round(3).to_string())"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) hazard curves
ax = axes[0]
for trt, c in colors.items():
    mask = lick_df['Treatment']==trt
    ax.plot(HAZ_BINS[:-1], H_rew[mask.values].mean(axis=0), color=c, lw=2, label=f'{trt} after reward')
    ax.plot(HAZ_BINS[:-1], H_om[mask.values].mean(axis=0), color=c, lw=2, ls='--', label=f'{trt} after omission')
ax.set_xlabel('Time after event (s)'); ax.set_ylabel('P(lick in bin)')
ax.set_title('Lick hazard'); ax.legend(fontsize=8)

# (b) first-lick latency distributions
ax = axes[1]
data_box = [lick_df.loc[lick_df['Treatment']==t, 'median_lat_rew_s'].dropna() for t in colors]
data_box += [lick_df.loc[lick_df['Treatment']==t, 'median_lat_om_s'].dropna() for t in colors]
bp = ax.boxplot(data_box, labels=['T7F\\nrew','T9F\\nrew','T7F\\nom','T9F\\nom'],
                patch_artist=True, widths=0.55)
for patch, col in zip(bp['boxes'], ['#2E7EB8','#E07B39','#7FBFDE','#F0B27A']):
    patch.set_facecolor(col)
ax.set_ylabel('Median first-lick latency (s)'); ax.set_title('First-lick latency')

# (c) bout stats
ax = axes[2]
for trt, c in colors.items():
    sub = lick_df[lick_df['Treatment']==trt]
    ax.scatter(sub['median_bout_dur_s'], sub['median_bout_len'], c=c, s=50, alpha=0.7, label=trt, edgecolors='white')
ax.set_xlabel('Median bout duration (s)'); ax.set_ylabel('Median bout length (licks)')
ax.set_title('Bout structure'); ax.legend()

plt.tight_layout()
display(fig)
plt.savefig(os.path.join(OUT_DIR, 'lick_dynamics.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'lick_dynamics.svg'), dpi=300)
print('Lick dynamics figure saved')"""))

# ===== Markdown: switch detection =====
cells.append(nbf.v4.new_markdown_cell("""### 1.3 规则切换检测延迟
每次 Mod 切换后，小鼠滚动准确率（窗口 20 trial）需要几个 trial 才持续偏向新规则。
- **检测延迟** = 从切换点到"滚动准确率显著偏向新侧"的 trial 数
- 组间分布比较"""))

# ===== Code: switch detection =====
cells.append(nbf.v4.new_code_cell("""WINDOW = 20   # rolling accuracy window
THRESH = 0.65 # sustained accuracy above this in new-direction = detected
MIN_STREAK = 5  # must hold for this many trials

def switch_detection_delay(df_sess):
    \"\"\"df_sess = trials of ONE session. Returns list of (block_idx, delay_in_trials).\"\"\"
    delays = []
    for b_idx, bg in df_sess.groupby('block_idx'):
        if b_idx == 0:
            continue  # first block has no preceding switch
        new_side = bg['block_side'].iloc[0]
        correct_col = (bg['choice'] == new_side).astype(int).values
        roll = pd.Series(correct_col).rolling(WINDOW, min_periods=WINDOW).mean()
        # find first trial where rolling acc >= THRESH for MIN_STREAK consecutive trials
        above = roll >= THRESH
        delay = np.nan
        streak = 0
        for i, a in enumerate(above):
            streak = streak + 1 if a else 0
            if streak >= MIN_STREAK:
                delay = i - MIN_STREAK + 1  # trial index within block when detection happened
                break
        delays.append({'block_idx': b_idx, 'delay_trials': delay,
                       'block_trials': len(bg)})
    return pd.DataFrame(delays)

all_delays = []
for (mid, dt, trt), g in trials.groupby(['Mouse_ID','Date','Treatment']):
    d = switch_detection_delay(g)
    d['Mouse_ID'] = mid; d['Date'] = dt; d['Treatment'] = trt
    all_delays.append(d)
delays_df = pd.concat(all_delays, ignore_index=True)

print(f'Total switch events analyzed: {len(delays_df)}')
print(f'Detected (delay not NaN): {delays_df["delay_trials"].notna().sum()} ({delays_df["delay_trials"].notna().mean()*100:.1f}%)')
print('\\nDelay distribution by treatment:')
print(delays_df.groupby('Treatment')['delay_trials'].describe().round(1).to_string())"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) delay distribution
ax = axes[0]
for trt, c in colors.items():
    d = delays_df.loc[delays_df['Treatment']==trt, 'delay_trials'].dropna()
    ax.hist(d, bins=np.arange(0, 105, 5), alpha=0.55, color=c, label=f'{trt} (n={len(d)})', density=True)
ax.axvline(delays_df['delay_trials'].median(), color='k', ls=':', lw=1.2, label=f'median={delays_df["delay_trials"].median():.0f}')
ax.set_xlabel('Trials to detect switch'); ax.set_ylabel('Density')
ax.set_title('Rule-switch detection delay'); ax.legend(fontsize=9)

# (b) per-mouse medians
ax = axes[1]
mouse_med = delays_df.groupby(['Mouse_ID','Treatment'])['delay_trials'].median().reset_index()
for trt, c in colors.items():
    sub = mouse_med[mouse_med['Treatment']==trt].sort_values('Mouse_ID')
    ax.scatter(sub['Mouse_ID'], sub['delay_trials'], c=c, s=90, label=trt, edgecolors='k', lw=0.6, zorder=3)
ax.set_ylabel('Median detection delay (trials)'); ax.set_title('Per-mouse switch detection')
ax.tick_params(axis='x', rotation=45); ax.legend()

plt.tight_layout()
display(fig)
plt.savefig(os.path.join(OUT_DIR, 'switch_delay.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'switch_delay.svg'), dpi=300)
print('Switch-delay figure saved')"""))

# ===== Final summary =====
cells.append(nbf.v4.new_markdown_cell("""---
## Phase 0 + 1 完成

产出文件（`output/advanced/`）：
- `trials_all.csv` — trial 级主表
- `sessions_meta.csv` — session 汇总
- `pca_manifold.png/svg` — 行为流形
- `lick_dynamics.png/svg` — lick 动力学
- `switch_delay.png/svg` — 规则切换延迟

后续 Phase 2（Q-learning）、Phase 3（HMM）、Phase 4（贝叶斯解码器 + GLMM）在 `phase_plan.md` 里，按此 notebook 末尾继续追加 cells。"""))

nb.cells = cells
with open(os.path.join(ROOT, '6-advanced_behavior.ipynb'), 'w') as f:
    nbf.write(nb, f)
print('notebook written:', os.path.join(ROOT, '6-advanced_behavior.ipynb'))
