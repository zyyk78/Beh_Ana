#!/usr/bin/env python3
# 给已存在的 glme_trial_data.csv 补 Treatment 列(无需重跑 extract 读 pkl)
# Treatment = pkl basename parts[1], Alm*/Arm* 合并成 m*(同 extract 脚本)
import re, sys, pandas as pd
from pathlib import Path
from collections import defaultdict

CSV = "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv"
ROOT = Path("/mnt/HDD1/zyyk78/")

_ARM_ALM_MERGE = {
    'Alm50p9F':'m50p9F','Arm50p9F':'m50p9F', 'Alm50p7F':'m50p7F','Arm50p7F':'m50p7F',
    'Alm25p9F':'m25p9F','Arm25p9F':'m25p9F', 'Alm25p7F':'m25p7F','Arm25p7F':'m25p7F',
    'Alm12p9F':'m12p9F','Arm12p9F':'m12p9F', 'Alm12p7F':'m12p7F','Arm12p7F':'m12p7F',
    'Alm0p9F':'m0p9F','Arm0p9F':'m0p9F',     'Alm0p7F':'m0p7F','Arm0p7F':'m0p7F',
    'Arc7F':'Ac7F','Alc7F':'Ac7F',
}
def treat(bname):
    parts = bname.replace('.pkl','').split('_')
    if len(parts) < 3: return '00'
    return _ARM_ALM_MERGE.get(parts[1], parts[1])

# 扫文件名建 (mouse,date) -> treatment 映射(不读 pkl, 秒级)
print("扫描 pkl 文件名建 (MouseID,Date)->Treatment 映射…")
mapping = {}
n_files = 0
for f in ROOT.rglob("*.pkl"):
    m = re.search(r'(\d{8})', f.name)
    if not m: continue
    n_files += 1
    key = (f.parent.name, m.group(1))
    mapping.setdefault(key, set()).add(treat(f.name))
# 每组应唯一(已验)
conflict = {k:v for k,v in mapping.items() if len(v) > 1}
print(f"扫到 {n_files} pkl, {len(mapping)} (mouse,date) 组, 冲突组={len(conflict)}")
if conflict:
    print("WARN 冲突:", list(conflict.items())[:5])

# 退化为一对一(冲突取多数,实际0冲突)
key2treat = {k: next(iter(v)) for k,v in mapping.items()}

df = pd.read_csv(CSV)
print(f"\n读 CSV: {len(df)} rows, 列={list(df.columns)}")
if 'Treatment' in df.columns:
    print("已有 Treatment 列, 覆盖更新")

df['Date_str'] = df['Date'].astype(str)
df['Treatment'] = df.apply(lambda r: key2treat.get((r['MouseID'], r['Date_str']), 'UNKNOWN'), axis=1)
df.drop(columns=['Date_str'], inplace=True)

unk = (df['Treatment'] == 'UNKNOWN').sum()
print(f"\nTreatment 补列完成: {len(df)} rows, UNKNOWN={unk}")
print("分布:")
print(df['Treatment'].value_counts())

# Treatment 紧贴 Date 后(顺序美观)
cols = list(df.columns)
cols.remove('Treatment')
ins = cols.index('Date') + 1
cols.insert(ins, 'Treatment')
df = df[cols]

df.to_csv(CSV, index=False)
print(f"\n写回 {CSV}, 列={list(df.columns)}")
