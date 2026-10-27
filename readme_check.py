"""
README 数据体检: 在批量处理(evt -> pkl)之前, 先把 README.xlsx 的问题全部找出来。

检查四类问题:
  1. 缺失记录 : 目录里有 .evt 文件, 但 README.xlsx 里没有对应日期的行  -> 需要手工补录
  2. 多余记录 : README.xlsx 里有日期, 但目录里没有对应的 .evt 文件
  3. 小数日期 : README 第1列的日期带了小数(如 20260615.3333), 会被截断取整
  4. 模式为空 : README 第3列(模式)为空的行

另外: 已存在的 pkl 若是"没匹配到 README 行"时生成的, 会缺少 tMod(模式) 字段,
      补完 README 后需要删掉该 pkl 才会重新生成。

用法:
  命令行: python readme_check.py /mnt/HDD1/zyyk78/Stage4
  notebook: import readme_check; readme_check.check(dir)   # 结果直接打印在 cell 下方
"""

import os
import re
import pickle
import datetime
import pandas as pd

DATE_PATTERN = re.compile(r'(\d{8})')
MODE_PATTERN = re.compile(r'\d')          # 会话模式里一定带数字, 如 T7F / F9W / Alm50p9F


def is_session_mode(mode):
    """判断第3列是"会话模式"还是"备注"(初始体重/限食第一天/没做...)"""
    if not isinstance(mode, str):
        return False
    return bool(MODE_PATTERN.search(mode))


def size_str(nbytes):
    """把字节数转成易读格式"""
    if nbytes is None:
        return ''
    units = ['B', 'KB', 'MB', 'GB']
    v = float(nbytes)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f'{v:.1f} {u}' if u != 'B' else f'{int(v)} B'
        v /= 1024


def typo_hint(readme_date, evt_dates):
    """README 里某个日期在 evt 中找不到时, 猜测是否日期写错了"""
    hints = []
    for e in evt_dates:
        diff = abs(e - readme_date)
        if diff in (1, 2, 3):
            hints.append(f'可能是 {e} (差 {diff} 天)')
        elif e % 10000 == readme_date % 10000:          # 同月同日, 年份不同
            hints.append(f'可能是 {e} (年份差 {abs(e - readme_date) // 10000} 年)')
    # 月日数字互换, 如 20260631 <-> 20261306
    y, m, d = readme_date // 10000, readme_date % 10000 // 100, readme_date % 100
    if 1 <= d <= 12:
        swapped = y * 10000 + d * 100 + m
        if swapped in evt_dates:
            hints.append(f'可能是 {swapped} (月日写反了)')
    return '; '.join(sorted(set(hints)))


def read_readme(readme_path):
    """读取 README.xlsx

    Returns:
        (df, dates, frac_rows)
        df        : 处理后的表, df[0] 是 Int64 日期列
        dates     : 日期整数集合
        frac_rows : [(行号, 原始值, 数值)] 带小数的日期
    """
    df = pd.read_excel(readme_path, sheet_name='Sheet1', header=None).iloc[2:, :]
    raw = df[0]
    num = pd.to_numeric(raw, errors='coerce')
    # pandas 3.x 不允许 float->Int64 的非等价转换, 先向下取整(截断, 不能四舍五入)
    df[0] = (num // 1).astype('Int64')
    frac_rows = []
    raw_v = raw.to_numpy()
    num_v = num.to_numpy()
    for i in range(len(num_v)):
        if pd.notna(num_v[i]) and num_v[i] % 1 != 0:
            frac_rows.append((i + 3, raw_v[i], float(num_v[i])))   # +3: 表格前两行是表头
    dates = set(int(d) for d in df[0].dropna())
    return df, dates, frac_rows


def pkl_has_tmod(pkl_path):
    """检查已有 pkl 是否带 tMod(模式) 字段"""
    try:
        with open(pkl_path, 'rb') as f:
            event = pickle.load(f)
        return 'tMod' in event
    except Exception:
        return None      # 读不了, 未知


def check(parent_dir):
    """扫描 parent_dir 下所有子目录, 找出 README 的所有问题

    Returns:
        dict: {'missing': DataFrame, 'extra': DataFrame, 'frac': DataFrame,
               'no_readme': list, 'summary': DataFrame}
    """
    missing, extra, frac_rows, no_readme, summary = [], [], [], [], []

    for root, dirs, files in os.walk(parent_dir):
        evts = sorted(f for f in files if f.endswith('.evt'))
        if not evts:
            continue
        animal = os.path.basename(root)
        readme_path = os.path.join(root, 'README.xlsx')
        if not os.path.exists(readme_path):
            no_readme.append((root, len(evts)))
            summary.append({'子目录': animal, '路径': root, 'evt总数': len(evts),
                            '待处理(无pkl)': len(evts), '缺README': len(evts), '无README文件': True})
            continue

        df, dates, frac = read_readme(readme_path)
        evt_dates = set()
        # README 日期 -> (模式, 行号), 便于给缺失项提示"上一条记录"
        date_info = {}
        for i, d in enumerate(df[0]):
            if pd.isna(d):
                continue
            d = int(d)
            mode = df.iloc[i, 2] if df.shape[1] > 2 else ''
            date_info[d] = (mode, i + 3)
        readme_dates_sorted = sorted(date_info)

        for f in evts:
            m = DATE_PATTERN.match(f)
            if not m:
                missing.append({'子目录': animal, '日期': '', 'evt文件': f,
                                'evt文件大小': size_str(os.path.getsize(os.path.join(root, f))),
                                'pkl已存在': os.path.exists(os.path.join(root, os.path.splitext(f)[0] + '.pkl')),
                                'pkl缺tMod': None, '建议': '文件名不含日期, 需手工核对'})
                continue
            d = int(m.group(1))
            evt_dates.add(d)
            if d not in dates:
                pkl_path = os.path.join(root, os.path.splitext(f)[0] + '.pkl')
                pkl_exist = os.path.exists(pkl_path)
                no_tmod = pkl_has_tmod(pkl_path) if pkl_exist else None
                if pkl_exist and no_tmod is False:
                    advice = '补README后删除该pkl重跑(现pkl缺tMod)'
                elif pkl_exist:
                    advice = '补README后删除该pkl重跑'
                else:
                    advice = '补README后即可正常处理'
                earlier = [x for x in readme_dates_sorted if x < d]
                near = earlier[-1] if earlier else None
                near_txt = f'{near} {date_info[near][0]}' if near is not None else '(README 里没有更早的记录)'
                missing.append({'子目录': animal, '日期': d, 'evt文件': f,
                                'evt文件大小': size_str(os.path.getsize(os.path.join(root, f))),
                                'pkl已存在': pkl_exist, 'pkl缺tMod': no_tmod,
                                'README最近一条': near_txt, '建议': advice,
                                'pkl完整路径': pkl_path})

        # README 有但没有对应 evt
        for d in sorted(dates - evt_dates):
            row = df[df[0] == d]
            mode = row.iloc[0, 2] if len(row) > 0 and row.shape[1] > 2 else ''
            if not is_session_mode(mode):
                continue        # 备注行(初始体重/限食第一天/没做), 不是会话, 不用管
            hint = typo_hint(d, evt_dates)
            today = datetime.date.today()
            d_obj = datetime.date(d // 10000, d % 10000 // 100, d % 100)
            if d_obj >= today:
                kind, advice = '今天/未来, 正常', '尚未到实验日期, 没有evt正常'
            elif hint:
                kind, advice = '疑似笔误', hint
            else:
                kind, advice = '缺文件', 'README有此会话记录但无对应evt, 确认文件是否没拷过来'
            extra.append({'子目录': animal, '日期': d, 'README模式': mode,
                           '类别': kind, '疑似笔误': hint, '建议': advice})

        for excel_row, raw_val, num_val in frac:
            frac_rows.append({'子目录': animal, 'Excel行号': excel_row, '原始值': raw_val,
                              '截断后': int(num_val), '建议': '改成整数日期'})

        no_pkl = sum(1 for f in evts if not os.path.exists(os.path.join(root, os.path.splitext(f)[0] + '.pkl')))
        n_missing = sum(1 for m in missing if m['子目录'] == animal)
        summary.append({'子目录': animal, '路径': root, 'evt总数': len(evts),
                        '待处理(无pkl)': no_pkl, '缺README': n_missing, '无README文件': False})

    return {
        'missing': pd.DataFrame(missing),
        'extra': pd.DataFrame(extra),
        'frac': pd.DataFrame(frac_rows),
        'no_readme': no_readme,
        'summary': pd.DataFrame(summary),
    }


def show(res):
    """把体检结果打印出来"""
    print('=' * 78)
    print('README 数据体检报告')
    print('=' * 78)
    print('\n【1】各子目录概况')
    if not res['summary'].empty:
        print(res['summary'].to_string(index=False))

    print('\n【2】缺失记录: 目录里有 evt, 但 README 里查不到日期  (需手工补录)')
    if res['missing'].empty:
        print('  无')
    else:
        show_cols = [c for c in res['missing'].columns if c != 'pkl完整路径']
        print(res['missing'][show_cols].to_string(index=False))
        print(f'  合计 {len(res["missing"])} 条')

    print('\n【3】多余记录: README 里有日期, 但没有对应 evt 文件')
    if res['extra'].empty:
        print('  无')
    else:
        print(res['extra'].to_string(index=False))
        print(f'  合计 {len(res["extra"])} 条')

    print('\n【4】小数日期: 第1列日期带小数, 会被截断取整  (建议直接改成整数)')
    if res['frac'].empty:
        print('  无')
    else:
        print(res['frac'].to_string(index=False))

    print('\n【5】没有 README.xlsx 的目录')
    if not res['no_readme']:
        print('  无')
    else:
        for root, n in res['no_readme']:
            print(f'  {root}  ({n} 个 evt)')


def bad_pkls(res):
    """挑出"缺 tMod"的 pkl: 这些是在没匹配到 README 行时生成的, 补完 README 后应删除重跑"""
    if res['missing'].empty:
        return pd.DataFrame(columns=['pkl完整路径', '子目录', '日期', 'evt文件'])
    mask = (res['missing']['pkl已存在'] == True) & (res['missing']['pkl缺tMod'] == False)
    return res['missing'][mask][['pkl完整路径', '子目录', '日期', 'evt文件']]


def delete_bad_pkls(res, dry_run=True):
    """删除缺 tMod 的 pkl (dry_run=True 时只列不删)

    建议: 手工补全 README 之后再删, 否则这些会话会一直缺 tMod
    """
    bad = bad_pkls(res)
    if bad.empty:
        print('没有缺 tMod 的 pkl, 无需清理。')
        return []
    print(f'找到 {len(bad)} 个缺 tMod 的 pkl:')
    deleted = []
    for _, row in bad.iterrows():
        p = row['pkl完整路径']
        if dry_run:
            print(f'  [dry-run] 将删除: {p}')
        else:
            os.remove(p)
            deleted.append(p)
            print(f'  已删除: {p}')
    if not dry_run:
        print(f'共删除 {len(deleted)} 个 pkl, 重新跑批处理 cell 即可生成带 tMod 的新 pkl。')
    return deleted


if __name__ == '__main__':
    import sys
    parent = sys.argv[1] if len(sys.argv) > 1 else '/mnt/HDD1/zyyk78/Stage4'
    show(check(parent))
