"""
数据预处理脚本：分析beh_total.xlsx，添加浓度列
"""
import re
from collections import defaultdict

# 改用shell执行
import subprocess
result = subprocess.run(['which', 'python3'], capture_output=True, text=True)
print(f"Python: {result.stdout}")

# 使用conda的python
PYTHON = '/opt/anaconda3/bin/python3'

# ============== 以下是实际处理代码 ==============
import sys
sys.path.insert(0, '/opt/anaconda3/lib/python3.11/site-packages')

import pandas as pd

def parse_concentration(path):
    """从文件名解析浓度"""
    if not path:
        return None, None

    filename = path.split('/')[-1].replace('.pkl','')

    # Stage3: 纯牛奶 (无浓度)
    if '/Stage3/' in path:
        return 0, "0%"

    # Stage4: 检查文件名模式
    # AlmXXpYYF: XX=浓度(豆奶), YY=某种标识
    # ArmXXpYYF: XX=浓度(豆奶+抗), YY=某种标识
    # SS开头的模式
    if 'SS' in filename.upper():
        # SS7L12p -> Alm12p7F, SS9L50p -> Alm50p9F
        match = re.search(r'SS(\d+)([LR])(\d*)p', filename)
        if match:
            num = match.group(1)  # 7 or 9
            side = match.group(2)  # L or R
            conc = match.group(3) if match.group(3) else '0'

            # 根据用户要求: SS开头的改Alm，浓度就是数字
            new_name = f"Alm{conc}p{num}F"
            return int(conc) if conc else 0, f"{conc}%" if conc else "0%"

        # 可能的变体 SS7L -> Alm0p7F
        match2 = re.search(r'SS(\d+)([LR])_', filename)
        if match2:
            num = match2.group(1)
            conc = '0'
            new_name = f"Alm0p{num}F"
            return 0, "0%"

    # Alm模式: 直接取数字
    match = re.search(r'Alm(\d+)p', filename, re.IGNORECASE)
    if match:
        conc = match.group(1)
        return int(conc), f"{conc}%"

    # Arm模式
    match = re.search(r'Arm(\d+)p', filename, re.IGNORECASE)
    if match:
        conc = match.group(1)
        return int(conc), f"{conc}%"

    # T9F, T7F 等纯牛奶模式
    if filename.startswith('T'):
        return 0, "0%"

    # MPtest 的纯牛奶
    if '/MPtest/' in path:
        return 0, "0%"

    return None, None

def extract_animal_id(path):
    """从路径提取动物ID"""
    if not path:
        return None
    parts = path.split('/')
    # 找类似 A04, B02, C01 等的目录
    for p in parts:
        if re.match(r'^[ABCD]\d+$', p):
            return p
    return None

def process_data():
    """处理数据"""
    print("读取Excel文件...")
    header, data = read_excel()

    print(f"共 {len(data)} 行数据")

    # 分析文件名模式
    print("\n=== 分析原始文件名模式 ===")
    patterns = defaultdict(list)
    for row in data:
        path = row.get('A', '')
        filename = path.split('/')[-1] if path else ''
        patterns[filename[:20]].append(path)

    print("原始文件名前20个样本:")
    for p in list(patterns.keys())[:15]:
        print(f"  {p}")

    # 处理每行，添加浓度信息
    print("\n=== 添加浓度列 ===")
    results = []
    for row in data:
        path = row.get('A', '')
        conc_val, conc_group = parse_concentration(path)
        animal = extract_animal_id(path)

        new_row = dict(row)
        new_row['conc_num'] = conc_val
        new_row['conc_group'] = conc_group
        new_row['animal_id'] = animal
        results.append(new_row)

    # 统计浓度分布
    conc_dist = defaultdict(int)
    for r in results:
        conc_dist[r['conc_group']] += 1

    print("浓度分布:")
    for g in sorted(conc_dist.keys(), key=lambda x: int(x.replace('%','') or 0)):
        print(f"  {g}: {conc_dist[g]}")

    return results

def save_results(results):
    """保存结果到Excel"""
    print("\n保存到Excel...")

    # 使用pandas保存
    import pandas as pd
    df = pd.DataFrame(results)

    # 重新排序列
    cols = ['Train', 'Date', 'animal_id', 'conc_group', 'conc_num',
            'Reward Ratio', 'Accuracy', 'Switching Latency',
            'Incorrect Switch', 'MeanRewardInterval_s', 'Cumulative_Attempt_auc',
            'Learning_auc', 'log_lr', 'avg_licks_after_reward',
            'avg_licks_after_noreward']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"已保存: {OUTPUT_FILE}")

    return df

if __name__ == "__main__":
    results = process_data()
    df = save_results(results)

    print("\n=== 数据预览 ===")
    print(df.head(10))