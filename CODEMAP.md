# Beh_Ana_J 代码地图（API 速查 + 数据流）

> 用途：浓缩 `behave_ana.py` / `fixing.py` / `ParseEvent.py` 的函数签名、Event 结构、数据流与已知坑，供后续对话直接引用，避免反复读源码。
> 源文件：`/home/jyy/Documents/Beh_Ana_J/`

---

## 1. 项目结构

| 文件 | 角色 |
|---|---|
| `1-Process.ipynb` | 主程序：evt → pkl，补全缺失起止时间 |
| `2-analysis.ipynb` | 主程序：从 pkl 提取/计算行为指标 |
| `3-main_AnaOfdata.ipynb` | 主程序：日志、显著性检验、回归 |
| `4-MS_data_proc.ipynb` | 主程序：贝叶斯解码器 |
| `ParseEvent.py` | evt 二进制 → Event dict（按 TTL bit 抽事件） |
| `fixing.py` | 事件序列补全、对齐、截断、丢帧修复 |
| `behave_ana.py` | 行为指标计算（正确率/奖励率/AUC/licking 等） |
| `cell_data.py` / `log_processer.py` / `process_data.py` / `training_convergence.py` | 辅助 |

pkl 数据根目录：`/mnt/HDD1/zyyk78/`（Stage3 / Stage4 / MPtest / llom_log），命名形如 `20250502_00.pkl`。

---

## 2. 数据流

```
evt(二进制, 每条12字节)
  └─ ParseEvent.read_event_file  → {vtTimestamps:[], TTLs:[]}
       └─ ParseEvent.extract_event(按 bit 边沿) → {event_name: {Time_S:[], Time_E:[]}}
            └─ ParseEvent.ParseEvent → Event dict
                 └─ fixing.fixing (补全/对齐/截断/丢帧) → pickle.dump → *.pkl
                      └─ 2-analysis.ipynb 加载 → behave_ana.* 计算指标 → data_collect → CSV
```

---

## 3. Event dict 结构

`Event` 是 `dict[str, dict]`，每个 value 形如 `{'Time_S': [...], 'Time_E': [...]}`，元素为 vtTimestamp（整数 tick）。

| key | bit | 含义 | 类型 |
|---|---|---|---|
| `CamT` | 1 | Basler 相机触发 | 短 |
| `LickL` | 3 | 左舔板 | 短 |
| `LickR` | 18 | 右舔板 | 短 |
| `SlnR` | 23 | 右电磁阀开（奖励 trial） | 短 |
| `SlnL` | 24 | 左电磁阀开（奖励 trial） | 短 |
| `ROmL` | 22 | 左奖励遗漏（无奖励 trial） | 短 |
| `ROmR` | 21 | 右奖励遗漏（无奖励 trial） | 短 |
| `ModL` | 17 | 高概率边切到左（correct side=L） | 短 |
| `ModR` | 12 | 高概率边切到右（correct side=R） | 短 |
| `MinS` | 9 | Miniscope 帧 | 短 |
| `MinT` | 15 | Miniscope 触发 | 短/长 |
| `Blackhole` | — | 屏蔽期（期间忽略选择），需完整 S/E | **长事件** |

> 注：`Blackhole` 不在 `ParseEvent` 默认 bit 表里，由 `fixing`/处理流程额外注入。

**语义约定**：
- `Sln*` = 正确选择且有奖励；`ROm*` = 无奖励（正确边但 omission，或错误边）。
- `ModL/ModR` 切换决定 `current_correct_side`，之后的 `Sln/ROm` 按是否等于该边判 `is_correct`。
- `Blackhole_S..E` 期间所有选择被忽略，并插入 `(False,'E',False)` 占位。

---

## 4. behave_ana.py 函数清单

### `sort_events(Event, ignore_list=[], long_event=[]) -> eve_list`
- 提取各事件 `Time_S`（长事件另取 `Time_E`），按 timestamp 排序。
- 返回 `[(timestamp, name), ...]`，name 形如 `'ModL'`、`'Blackhole_S'`、`'Blackhole_E'`。
- **防御**：跳过非字符串 key、value 非 dict、缺 `Time_S`/`Time_E` 的项。
- ⚠️ 不检查 `Time_S` 是否为 numpy array（但只迭代不取真值，故不报错）。

### `reward_rate(Event, window=(0,-1)) -> (L, R, T)`
- 左/右/总奖励比 = `Sln / (Sln+ROm)`（按边）。
- 内部 `import fixing`，若有 `Blackhole` 则按每个屏蔽区间 `truncate(..., 'inner')` 剔除，再按 `window` 截断。
- ⚠️ 第 48 行 `not Event['Blackhole']['Time_S']==[]` 对 array 取真值会 ambiguous（在 2-analysis 被 try 吞）。

### `transform_sequence(eve_list) -> [(is_correct, correct_side, reward_or_not), ...]`
- 把原始时间序列转成 trial 序列。`correct_side` ∈ `{'L','R','E'}`，`E` 表示 Blackhole 占位。
- 首个 Mod 之前若无 Mod，默认 `correct_side='L'`。

### `accuracy_rate(eve_list, window=(0,-1)) -> (L, R, T)` 或 `(None,None,None)`
- 基于 `transform_sequence`，按边统计 LC/LW/RC/RW 算正确率。
- `window[1] != -1` 且 trial 数不足时返回 `(None,None,None)`。

### `learning_auc(trial_sequence) -> float | np.nan`
- 累计正确率曲线下面积（归一化 x 到 [0,1]）。空序列返回 `np.nan`。

### `cumulative_attempt_auc(eve_list) -> float | np.nan`
- 仅取 `SlnL/SlnR/ROmL/ROmR`，累计尝试数 vs 归一化时间的 AUC。<2 次尝试返回 `np.nan`。

### `log_lr(eve_list, alpha=0.5) -> float`
- `log((n_left + alpha) / (n_right + alpha))`，left=`{SlnL,ROmL}`，right=`{SlnR,ROmR}`。

### `licking_rate(Event, eve_list) -> (avg_licks_reward, avg_licks_noreward, total_lick_events)`
- 在相邻 `Sln/ROm` 事件区间内用 `np.searchsorted` 数 `LickL+LickR` 次数，分 reward/noreward 求均值。
- `time_scale = 1000.0 if max(timestamps) > 100000 else 1.0`。
- ⚠️ 第 399/401 行 `... and Event['LickL']['Time_S']` 对 array 取真值会 ambiguous（**不在 try 里，会中断**）。

### 切换分析组
- `analyze_switching_patterns(trial_sequence) -> [switch_value, ...]`：正=切到正确，负=切到错误，绝对值=距上次切换的 trial 数。
- `get_switching_latency(eve_list) -> [[pattern], ...]`：按 Mod 阶段分段后各段切换模式。
- `get_switch_sequence(eve_list, lenth=5) -> [rewards_before_switch, ...]`：切换前 `lenth` 个 trial 的 reward 序列。
- `get_random_ns_sequence(eve_list, lenth=5, num_get=50) -> [non_switch_rewards, ...]`：非切换段同样序列，超量随机抽样。
- `switch_bayes_reg(sw_seq, ns_seq) -> (weights, bias, acc)`：sklearn LogisticRegression 分类切换/非切换。

---

## 5. fixing.py 函数清单（按调用频率排序）

### `truncate(Event_S, truncate_time, remove_region='outer') -> list`
- 截断时间序列。`truncate_time=[start, end]`，`-1` 表示忽略该端。
- `outer`：删两端（用 `pop(0)`/`pop(-1)`）；`inner`：删中间（用 `remove`）。
- ⚠️ 用 `pop`/`remove`，**只接受 list**；若 `Event_S` 是 numpy array 会 `AttributeError`。

### `fixing(Event, delta1=25000, delta2=25000, com_file_path=None, matched_row=None)`
- 主修复入口：识别 trial、对齐 hw/sw 时间、补缺失起止、截断无用信号、丢帧检测。
- 内部多处写回 `Event[i]['Time_S'] = truncate(...)`（保持 list）。

### 对齐 / 匹配组
- `clean_dtw_matches(hw_times_S, sw_times_S) -> (matched_hw, matched_sw)`：DTW 配对，返回 **np.array**。
- `align_time_sequences(hw_times_S, sw_times_S, max_ratio=1.1, max_abs_diff=None)`：时间序列对齐。
- `Event_compare(hw_times_S, sw_time_S)`：hw/sw 事件比较。
- `Snap_to_reference(Event_S, Event_ref_1_S, Event_ref_2_S=None, delta=25000)`：吸附到参考时间。

### trial 识别组
- `filtering_sig(Event_, min_duration=10)`：信号滤波。
- `recog_trials(Lick_S, delta=25000)`：识别 trial。
- `find_not_matched(Trials_S, Ref_trials_S, addtime=0, delta=25000)`：找未匹配 trial。
- `process_trials(Mod_ref_S, Cor_Sln_S, Inc_Sln_S, Inc_ROm_S, addtime=500)`：归类 trial。
- `find_missing_ranges(Event, com_Event, window_size=10, max_missing=1, checklist=None)`：丢帧范围检测。
- `bubbleSort(arr)`：冒泡排序（辅助）。

---

## 6. ParseEvent.py 函数清单

### `read_event_file(event_fn) -> {'vtTimestamps':[...], 'TTLs':[...]} | None`
- 读 evt 二进制：每条 12 字节（`QI` = 8 字节 timestamp + 4 字节 TTL），跳过首尾各 1 条头尾记录。
- `event_count = file_size // 12 - 2`。

### `extract_event(st_event_data, bits) -> {'Time_S':[...], 'Time_E':[...]}`
- `bits = [bit_pos, init_val]`。对 TTL 做 `bit >> bits[0] & 1` 取该 bit 序列，`np.diff` 求边沿。
- `init_val==0`：上升沿为 Start，下降沿为 End；`init_val==1` 反之。
- 返回的 `Time_S`/`Time_E` 是 **list**（推导式生成）。

### `ParseEvent(event_fn, st_events_bits=None) -> Event dict`
- 默认 bit 表见第 3 节。逐 bit 调 `extract_event` 组装 Event。

---

## 7. 2-analysis.ipynb 关键流程（cell 7）

```python
matches = [[str(f)] for f in filtered]   # 每个 pkl 包一层数组
for pkl_path in matches:
    Event = pickle.load(open(pkl_path[0],'rb'))
    eve_list = behave_ana.sort_events(Event,
                  ignore_list=['MinS','CamT','LickL','LickR','tMod'],
                  long_event=['MinT','Blackhole'])
    try:                              # ⚠️ 裸 except: pass 吞掉所有异常
        tot_rew = behave_ana.reward_rate(Event)[2]
        tca_auc = behave_ana.cumulative_attempt_auc(eve_list)
        log_lr  = behave_ana.log_lr(eve_list)
    except: pass
    if Event['ModL']['Time_S']==[] and Event['ModR']['Time_S']==[]:  # ⚠️ array 隐患
        tot_acc = 0
    else:
        tot_acc = behave_ana.accuracy_rate(eve_list)[2]
        lrn_auc = behave_ana.learning_auc(behave_ana.transform_sequence(eve_list))
        avg_*  = behave_ana.licking_rate(Event, eve_list)            # ⚠️ array 隐患
    data_collect.append({...})
```

指标字段：`date, tot_rew, tot_acc, lrn_auc, tca_auc, log_lr, eve_list, avg_licks_after_reward, avg_licks_after_noreward`。

---

## 8. 已知坑 & 报错定位

### 8.1 IndexError: ... are valid indices（**已确认根因，2026-07-03**）
- 报错行：cell 7（`f632f8c4`）第 39 行 `Event['ModL']['Time_S']`。
- 根因：该 pkl 的 `Event['ModL']`（或 `ModR`）**不是 dict，而是 numpy array**，用字符串 `'Time_S'` 索引 array → `IndexError: only integers, slices... are valid indices`。
  - dict 缺 `Time_S` → `KeyError`；list → `TypeError`；**只有 array 才是 IndexError**。
- 注意：`/mnt/HDD1/zyyk78/` 下 758 个 pkl 全部正常（`ModL` 均为 dict），故报错 pkl **不在该目录**，来自用户从 Excel 手动拷贝的 `matches`（旧版/异构 pkl，`ModL` 是裸 array）。
- `sort_events` 不报错：对 `ModL` 是 array 的情况，`isinstance(v, dict)` 为 False → `continue` 跳过，`eve_list` 里没有 ModL；但下游第 39 行**硬访问** `Event['ModL']['Time_S']` 无防御 → 炸。
- **防御加错位置**：`sort_events` 的 `isinstance(i, str)` / `isinstance(v, dict)` 防御挡的是「不爆炸的地方」，真正爆炸的第 39 行没防御。
- 同类隐患（`ModL`/`LickL`/`Blackhole` 是 array 时都会炸）：
  1. cell 7 第 39 行 `Event['ModL']['Time_S']==[]` — **不在 try，中断循环**（已触发）
  2. `licking_rate` 第 399/401 行 `... and Event['LickL']['Time_S']` — **不在 try，中断**
  3. `reward_rate` 第 48 行 `not Event['Blackhole']['Time_S']==[]` — 在 try，被吞
  4. 诊断 cell `3e1af5c2` 的 `Ev['ModL']['Time_S'] == []` 同样会炸

### 8.2 裸 except 掩盖问题
- `except: pass` 让 `reward_rate` 等异常静默，`tot_rew/tca_auc/log_lr` 默认 0，无法定位。
- 改 `except Exception as e: print(pkl_path[0], type(e).__name__, e)`。

### 8.3 truncate 只吃 list
- `fixing.truncate` 用 `pop`/`remove`，传 numpy array 会 `AttributeError`。

### 8.4 reward_rate 的 Blackhole 判断
- `if not Event['Blackhole']['Time_S']==[]` 语义等价 `!= []`，但 array 时会 ambiguous；改 `len(...) > 0`。

### 8.5 accuracy_rate window 不足返回 None
- `window[1] != -1` 且 trial 数不足时返回 `(None,None,None)`，下游 `tot_acc` 可能是 None 而非 0。

---

## 9. 修复 checklist

### 9.1 立即修复（让坏 pkl 不再中断循环）
- [ ] **cell 7（`f632f8c4`）第 39 行**：硬访问 `Event['ModL']['Time_S']` 无防御，`ModL` 是 array 时炸 IndexError。
  改用结构安全取值：
  ```python
  def _time_s(Event, key):
      """安全取 Event[key]['Time_S']，返回 list（异常/缺失返回 []）"""
      v = Event.get(key)
      if isinstance(v, dict):
          ts = v.get('Time_S')
          return list(ts) if ts is not None else []
      return []   # array / list / None 等一律当空
  ```
  第 39 行 → `if len(_time_s(Event,'ModL'))==0 and len(_time_s(Event,'ModR'))==0:`
- [ ] cell 7 第 36 行裸 `except: pass` → `except Exception as e: print(pkl_path[0], type(e).__name__, e)`，避免静默归零。
- [ ] `behave_ana.licking_rate` 第 399/401 行 `and Event['LickL']['Time_S']` → 用 `_time_s` 或 `len(...) > 0`。
- [ ] `behave_ana.reward_rate` 第 48 行 `not Event['Blackhole']['Time_S']==[]` → `len(Event.get('Blackhole',{}).get('Time_S',[])) > 0`。
- [ ] 诊断 cell `3e1af5c2` 同样套 `_time_s`，否则它会先于你定位就炸。

### 9.2 治本（找到源头）
- [ ] 找出 `matches` 里 `ModL` 是 array 的 pkl：扫一遍，打印 type 非 dict 的项。
- [ ] 溯源 `fixing.py`：`clean_dtw_matches` / `align_time_sequences` 返回 `np.array`，若被直接赋回 `Event['ModL'] = array`（而非 `Event['ModL']={'Time_S':array.tolist(),...}`）即污染。在落盘前 `.tolist()` 并包成 dict。
- [ ] `accuracy_rate` 返回 `(None,None,None)` 时下游 `tot_acc` 可能是 None，CSV 里要兜底成 0/NaN。

### 9.3 推荐的健壮版 cell 7 主循环骨架
```python
def _time_s(Event, key):
    v = Event.get(key)
    if isinstance(v, dict):
        ts = v.get('Time_S')
        return list(ts) if ts is not None else []
    return []

for pkl_path in matches:
    try:
        with open(pkl_path[0], "rb") as f:
            Event = pickle.load(f)
        long_event=['MinT','Blackhole']
        ignore_list=['MinS','CamT','LickL','LickR','tMod']
        eve_list = behave_ana.sort_events(Event, ignore_list=ignore_list, long_event=long_event)
        date_info = os.path.basename(pkl_path[0]).split('_')[0][:8]
        tot_rew = tca_auc = log_lr = tot_acc = lrn_auc = 0
        avg_licks_after_reward = avg_licks_after_noreward = 0
        try:
            _,_,tot_rew = behave_ana.reward_rate(Event)
            tca_auc = behave_ana.cumulative_attempt_auc(eve_list)
            log_lr  = behave_ana.log_lr(eve_list)
        except Exception as e:
            print(f'⚠️ {pkl_path[0]} 指标(reward/auc/lr)失败: {type(e).__name__}: {e}')
        if len(_time_s(Event,'ModL'))==0 and len(_time_s(Event,'ModR'))==0:
            tot_acc = 0
        else:
            _,_,tot_acc = behave_ana.accuracy_rate(eve_list)
            trial_sequence = behave_ana.transform_sequence(eve_list)
            lrn_auc = behave_ana.learning_auc(trial_sequence)
            avg_licks_after_reward, avg_licks_after_noreward, _ = behave_ana.licking_rate(Event, eve_list)
        data_collect.append({
            'date': date_info, 'tot_rew': tot_rew, 'tot_acc': tot_acc,
            'lrn_auc': lrn_auc, 'tca_auc': tca_auc, 'log_lr': log_lr,
            'eve_list': eve_list,
            'avg_licks_after_reward': avg_licks_after_reward,
            'avg_licks_after_noreward': avg_licks_after_noreward,
        })
    except Exception as e:
        print(f'💥 {pkl_path[0]} 整体跳过: {type(e).__name__}: {e}')
        continue
```
