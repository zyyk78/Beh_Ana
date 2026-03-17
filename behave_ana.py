import numpy as np
def sort_events(Event,ignore_list=[],long_event=[]):
    """提取事件的开始时间并进行排序

    Args:
        Event (dict): 完整的事件列表
        ignore_list (list, optional): 需要忽略的关键字,其中如果某个关键字下面没有Time_S和Time_E同样会被自动忽略. Defaults to None.
        long_event (list, optional): "长事件"列表(需要完整记录开始和结束时间,Event_S和Event_E标记),否则只记录开始时间
    Returns:
        eve_list(list): 返回排序好的事件list,其中包括[timestamp,事件名称]
    """    
    eve_list=[]
    for i in Event:
        if not i in ignore_list:
            if 'Time_S' in Event[i] and 'Time_E' in Event[i] :
                if i in long_event:
                    for timestamp in Event[i]['Time_S']:
                        eve_list.append((timestamp,i+'_S'))
                    for timestamp in Event[i]['Time_E']:
                        eve_list.append((timestamp,i+'_E'))    
                else:
                    for timestamp in Event[i]['Time_S']:
                        eve_list.append((timestamp,i))    
    eve_list.sort(key=lambda x: x[0])
    
    return eve_list

def reward_rate(Event,window=(0,-1)):
    """计算两边和总体的奖励比例(如果Event中出现了Blackhol关键字则会自动应用)

    Args:
        Event (dict): 完整的Event
        window(list): 时间窗口
    Returns:
        tuple(int,int,int): 左,右,总体的奖励比
    """    
    import fixing
    cal_list=['SlnL','SlnR','ROmL','ROmR']
    Event_trunc={}
    if 'Blackhole' in Event:
        if not Event['Blackhole']['Time_S']==[]:
            for i in cal_list:
                Event_trunc[i]=Event[i]['Time_S']
                for j in range(len(Event['Blackhole']['Time_S'])):
                    Event_trunc[i]=fixing.truncate(Event_trunc[i],(Event['Blackhole']['Time_S'][j],Event['Blackhole']['Time_E'][j]),'inner')
                Event_trunc[i]=fixing.truncate(Event_trunc[i],window)
        else:
            for i in cal_list:
                Event_trunc[i]=Event[i]['Time_S']
                Event_trunc[i]=fixing.truncate(Event_trunc[i],window)
    else:
        for i in cal_list:
            Event_trunc[i]=Event[i]['Time_S']
            Event_trunc[i]=fixing.truncate(Event_trunc[i],window)
            
    L=len(Event_trunc['SlnL'])/(len(Event_trunc['ROmL'])+len(Event_trunc['SlnL'])) if (len(Event_trunc['ROmL'])+len(Event_trunc['SlnL'])) >0 else 0
    R=len(Event_trunc['SlnR'])/(len(Event_trunc['ROmR'])+len(Event_trunc['SlnR'])) if (len(Event_trunc['ROmR'])+len(Event_trunc['SlnR'])) >0 else 0
    T=(len(Event_trunc['SlnR'])+len(Event_trunc['SlnL']))/(len(Event_trunc['ROmL'])+len(Event_trunc['SlnL'])+len(Event_trunc['ROmR'])+len(Event_trunc['SlnR']))
    return L,R,T

def transform_sequence(eve_list):
    """将原始序列转换为选择结果序列
    
    Args:
        eve_list: 原始序列，格式为[[TimeStamp, Event], ...]
    Returns:
        list: 转换后的序列，每个元素为(is_correct, correct_side,reward_or_not)
    """
    trial_sequence = []
    current_correct_side = None
    in_blackhole = False
    find_mod=False
    for _,event in eve_list:
        if event=='ModL' or event=='ModR':
            find_mod=True
            break
    if not find_mod:
        current_correct_side='L'
    for ts,event in eve_list:
        if event == 'Blackhole_S':
            in_blackhole = True
            current_correct_side=None
            trial_sequence.append((ts,False,'E',False))
            continue
        
        if event == 'Blackhole_E':
            in_blackhole = False
            continue
        
        if in_blackhole:
            continue
        
        if event.startswith('Mod'):
            current_correct_side = event[-1]  # 'L' 或 'R'
            continue
        
        if current_correct_side is None:
            continue  # 忽略没有Mod前的选择事件
        
        if event.startswith('Sln') or event.startswith('ROm'):
            choice = event[-1]  # 'L' 或 'R'
            result = True if event.startswith('Sln') else False
            is_correct = (choice == current_correct_side)
            trial_sequence.append((ts,is_correct, current_correct_side,result))
    
    return trial_sequence

def accuracy_rate(eve_list,window=(0,-1)):
    """计算两边和总体的正确比例(如果Event中出现了Blackhol关键字则会自动应用)
    
    Args:
        eve_list: 排好的时间序列
        window(list): trial数窗口
    Returns:
        tuple(int,int,int):左右总体的正确比例
    """
    LC, LW, RC, RW = 0, 0, 0, 0
    trial_sequence=transform_sequence(eve_list)  #(is_correct, correct_side,reward_or_not)
    count=0
    if window[1]==-1:
        trial_sequence=[trial_sequence[i] for i in range(window[0],len(trial_sequence))]
    else:
        trial_sequence=[trial_sequence[i] for i in range(window[0],min(len(trial_sequence),window[1]))]
    for _,is_correct, correct_side,_ in trial_sequence:
        if correct_side == 'L':
            if is_correct:
                LC += 1
            else:
                LW += 1
        else:  # 'R'
            if is_correct:
                RC += 1
            else:
                RW += 1
    if window[1] != -1:
        if len(trial_sequence) < window[1]-window[0]:
            return None,None,None
    # 计算正确率
    L = LC / (LC + LW) if (LC + LW) > 0 else 0
    R = RC / (RC + RW) if (RC + RW) > 0 else 0
    T = (LC + RC) / (LC + LW + RC + RW) if (LC + LW + RC + RW) > 0 else 0
    
    return L,R,T

def analyze_switching_patterns(trial_sequence):
    """分析模式切换模式（基于转换后的序列）
    
    Args:
        trial_sequence: 转换后的序列，格式为[[is_correct, correct_side], ...]
    
    Returns:
        list: 切换模式列表，每个元素形如 [5, -7, 20],(正数表示正确切换，负数表示错误切换，数值表示从上次切换mod之后的trial次数)
    """
    if not trial_sequence:
        return []
    
    switch_pattern = []
    count = 0
    current_state=False #默认是错误边开始
    for _,is_correct, _,_ in trial_sequence:
        count += 1
        # 检测模式切换
        if is_correct != current_state:
            # 确定切换方向
            switch_value = count if is_correct  else -count
            switch_pattern.append(switch_value)
            
            current_state = is_correct
    
    return switch_pattern

def get_switching_latency(eve_list):
    """获取所有Mod阶段的切换模式
    
    Args:
        eve_list: 原始序列，格式为[[TimeStamp, Event], ...]
    
    Returns:
        list: 包含所有切换模式的列表，每个元素是一个切换模式列表
    """
    # 首先转换整个序列
    transformed = transform_sequence(eve_list)
    # 然后分割成不同的Mod阶段
    mod_phases = []
    current_phase = []
    current_mod = None
    
    for is_correct, correct_side,_ in transformed:
        if correct_side != current_mod and not correct_side=='E':
            if current_phase:  # 保存上一个phase
                mod_phases.append(current_phase)
            current_phase = []
            current_mod = correct_side
        current_phase.append([is_correct, correct_side])
    
    if current_phase:  # 添加最后一个phase
        mod_phases.append(current_phase)
        
    # 分析每个Mod阶段的切换模式
    all_patterns = []
    for phase in mod_phases:
        pattern = analyze_switching_patterns(phase)
        if pattern:  # 只添加有切换的pattern
            all_patterns.append(pattern)
    
    return all_patterns


def get_switch_sequence(eve_list,lenth=5):
    transformed = transform_sequence(eve_list)  #(is_correct, correct_side,reward_or_not)
    switch_rewards = []
    previous_choice = None  # 用于记录上一次的实际选择
    
    for i in range(1, len(transformed)):
        current_is_correct, current_correct_side, current_reward = transformed[i]
        previous_is_correct, previous_correct_side, previous_reward = transformed[i-1]
        
        # 确定当前和上一次的实际选择
        current_actual_choice = current_correct_side if current_is_correct else ('L' if current_correct_side == 'R' else 'R')
        previous_actual_choice = previous_correct_side if previous_is_correct else ('L' if previous_correct_side == 'R' else 'R')
        
        # 如果是第一次迭代，只记录previous_choice
        if previous_choice is None:
            previous_choice = previous_actual_choice
            continue
        
        # 检查是否发生了切换
        if current_actual_choice != previous_choice:
            # 收集前5条数据的reward_or_not
            rewards_before_switch = []
            for j in range(max(0, i-lenth), i):  # 确保不越界
                rewards_before_switch.append(1 if transformed[j][2] else 0)
            if len(rewards_before_switch)==lenth:
                switch_rewards.append(rewards_before_switch)
        
        # 更新previous_choice
        previous_choice = current_actual_choice
    
    return switch_rewards

def get_random_ns_sequence(eve_list,lenth=5,num_get=50):
    import random
    transformed = transform_sequence(eve_list)  #(is_correct, correct_side,reward_or_not)
    non_switch_rewards = []
    previous_choice = None  # 用于记录上一次的实际选择
    
    for i in range(1, len(transformed)):
        current_is_correct, current_correct_side, current_reward = transformed[i]
        previous_is_correct, previous_correct_side, previous_reward = transformed[i-1]
        
        # 确定当前和上一次的实际选择
        current_actual_choice = current_correct_side if current_is_correct else ('L' if current_correct_side == 'R' else 'R')
        previous_actual_choice = previous_correct_side if previous_is_correct else ('L' if previous_correct_side == 'R' else 'R')
        
        # 如果是第一次迭代，只记录previous_choice
        if previous_choice is None:
            previous_choice = previous_actual_choice
            continue
        
        # 检查是否发生了切换
        if current_actual_choice == previous_choice:
            # 收集前5条数据的reward_or_not
            rewards_before_switch = []
            for j in range(max(0, i-lenth), i):  # 确保不越界
                rewards_before_switch.append(1 if transformed[j][2] else 0)
            if len(rewards_before_switch)==lenth:
                non_switch_rewards.append(rewards_before_switch)
        
        # 更新previous_choice
        previous_choice = current_actual_choice
    if len(non_switch_rewards) > num_get:
        non_switch_rewards=random.sample(non_switch_rewards,num_get)
    return non_switch_rewards  
  
def switch_bayes_reg(sw_seq,ns_seq):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    x_input=np.array(sw_seq+ns_seq)
    y_input=np.array([1 for i in range(len(sw_seq))]+[0 for i in range(len(ns_seq))])
    model = LogisticRegression()
    model.fit(x_input,y_input)
    weights = model.coef_
    bias = model.intercept_
    y_pred=model.predict(x_input)
    acc = accuracy_score(y_input, y_pred)
    return weights,bias,acc

def learning_auc(trial_sequence):
    is_correct = np.array([t[1] for t in trial_sequence])

    cum_correct = np.cumsum(is_correct)
    trials = np.arange(1, len(is_correct) + 1)

    cum_accuracy = cum_correct / trials
    x = trials / trials[-1]  # 归一化到 [0,1]

    auc = np.trapz(cum_accuracy, x)
    return auc

def cumulative_attempt_auc(eve_list):
    attempt_events = {"SlnL", "SlnR", "ROmL", "ROmR"}
    attempts = [
        (t, e) for t, e in eve_list
        if e in attempt_events
    ]

    if len(attempts) < 2:
        return np.nan

    times = np.array([t for t, _ in attempts])

    t0 = times[0]
    t1 = times[-1]

    x = (times - t0) / (t1 - t0)        # 归一化时间
    y = np.arange(1, len(times) + 1)    # 累计尝试数
    y = y / y[-1]                       # 归一化累计尝试

    auc = np.trapz(y, x)
    return auc

def log_lr(eve_list, alpha=0.5):
    left_events  = {"SlnL", "ROmL"}
    right_events = {"SlnR", "ROmR"}
    n_left = sum(1 for _, e in eve_list if e in left_events)
    n_right = sum(1 for _, e in eve_list if e in right_events)

    return np.log((n_left + alpha) / (n_right + alpha))

from bisect import bisect_left, bisect_right


def licking_rate(Event, eve_list):    
    event_map = {'SlnL':'reward', 'SlnR':'reward', 'ROmL':'noreward', 'ROmR':'noreward'}
    valid_events = list(event_map.keys())
    
    all_timestamps = [t for t, _ in eve_list]
    time_scale = 1000.0 if (len(all_timestamps) > 0 and max(all_timestamps) > 100000) else 1.0
    
    filtered_events = []
    for timestamp, eve_name in eve_list:
        if eve_name in valid_events:
            filtered_events.append( (timestamp/time_scale, event_map[eve_name]) )

    all_lick_times = []
    if 'LickL' in Event and 'Time_S' in Event['LickL'] and Event['LickL']['Time_S']:
        all_lick_times.extend([t/time_scale for t in Event['LickL']['Time_S']])
    if 'LickR' in Event and 'Time_S' in Event['LickR'] and Event['LickR']['Time_S']:
        all_lick_times.extend([t/time_scale for t in Event['LickR']['Time_S']])
    all_lick_times.sort()

    reward_lick_counts, noreward_lick_counts = [], []
    for i in range(len(filtered_events)-1):
        curr_ts, curr_type = filtered_events[i]
        next_ts, _ = filtered_events[i+1]
        start_idx = np.searchsorted(all_lick_times, curr_ts, side='left')
        end_idx = np.searchsorted(all_lick_times, next_ts, side='left')
        lick_count = end_idx - start_idx
        
        if curr_type == 'reward':
            reward_lick_counts.append(lick_count)
        else:
            noreward_lick_counts.append(lick_count)

    avg_licks_reward = np.mean(reward_lick_counts) if reward_lick_counts else 0
    avg_licks_noreward = np.mean(noreward_lick_counts) if noreward_lick_counts else 0
    total_lick_events = len(reward_lick_counts) + len(noreward_lick_counts)

    return avg_licks_reward, avg_licks_noreward, total_lick_events
