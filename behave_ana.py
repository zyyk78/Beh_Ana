def sort_events(Event,ignore_list=None,long_event=None):
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
        if 'Time_S' in Event[i] and 'Time_E' in Event[i] and not i in ignore_list:
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

def reward_rate(Event,window=[0,-1]):
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
        for i in cal_list:
            Event_trunc[i]=Event[i]['Time_S']
            for j in Event['Blackhole']:
                Event_trunc[i]=fixing.truncate(Event_trunc[i],j,'inner')
            Event_trunc[i]=fixing.truncate(Event_trunc[i],window)
    else:
        for i in cal_list:
            Event_trunc[i]=Event[i]['Time_S']
            Event_trunc[i]=fixing.truncate(Event_trunc[i],window)
            
    L=len(Event_trunc['SlnL'])/(len(Event_trunc['ROmL'])+len(Event_trunc['SlnL']))
    R=len(Event_trunc['SlnR'])/(len(Event_trunc['ROmR'])+len(Event_trunc['SlnR']))
    T=(len(Event_trunc['SlnR'])+len(Event_trunc['SlnL']))/(len(Event_trunc['ROmL'])+len(Event_trunc['SlnL'])+len(Event_trunc['ROmR'])+len(Event_trunc['SlnR']))
    return L,R,T

def transform_sequence(eve_list):
    """将原始序列转换为选择结果序列
    
    Args:
        eve_list: 原始序列，格式为[[TimeStamp, Event], ...]
    
    Returns:
        list: 转换后的序列，每个元素为(is_correct, correct_side,reward_or_not)
    """
    transformed_eve_list = []
    current_correct_side = None
    in_blackhole = False
    
    for timestamp, event in eve_list:
        if event == 'Blackhole_S':
            in_blackhole = True
            current_correct_side=None
            transformed_eve_list.append((False,'E',False))
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
            transformed_eve_list.append((is_correct, current_correct_side,result))
    
    return transformed_eve_list

def accuracy_rate(eve_list):
    """计算两边和总体的正确比例(如果Event中出现了Blackhol关键字则会自动应用)
    
    Args:
        eve_list: 排好的时间序列
    
    Returns:
        tuple(int,int,int):左右总体的正确比例
    """
    LC, LW, RC, RW = 0, 0, 0, 0
    tr_eve_list=transform_sequence(eve_list)
    for is_correct, correct_side,_ in tr_eve_list:
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
    
    # 计算正确率
    L = LC / (LC + LW) if (LC + LW) > 0 else 0
    R = RC / (RC + RW) if (RC + RW) > 0 else 0
    T = (LC + RC) / (LC + LW + RC + RW) if (LC + LW + RC + RW) > 0 else 0
    
    return L,R,T

def analyze_switching_patterns(transformed_sequence):
    """分析模式切换模式（基于转换后的序列）
    
    Args:
        transformed_sequence: 转换后的序列，格式为[[is_correct, correct_side], ...]
    
    Returns:
        list: 切换模式列表，每个元素形如 [5, -7, 20]
              (正数表示L→R切换，负数表示R→L切换，数值表示延迟次数)
    """
    if not transformed_sequence:
        return []
    
    switch_pattern = []
    current_side = transformed_sequence[0][1]  # 初始正确侧
    count_since_last_switch = 0
    
    for is_correct, correct_side in transformed_sequence:
        count_since_last_switch += 1
        
        # 检测模式切换
        if correct_side != current_side:
            # 确定切换方向 (L→R为正，R→L为负)
            switch_value = count_since_last_switch if current_side == 'L' else -count_since_last_switch
            switch_pattern.append(switch_value)
            
            # 重置计数器
            current_side = correct_side
            count_since_last_switch = 0
    
    return switch_pattern

def get_all_switching_patterns(eve_list):
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

