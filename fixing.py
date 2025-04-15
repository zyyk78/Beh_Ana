def filtering_sig(Event_A,min_duration=10):
    #去除信号噪声
    time_s = Event_A['Time_S']
    time_e = Event_A['Time_E']
    if len(time_e) == 0:
        return Event_A
    
    
    valid_indices = []
    
    for i in range(len(time_e)):
        duration = time_e[i] - time_s[i]
        if duration >= min_duration:
            valid_indices.append(i)
        else:
            print(f"Removed an abnormal event")
    Event_B={
        'Time_S' :[],
        'Time_E' :[]
    }
    Event_B['Time_S'] = [time_s[i] for i in valid_indices]
    Event_B['Time_E'] = [time_e[i] for i in valid_indices]
    if len(time_s) > len(time_s):
        Event_B['Time_S'].append(Event_A['Time_S'][-1])
    return Event_B

def recog_trials(lick_times,delta = 25000):
    #识别lick信号中的trial
    trials = []

    for i in range(1, len(lick_times)):
        if lick_times[i] - lick_times[i-1] > delta:
            trials.append(lick_times[i-1])

    return trials   

def find_not_matched(trials,ref_trials,addtime=0,delta=25000):
    #寻找不与参考匹配的trial
    missing_trials = []
    for trial in trials:
        break_flag = False
        for ref in ref_trials:
            if ref -delta <= trial <= ref + delta:
                break_flag = True
                break 
        if break_flag:
            continue
        missing_trials.append(trial+addtime)
    return missing_trials


def process_trials(mod_times, cor_s , cor_r , inc_s, inc_r,addtime=500):
    #识别switch情况
    window_size=30
    success_threshold=21
    
    all_events = []
    
    for t in mod_times:
        all_events.append((t, 'mod'))
    
    for t in cor_s:
        all_events.append((t, 'cor'))
    for t in cor_r:
        all_events.append((t, 'cor'))
    for t in inc_s:
        all_events.append((t, 'inc'))
    for t in inc_r:
        all_events.append((t, 'inc'))
    
    all_events.sort(key=lambda x: x[0])
    
    mod_times_f = []
    window = [0] * window_size  
    pointer = 0 
    active = -1
    i=0 
    threshold=success_threshold
    while i < len(all_events):
        time = all_events[i][0]
        event_type = all_events[i][1]
        if event_type == 'mod':
            if active == -1 :
                active = i           # 记录目前激活状态 0代表本边已达成条件 非0代表为达成条件且记录连上一个mod位置
                i+=1
            else:                    # 如果两个mod中间没有找到一个信号
                window = [0] * window_size 
                i= max(active-30,0)  #倒带到上一个mod之前
                active = -1
                threshold -= 1       #降低阈值  
            continue
        
        if event_type == 'cor':
            window[pointer] = 1  
        elif event_type == 'inc':
            window[pointer] = 0  
        
        pointer = (pointer+1) % window_size 

        success_count = sum(window)
        if (success_count >= threshold) and not active == -1:
            mod_times_f.append(time+addtime) 
            active = -1  
            threshold=success_threshold
        i+=1
    return mod_times_f

def clean_dtw_matches(hw_times, sw_times):
    #清理dwt的匹配结果
    import numpy as np
    from dtaidistance import dtw    

    distance, paths = dtw.warping_paths(hw_times, sw_times)
    path = dtw.best_path(paths)
    
    match_residuals = []
    for hw_idx, sw_idx in path:
        residual = abs(hw_times[hw_idx] - sw_times[sw_idx])
        match_residuals.append((hw_idx, sw_idx, residual))
    
    match_residuals.sort(key=lambda x: x[2])

    unique_matches = {}
    used_hw, used_sw = set(), set()
    for hw_idx, sw_idx, _ in match_residuals:
        if hw_idx not in used_hw and sw_idx not in used_sw:
            unique_matches[hw_idx] = sw_idx
            used_hw.add(hw_idx)
            used_sw.add(sw_idx)
    

    matched_hw = [hw_times[i] for i in sorted(unique_matches.keys())]
    matched_sw = [sw_times[j] for j in [unique_matches[i] for i in sorted(unique_matches.keys())]]
    
    return np.array(matched_hw), np.array(matched_sw)

def truncate(Event,truncate_time=None):
    while len(Event)>0:
        if truncate_time[0] <0 :
            break
        if Event[0]>=truncate_time[0]:
            break
        Event.pop(0)
    while len(Event)>0:
        if truncate_time[1] <0 :
            break
        if Event[-1]<=truncate_time[1]:
            break
        Event.pop(-1)
    return Event

def align_time_sequences(hw_times, sw_times, max_ratio=1.1, max_abs_diff=None):

    if not hw_times or not sw_times:
        return []
    
    hw_deltas = [hw_times[i+1] - hw_times[i] for i in range(len(hw_times)-1)]
    sw_deltas = [sw_times[i+1] - sw_times[i] for i in range(len(sw_times)-1)]
    

    dp = [[float('inf')] * (len(sw_deltas)+1) for _ in range(len(hw_deltas)+1)]
    dp[0][0] = 0

    for i in range(len(hw_deltas)+1):
        for j in range(len(sw_deltas)+1):
            if i == 0 and j == 0:
                continue
                
            current_min = float('inf')
            
            if i > 0:
                cost = hw_deltas[i-1]  
                if dp[i-1][j] + cost < current_min:
                    current_min = dp[i-1][j] + cost
            
            if j > 0:
                cost = sw_deltas[j-1]  
                if dp[i][j-1] + cost < current_min:
                    current_min = dp[i][j-1] + cost
            
            if i > 0 and j > 0:
                delta1 = hw_deltas[i-1]
                delta2 = sw_deltas[j-1]
                
                if delta1 == 0 and delta2 == 0:
                    ratio = 1.0
                elif delta1 == 0 or delta2 == 0:
                    ratio = float('inf')
                else:
                    ratio = max(delta1/delta2, delta2/delta1)
                
                abs_diff = abs(delta1 - delta2)
                
                if (ratio <= max_ratio and 
                    (max_abs_diff is None or abs_diff <= max_abs_diff)):
                    cost = abs_diff  
                    if dp[i-1][j-1] + cost < current_min:
                        current_min = dp[i-1][j-1] + cost
            
            dp[i][j] = current_min
    
    alignment = []
    i, j = len(hw_deltas), len(sw_deltas)

    hw_idx, sw_idx = len(hw_times)-1, len(sw_times)-1
    alignment.append((hw_times[hw_idx], sw_times[sw_idx]))
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + abs(hw_deltas[i-1] - sw_deltas[j-1]):

            i -= 1
            j -= 1
            hw_idx -= 1
            sw_idx -= 1
            alignment.append((hw_times[hw_idx], sw_times[sw_idx]))
        elif i > 0 and dp[i][j] == dp[i-1][j] + hw_deltas[i-1]:

            i -= 1
            hw_idx -= 1
            alignment.append((hw_times[hw_idx], None))
        elif j > 0 and dp[i][j] == dp[i][j-1] + sw_deltas[j-1]:

            j -= 1
            sw_idx -= 1
            alignment.append((None, sw_times[sw_idx]))
        else:

            raise RuntimeError("Error in alignment traceback")

    alignment.reverse()
    
    return alignment


def Event_compare(Event=list,Event_log=list):
    #对比两个事件的时间序列,并进行线性拟合
    import numpy as np
    from sklearn.linear_model import LinearRegression
    # from sklearn.linear_model import RANSACRegressor
    # from scipy import interpolate
    # hw_times = np.asarray(Event)
    # sw_times = np.asarray(Event_log)
    alignment=align_time_sequences(Event, Event_log)
    hw=[]
    sw=[]
    for pair in alignment:
        if (not pair[0]  == None) and (not pair[1]  == None):
            hw.append(pair[0])
            sw.append(pair[1])
    print(f'{len(Event)} events in hw, {len(Event_log)} events in sw, {len(sw)} matched!')
    sw=np.array(sw).reshape(-1, 1)
    hw=np.array(hw)
    model = LinearRegression()
    model.fit(sw,hw)
    a=model.coef_[0]
    b=model.intercept_
    r=model.score(sw,hw)
    print(f'Fit data: a={a:.2f}, b={(b/50000):.2f}(s), r^2={r:.2f}')
    
    # dwt+RANSAC的方法不稳定已放弃
    # ave_shift = np.mean(hw_times) - np.mean(sw_times)
    # sw_times = sw_times+ave_shift
    
    # matched_hw, matched_sw = clean_dtw_matches(hw_times, sw_times)
    # ransac = RANSACRegressor(residual_threshold=0.3)
    # ransac.fit(matched_sw.reshape(-1, 1), matched_hw)
    # a = ransac.estimator_.coef_[0]
    # b = ransac.estimator_.intercept_
    # b_=b+ave_shift
    # a_ = a
    # print(f'{len(matched_hw)}/{len(hw_times)} matched! a={a_}, b={b_} ')
    # return a_,b_
    return a,b,r

def bubbleSort(arr):
    #冒泡排序, 在数组已经排好时非常快
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
                
def Snap_to_reference(Event,Event_ref_1,Event_ref_2=None,delta=25000):
    #吸附到参考时间
    import bisect
    Event_ref_1=bubbleSort(Event_ref_1) if not Event_ref_1==None else None
    Event_ref_2=bubbleSort(Event_ref_2) if not Event_ref_2==None else None
    snapped_events = []
    for t in Event:
        if not Event_ref_1 == None:
            if len(Event_ref_1)>0:
                idx_r = min(bisect.bisect_left(Event_ref_1, t),len(Event_ref_1)-1)
                idx_l=max(idx_r-1,0)
                snapped_events.append(Event_ref_1[idx_l] if (t-Event_ref_1[idx_l])<(Event_ref_1[idx_r]-t) else Event_ref_1[idx_r])
                if abs(snapped_events[-1] - t ) > delta:
                    snapped_events.pop(-1)   #吸附失败就删掉
                else:
                    continue   #如果吸附成功就跳过本条
        if not Event_ref_2 == None:
            if len(Event_ref_2)>0:
                idx_r = min(bisect.bisect_left(Event_ref_2, t),len(Event_ref_2)-1)
                idx_l=max(idx_r-1,0)
                snapped_events.append(Event_ref_2[idx_l] if t-Event_ref_2[idx_l]<Event_ref_2[idx_r]-t else Event_ref_2[idx_r])
                if abs(snapped_events[-1] - t ) > delta:
                    snapped_events.pop(-1)    #如果失败就删掉
                else:
                    continue  #如果吸附成功就跳过本条
        
        snapped_events.append(t)     #上面全部失败了,就直接加入t

    return snapped_events

def find_missing_ranges(Event, log_Event, window_size=10, min_missing=1):
    missing_ranges = {}
    checklist=['ModL','ModR','SlnL','SlnR','ROmL','ROmR']
    
    all_event_times = []
    for event_type in checklist:
        if not Event[event_type]['Time_S'] == []:
            all_event_times.extend(Event[event_type]['Time_S'])
    all_event_times = sorted(all_event_times)

    all_log_times = []
    for event_type in checklist:
        all_log_times.extend(log_Event[event_type])
    all_log_times = sorted(all_log_times)

    missing_ranges = []
    start_missing = None
    n = len(all_event_times)

    for i in range(n - window_size + 1):
        window_start = all_event_times[i]
        window_end = all_event_times[i + window_size - 1]

        has_log = any(window_start <= t <= window_end for t in all_log_times)

        if not has_log and start_missing is None:

            start_missing = window_start
        elif has_log and start_missing is not None:

            missing_ranges.append([start_missing, window_end])
            start_missing = None

    if start_missing is not None:
        missing_ranges.append([start_missing, all_event_times[-1]])

    filtered_ranges = []
    for range_start, range_end in missing_ranges:
        if (range_end - range_start) >= (min_missing - 1):
            filtered_ranges.append([range_start, range_end])

    return filtered_ranges


def fixing(Event,delta1=25000,delta2=25000,log_file=None,matched_row=None):
    #主入口  
    addtime=500
    bol_fix=False   
    
    fix_ref={    #对侧边参考
        'SlnL':'ROmL',
        'SlnR':'ROmR',
        'ROmL':'SlnL',
        'ROmR':'SlnR'
    }
    
    too_long_events=[]    
    for i in ['LickL','LickR']:
        Event[i] = filtering_sig(Event[i])    #去毛刺
        for j in range(len(Event[i]['Time_E'])):
            if Event[i]['Time_E'][j] - Event[i]['Time_S'][j] >50000: 
                too_long_events.append([Event[i]['Time_S'][j]-10,Event[i]['Time_E'][j]+10])   #去除卡死的信号
                
    Event['Blackhole']={   # 用这个事件记录那些有问题的时间段
        'Time_S':[],
        'Time_E':[]
    }
    
    for i in too_long_events:   
        Event['Blackhole']['Time_S'].append(i[0])
        Event['Blackhole']['Time_E'].append(i[1])
    if len(too_long_events)>0:
        print(f'Too long event detected!{too_long_events}')   

    
    if matched_row.iloc[0,2]== 'P':      #P模式下清理可能的问题
        Event['ModL']['Time_S'] = []
        Event['ModR']['Time_S'] = []
        
    for i in fix_ref:
        if Event[i]['Time_E']==[]:
            bol_fix=True   #是否需要修复
            break
        
    if (Event['ModL']['Time_E']==[]) ^ (Event['ModR']['Time_E']==[]):
        bol_fix=True     #同上
    if not bol_fix:
        print('No missing event, skipping fixing')
        return Event
    
    import os
    import log_processer
    import numpy as np
    
    if  os.path.exists(log_file):
        Log_Events=log_processer.process_log_file(log_file)
        log_Event = None
        if matched_row.empty:
            print(f"Warning: No matching row found in README.xlsx.")
            event_num=input(f"Please input the event number of event: ")
            log_Event=Log_Events(event_num)       #没有csv,需要指定事件编号
        else:
            rec_start_time = (matched_row.iloc[0, 6].hour*3600+matched_row.iloc[0, 6].minute*60+matched_row.iloc[0, 6].second)
            rec_end_time =  (matched_row.iloc[0, 7].hour*3600+matched_row.iloc[0, 7].minute*60+matched_row.iloc[0, 7].second)
            time_dff = []
            for i in range(len(Log_Events)):
                time_dff.append([abs(Log_Events[i]['Time_Range_second'][0]-rec_start_time)+abs(Log_Events[i]['Time_Range_second'][1]-rec_end_time),i])
            time_dff.sort(key=lambda x: x[0])
            log_Event=Log_Events[time_dff[0][1]]   #匹配找到和实验csv记录的时间最接近的Event
            print(f"Log file {log_file} found. Event is matched to {time_dff[0][1]}, time difference is {time_dff[0][0]} seconds.")
    else:
        print(f"Warning: Log file {log_file} does not exist. Cannot process log events.")
        Log_Events=None
    


    #先进行无日志辅助的修正，精度较低
    for i in fix_ref:
        if Event[i]['Time_E']==[] :
            if Event[fix_ref[i]]['Time_E'] ==[]:
                print('No Reference Event! Cannot fix! ')
                continue
            else:
                print(f'No {i} Event! Fixing ')
                found_trials = recog_trials(Event['LickL']['Time_S'] if i== 'ROmL' or 'SlnL' else Event['LickR']['Time_S'],delta=delta1)   #识别trial
                for j in fix_ref:
                    if not j==i: 
                        found_trials = find_not_matched(found_trials, Event[j]['Time_S'],0,delta=delta2)     #剔除那些能被解释为其他事件的(为什么识别出的信号这么杂乱?)
                Event[i]['Time_S'] = truncate(found_trials,[Event[fix_ref[i]]['Time_S'][0],Event[fix_ref[i]]['Time_S'][-1]])  #截断(删掉那些没用的信号)
    
    if(Event['ModL']['Time_E']==[] and (not Event['ModR']['Time_E']==[])):   #识别ModL/R
        print('No ModL Event! Fixing ')# 下面传入的分别是 : 对边mod 对边sln和rom,本边sln和rom
        Event['ModL']['Time_S'] = process_trials(Event['ModR']['Time_S'], Event['SlnR']['Time_S'], Event['ROmR']['Time_S'] , Event['SlnL']['Time_S'], Event['ROmL']['Time_S'],addtime)
    if(Event['ModR']['Time_E']==[] and (not Event['ModL']['Time_E']==[])):
        print('No ModR Event! Fixing ')
        Event['ModR']['Time_S'] = process_trials(Event['ModL']['Time_S'], Event['SlnL']['Time_S'], Event['ROmL']['Time_S'] , Event['SlnR']['Time_S'], Event['ROmR']['Time_S'],addtime)

    if Event['ModR']['Time_E']==[] and Event['ModL']['Time_E']==[]:
        Event['ModR']['Time_S']=[]
        Event['ModL']['Time_S']=[]
        print('No Mod info!')
    if not Log_Events==None:
        #日志辅助修正
        print('Fixing with log! ')
        a_list=[]
        b_list=[]
        for i in fix_ref:
            if not Event[i]['Time_E']==[]:
                if(len(log_Event[i])>20):
                    print('Matching',i)
                    a,b,r=Event_compare(Event[i]['Time_S'],log_Event[i])   #将软硬件记录的事件进行初步对其
                    if a<0.95 or r<0.95:
                        print(f'{i} Event not matched! Removed!')  #清理那些效果不好的数据段
                    else:
                        a_list.append(a)
                        b_list.append(b)
                else:
                    print(f'{i} Event too short! Removed!')
                    continue
        a=np.mean(a_list)
        a_std=np.std(a_list)
        b=np.mean(b_list)
        b_std=np.std(b_list)
        
        to_remove = []
        for i in range(len(b_list)):
            if b_list[i] > b + 2 * b_std or b_list[i] < b - 2 * b_std:   #剔除离群点
                print(f'Event reference {i} Event too far! Marked!')
                to_remove.append(i)
        to_remove.sort(reverse=True)  #从大到小删除
        for idx in to_remove:  
            a_list.pop(idx)
            b_list.pop(idx)
                
        a=np.mean(a_list)
        a_std=np.std(a_list)        
        b=np.mean(b_list)
        b_std=np.std(b_list)
        
        
        log_Event_fix = {}
        for i in log_Event:
            log_Event_fix[i] = [a*x+b for x in log_Event[i]]   #更新软件记录
            
        missing_ranges=find_missing_ranges(Event, log_Event_fix)   #识别软件记录缺失的部分(可能)
        for i in missing_ranges:
            Event['Blackhole']['Time_S'].append(i[0])
            Event['Blackhole']['Time_E'].append(i[1])  
            
            
        if len(missing_ranges) > 0:            #如果出现了缺失区域 则在清洗后重新配对
            print(f"Missing ranges detected: {missing_ranges}")
            
            a_list=[]
            b_list=[]
            for i in fix_ref:
                if not Event[i]['Time_E']==[]:
                    if(len(log_Event[i])>20):
                        print('Re-Matching',i)
                        
                        removed_missing_Event=Event[i]['Time_S']
                        to_remove = []
                        for j in missing_ranges:
                            for k in range(len(Event[i]['Time_S'])):
                                if Event[i]['Time_S'][k] >j[0] and Event[i]['Time_S'][k] <j[1]:  #如果某个id落在了missing区间内
                                    to_remove.append(k)
                        to_remove=list(set(to_remove))    
                        to_remove.sort(reverse=True)        
                        for j in to_remove:
                            removed_missing_Event.pop(j)
                            
                        
                        a,b,r=Event_compare(removed_missing_Event,log_Event[i])   #重新将软硬件记录的事件进行对齐
                        if a<0.95 or r<0.95:
                            print(f'{i} Event not matched! Removed!')  #清理那些效果不好的数据段
                        else:
                            a_list.append(a)
                            b_list.append(b)
                    else:
                        print(f'{i} Event too short! Removed!')
                        continue
            a=np.mean(a_list)
            a_std=np.std(a_list)        
            b=np.mean(b_list)
            b_std=np.std(b_list)       
             
            to_remove=[]
            for i in range(len(b_list)):
                if b_list[i] > b + 2 * b_std or b_list[i] < b - 2 * b_std:   #剔除离群点
                    print(f'Event reference {i} Event too far! Marked!')
                    to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):  
                a_list.pop(idx)
                b_list.pop(idx)
            a=np.mean(a_list)
            a_std=np.std(a_list)        
            b=np.mean(b_list)
            b_std=np.std(b_list)
        print(f'{len(a_list)} Events matched! a={a:.2f}±{a_std:.2f}, b={(b/50000):.2f}±{(b_std/50000):.2f}(s)')  
        
        
        for i in fix_ref:
            if Event[i]['Time_E']==[]:
                print(f'No {i} Event! Fixing ')
                if not missing_ranges==[]:
                    missing_events=[]
                    for j in missing_ranges:
                        missing_events.extend(truncate(Event[i]['Time_S'],j))
                        
                Event[i]['Time_S'] = log_Event_fix[i]
                Event[i]['Time_S'] = Snap_to_reference(Event[i]['Time_S'],Event[fix_ref[i]]['Time_S'], Event['LickL']['Time_S'] if i== 'ROmL' or 'SlnL' else Event['LickR']['Time_S'],delta=delta2)
                Event[i]['Time_S'] = truncate(Event[i]['Time_S'],[0,-1])
                if not missing_ranges==[]:
                    Event[i]['Time_S']. extend(missing_events)
                    Event[i]['Time_S'].sort()
                
        if  matched_row.iloc[0,2]=='P':
            Event['ModL']['Time_S'] = []
            Event['ModR']['Time_S'] = []
        if not (log_Event_fix['ModL']==[] and log_Event_fix['ModR']==[]):
            if Event['ModL']['Time_E']==[]:
                print('No ModL Event! Fixing ')
                if not missing_ranges==[]:
                    missing_events=[]
                    for j in missing_ranges:
                        missing_events.extend(truncate(Event['ModL']['Time_S'],j))
                        
                Event['ModL']['Time_S'] = log_Event_fix['ModL']
                ref_list=Event['SlnR']['Time_S']+ Event['ROmR']['Time_S']
                ref_list.sort()
                Event['ModL']['Time_S'] = Snap_to_reference(log_Event_fix['ModL'], ref_list, delta=delta2)
                Event['ModL']['Time_S'] = [x+addtime*2 for x in Event['ModL']['Time_S']]
                if not missing_ranges==[]:
                    Event['ModL']['Time_S'].extend(missing_events)
                    Event['ModL']['Time_S'].sort()

            if Event['ModR']['Time_E']==[]:

                print('No ModR Event! Fixing ')
                if not missing_ranges==[]:
                    missing_events=[]
                    for j in missing_ranges:
                        missing_events.extend(truncate(Event['ModR']['Time_S'],j))
                        
                Event['ModR']['Time_S'] = log_Event_fix['ModR']
                ref_list=Event['SlnL']['Time_S']+ Event['ROmL']['Time_S']
                ref_list.sort()
                Event['ModR']['Time_S'] = Snap_to_reference(log_Event_fix['ModR'], ref_list, delta=delta2)
                Event['ModR']['Time_S'] = [x+addtime*2 for x in Event['ModR']['Time_S']]
                if not missing_ranges==[]:
                    Event['ModR']['Time_S'].extend(missing_events)
                    Event['ModR']['Time_S'].sort()
        for i in fix_ref:
            if Event[i]['Time_E']==[]:
                Event[i]['Time_S'] = [x+addtime for x in Event[i]['Time_S']]  
    return Event,log_Event_fix
