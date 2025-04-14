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

def recog_trials(lick_times, min=0 , max =0x7fffffff,delta = 25000):
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


def missing_trig(lick_times,ref_times,delta1=25000,delta2=25000,addtime=500):
    #识别缺失的trial信息
    trials = recog_trials(lick_times
                          ,min=ref_times[0]
                          ,max=ref_times[-1],delta=delta1)
    missing_trials = find_not_matched(trials, ref_times,addtime,delta=delta2)
    while len(missing_trials) > 0:
        if missing_trials[0] < ref_times[0]:
            missing_trials.pop(0)
        else:
            break
    while len(missing_trials) > 0:
        if missing_trials[-1] > ref_times[-1]:
            missing_trials.pop(-1)
        else:
            break

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
                print('Turning point not found! Reucing threshold to', threshold) 
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
    print("No Log Data, skipping compare")
    while 1==1:
        if Event[0]>=truncate_time[0]:
            break
        Event.pop(0)
    while 1==1:
        if Event[-1]<=truncate_time[1]:
            break
        Event.pop(-1)
    return Event

def Event_compare(Event=list,Event_log=list):
    #对比两个事件的时间序列,并进行线性拟合
    from sklearn.linear_model import RANSACRegressor
    from scipy import interpolate
    import numpy as np
    
    hw_times = np.asarray(Event)
    sw_times = np.asarray(Event_log)
    sw_times = (sw_times-min(sw_times))/(max(sw_times)-min(sw_times))
    sw_times = (sw_times*(max(hw_times)-min(hw_times))+min(hw_times))
    
    matched_hw, matched_sw = clean_dtw_matches(hw_times, sw_times)
    ransac = RANSACRegressor(residual_threshold=0.3)
    ransac.fit(matched_sw.reshape(-1, 1), matched_hw)
    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    a_=a * (max(hw_times) - min(hw_times)) / (max(sw_times) - min(sw_times))
    b_=b + a * (min(hw_times) - (max(hw_times) - min(hw_times)) * min(sw_times) / (max(sw_times) - min(sw_times)))
    print(f'{len(matched_hw)}/{len(hw_times)} matched! a={a_}, b={b_} ')
    return a_,b_

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
    Event_ref_1=bubbleSort(Event_ref_1)
    Event_ref_2=bubbleSort(Event_ref_2) if not Event_ref_2==None else None
    snapped_events = []
    for t in Event:
        idx = bisect.bisect_left(Event_ref_1, t)-1
        snapped_events.append(Event_ref_1[max(idx - 1,0)] if (t-Event_ref_1[max(idx - 1,0)])<(Event_ref_1[idx]-t) else Event_ref_1[idx])
        if abs(snapped_events[-1] - t ) > delta:
            snapped_events.pop(-1)
        else:
            continue
        if not Event_ref_2 == None:
            idx = bisect.bisect_left(Event_ref_2, t)-1
            snapped_events.append(Event_ref_2[max(idx - 1,0)] if t-Event_ref_2[max(idx - 1,0)]<Event_ref_2[idx]-t else Event_ref_2[idx])
            if abs(snapped_events[-1] - t ) > delta:
                snapped_events.pop(-1)  
                snapped_events.append(t)  
        else :
            snapped_events.append(t)  
            continue

    return snapped_events

def fixing(Event,delta1=25000,delta2=25000,log_file=None,matched_row=None):
    #主入口  delta1 trial识别
    addtime=500
    bol_fix=False
    fix_ref={
        'SlnL':'ROmL',
        'SlnR':'ROmR',
        'ROmL':'SlnL',
        'ROmR':'SlnR'
    }
    for i in Event:
        Event[i] = filtering_sig(Event[i])
    for i in fix_ref:
        if Event[i]['Time_E']==[]:
            bol_fix=True
            break
    if (Event['ModL']['Time_E']==[]) ^ (Event['ModR']['Time_E']==[]):
        bol_fix=True
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
            log_Event=Log_Events(event_num)
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
    

    if Log_Events==None:
        #无日志辅助的修正，精度较低
        print('No Log Data!')
        st_seq=[]
        ed_seq=[]
        for i in Event:
            Event[i]=filtering_sig(Event[i])
            if not Event[i]['Time_E']==[] and i == 'ROmL'or'ROmR'or'SlnL'or'SlnR'or'ModL'or'ModR':
                st_seq.append(Event[i]['Time_S'][0]) 
                ed_seq.append(Event[i]['Time_S'][-1])
        truncate_time=[min(st_seq),max(st_seq)]
        
        for i in fix_ref:
            if Event[i]['Time_E']==[] :
                if Event[fix_ref[i]]['Time_E'] ==[]:
                    print('No SlnL and ROmL Event! Cannot fix! \n')
                    continue
                else:
                    Event[i]['Time_S'] = missing_trig(Event[fix_ref[i]]['Time_S'], Event[fix_ref[i]]['Time_S'],delta1,delta2,addtime)
                    Event[i]['Time_S']=truncate(Event[i]['Time_S'],truncate_time)  
                     
        start_time_seq=[]
        for i in fix_ref:     
            start_time_seq.append(Event[i]['Time_S'][0])
        start_time=min(start_time_seq)
        
        if(Event['ModL']['Time_E']==[] and not Event['ModR']['Time_E']==[]): 
            print('No ModL Event! Fixing \n')# 下面传入的分别是 : 对边mod 对边sln和rom,本边sln和rom
            Event['ModL']['Time_S'] = process_trials(Event['ModR']['Time_S'], Event['SlnR']['Time_S'], Event['ROmR']['Time_S'] , Event['SlnL']['Time_S'], Event['ROmL']['Time_S'],addtime)
            if(Event['ModR']['Time_S'][0] > start_time):
                Event['ModL']['Time_S'].insert(0,start_time - addtime)   ##如果在另一边的mod信号之前就已经有了trial的判定信息,则意味着实际上实验是从本侧开始的,应当在本侧的开头的插入一个mod的信息
        if(Event['ModR']['Time_E']==[] and not Event['ModL']['Time_E']==[]):
            print('No ModR Event! Fixing \n')
            Event['ModR']['Time_S'] = process_trials(Event['ModL']['Time_S'], Event['SlnL']['Time_S'], Event['ROmL']['Time_S'] , Event['SlnR']['Time_S'], Event['ROmR']['Time_S'],addtime)
            if(Event['ModL']['Time_S'][0] > start_time):
                Event['ModR']['Time_S'].insert(0,start_time - addtime)
        if Event['ModR']['Time_E']==[] and Event['ModL']['Time_E']==[]:
            Event['ModR']['Time_S']=[]
            Event['ModL']['Time_S']=[]
            print('No Mod info!')
    else:
        #日志辅助修正
        print('Fixing with log! ')
        a_list=[]
        b_list=[]
        for i in fix_ref:
            if not Event[i]['Time_E']==[]:
                a,b=Event_compare(Event[i]['Time_S'],log_Event[i])
                a_list.append(a)
                b_list.append(b)
        a=np.mean(a_list)
        a_std=np.std(a_list)
        b=np.mean(b_list)
        b_std=np.std(b_list)
        print(f'{len(a_list)}Events matched! a={a}±{a_std}, b={b/50000}±{b_std/50000} (seconds)')
        for i in fix_ref:
            if Event[i]['Time_E']==[]:
                print(f'No {i} Event! Fixing \n')
                Event[i]['Time_S'] = [a*x+b for x in log_Event[i]]
                Event[i]['Time_S'] = Snap_to_reference(Event[i]['Time_S'],Event[fix_ref[i]]['Time_S'], Event['LickL']['Time_S'] if i== 'ROmL' or 'SlnL' else Event['LickR']['Time_S'],delta=delta2)
        if not (log_Event['ModL']==[] and log_Event['ModR']==[]):
            if Event['ModL']['Time_E']==[]:
                print('No ModL Event! Fixing \n')
                Event['ModL']['Time_S'] = [a*x+b for x in log_Event[i]]
                ref_list=Event['SlnR']['Time_S']+ Event['ROmR']['Time_S']
                ref_list.sort()
                Event['ModL']['Time_S'] = Snap_to_reference(log_Event['ModL'], ref_list, delta=delta2)
                Event['ModL']['Time_S'] = [x+addtime*2 for x in Event['ModL']['Time_S']]
            if Event['ModR']['Time_E']==[]:
                print('No ModR Event! Fixing \n')
                Event['ModR']['Time_S'] = [a*x+b for x in log_Event[i]]
                ref_list=Event['SlnL']['Time_S']+ Event['ROmL']['Time_S']
                ref_list.sort()
                Event['ModR']['Time_S'] = Snap_to_reference(log_Event['ModR'], ref_list, delta=delta2)
                Event['ModR']['Time_S'] = [x+addtime*2 for x in Event['ModR']['Time_S']]
        for i in fix_ref:
            if Event[i]['Time_E']==[]:
                Event[i]['Time_S'] = [x+addtime for x in Event[i]['Time_S']]  
    return Event
