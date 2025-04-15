# -*- coding: utf-8 -*-
import re

class EventProcessor:
    def __init__(self):
        self.events = []
        self.current_event = None
        self.in_event = False
        self.event_start_time = None  
    
    def process_line(self, line):
        # 检查是否为INF日志
        if "[INF]" not in line:
            return
        
        # 提取时间戳和内容
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', line)
        if not timestamp_match:
            return
        
        timestamp_str = timestamp_match.group(1).split()[1]  # 只取时间部分
        h, m, s = timestamp_str.split(':')
        timestamp =round((int(h)*3600 + int(m)*60 + float(s))*50000)  #50kHz
        
        content = line.split('[INF]')[-1].strip()
        
        # 检查是否是开始事件
        if "Start to Detect Lick" in content and not self.in_event:
            self.current_event = {
                'Time_Range_second': [],
                'ROmL': [],
                'ROmR': [],
                'SlnL': [],
                'SlnR': [],
                'ModL': [],
                'ModR': []
            }
            self.in_event = True
            self.event_start_time = timestamp 
            return
        
        # 如果不在事件中，忽略所有行
        if not self.in_event:
            return
        
        # 检查是否是结束事件
        if "Reset!" in content:
            self.current_event['Time_Range_second']=[int(self.event_start_time/50000),int(timestamp/50000)]
            if self.current_event:  # 确保有事件数据
                self.events.append(dict(self.current_event))
                self.current_event = None
            self.in_event = False
            self.event_start_time = None 
            return
        
        # 处理泵信息
        self._process_pump_info(content, timestamp-self.event_start_time if self.event_start_time else timestamp)
        
        # 处理切换类型信息
        self._process_switch_type(content, timestamp-self.event_start_time if self.event_start_time else timestamp)
    
    def _process_pump_info(self, content, timestamp):
        pump_pattern = r'Pump (Left|Right) - lick (correct|wrong)(?: without reward)?'
        matches = re.findall(pump_pattern, content)
        
        for direction, lick_type in matches:
            key = None
            if "without reward" in content:
                # 没有奖励，记录到ROM
                key = f'ROm{direction[0]}'  # Left->L, Right->R
            else:
                # 有奖励，记录到SLN
                key = f'Sln{direction[0]}'
            
            if key:
                self.current_event[key].append(timestamp)
    
    def _process_switch_type(self, content, timestamp):
        if "Switch Type!" in content:
            trial_match = re.search(r'Trial Type is (Left|Right)', content)
            if trial_match:
                direction = trial_match.group(1)
                key = f'Mod{direction[0]}'  # Left->L, Right->R
                self.current_event[key].append(timestamp)
    
    def get_events(self):
        return self.events


def process_log_file(file_path):
    processor = EventProcessor()
    current_entry = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是新的日志行（以年份开头）
            if line.startswith('202'):
                # 处理之前收集的完整日志条目
                if current_entry:
                    full_entry = ' '.join(current_entry)
                    processor.process_line(full_entry)
                    current_entry = []
                # 开始新的日志条目
                current_entry.append(line)
            else:
                current_entry.append(line)
        
        # 处理文件末尾的最后一条日志
        if current_entry:
            full_entry = ' '.join(current_entry)
            processor.process_line(full_entry)
    
    return processor.get_events()

