
def read_event_file(event_fn):
    import os
    import struct
    # 获取文件信息和文件大小
    file_info = os.stat(event_fn)
    file_size = file_info.st_size

    EVENTSZ = 12
    event_count = (file_size // EVENTSZ) - 2

    if event_count > 0:
        vt_timestamps = [0] * event_count
        ttls = [0] * event_count
        with open(event_fn, 'rb') as fid:
            fid.seek(EVENTSZ, os.SEEK_SET)
            for n_event in range(event_count):
                vt_timestamps[n_event], ttls[n_event] = struct.unpack('QI', fid.read(EVENTSZ))
        st_event_data = {
            'vtTimestamps': vt_timestamps,
            'TTLs': ttls
        }
        return st_event_data
    else:
        return None



def extract_event(st_event_data, bits):
    import numpy as np
    bit_val = [bit >> bits[0] & 1 for bit in st_event_data['TTLs']]
    init_val = bits[1]
    de_bit_val = np.diff(np.concatenate(([init_val], bit_val)))

    if init_val == 0:
        lg_s = [i for i, x in enumerate(de_bit_val) if x == 1]
        lg_e = [i for i, x in enumerate(de_bit_val) if x == -1]
    else:
        lg_s = [i for i, x in enumerate(de_bit_val) if x == -1]
        lg_e = [i for i, x in enumerate(de_bit_val) if x == 1]
    event = {
        'Time_S': [st_event_data['vtTimestamps'][i] for i in lg_s],
        'Time_E': [st_event_data['vtTimestamps'][i] for i in lg_e]
    }
    return event



def ParseEvent(event_fn, st_events_bits=None):
    if st_events_bits is None:
        st_events_bits = {
            'CamT'  :[1 , 0],
            'LickL' :[3 , 0],
            'LickR' :[18, 0],
            'SlnR'  :[23, 0],
            'SlnL'  :[24, 0],
            'ROmL'  :[22, 0],
            'ROmR'  :[21, 0],
            'ModL'  :[17, 0],
            'ModR'  :[12, 0],
            'MinS'  :[9 , 1],
            'MinT'  :[15, 0]
        }

    st_event_data = read_event_file(event_fn)
    st_events = {}

    for event_field in st_events_bits:
        st_events[event_field] = extract_event(st_event_data, st_events_bits[event_field])
    return st_events
    
