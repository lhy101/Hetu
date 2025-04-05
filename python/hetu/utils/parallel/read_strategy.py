import json
from ast import literal_eval

def read_strategy(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # 转换rank_to_device_mapping的key为整数
    config['rank_to_device_mapping'] = {
        int(k): v for k, v in config['rank_to_device_mapping'].items()
    }
    
    # 确保所有列表类型正确
    list_fields = [
        'cp_list', 'layers_num_list', 'micro_batch_num_list',
        'unused_rank', 'seq_len_list'
    ]
    for field in list_fields:
        if field in config and isinstance(config[field], str):
            config[field] = literal_eval(config[field])
    
    return config