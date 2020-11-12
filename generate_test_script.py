# /usr/bin/env python
import os
import re
import warnings
from pprint import pprint


raw_exps = """
baseline_gn_diou2_ssh_sgdr_iouaware 361
baseline_gn_diou2_ssh_sgdr 151
baseline_gn_diou2_sgdr 151
baseline_gn_sgdr 151
baseline_gn_diou2_ssh_sgdr_iouaware_reducenorm 31
baseline_gn_diou2_ssh_sgdr_iouaware_nonorm 31
baseline_gn_diou2_ssh_sgdr_iouaware_reducenorm_dcn 31
"""

exps = [x.split(' ') for x in raw_exps.split('\n') if x]
exps = {x[0]: f'epoch_{x[1]}' for x in exps}
# exps = {x[0]: 'latest' for x in exps}  # force use latest epoch

# daily
sub_list = [
    # common
    # (r'img_scale=\(9108, 1024\)', 'img_scale=(2150, 1600)'),
    # (r'dict\(type=\'Resize\', keep_ratio=True, keep_height=True\)', 'dict(type=\'Resize\', keep_ratio=True)'),
    (r'nms_pre=-?\d*', 'nms_pre=10000'),
    (r'max_per_img=-?\d*', 'max_per_img=80000'),
#     (r'score_thr=0\.02,', 'score_thr=0.02,\n    height_thresh=10,\n    pre_nms_filter=True,')
]

# suffix = '_2150_1600_nms10k_before_h10'
suffix = '_2150_1600_nms10k'

def check_is_test(config):
    data_cfg = re.search(r'data = dict\(.*\n\)', config, flags=re.DOTALL).group()

    train_cfg = re.search(r'\n\s*\#?\s*test\s*=\s*dict\(.*WIDER_train.*?pipeline=test_pipeline\)', data_cfg, flags=re.DOTALL).group()
    val_cfg = re.search(r'\n\s*?test\s*=\s*dict\(.*?WIDER_val.*?pipeline=test_pipeline\)', data_cfg, flags=re.DOTALL).group()
    assert '#' in train_cfg and '#' not in val_cfg, 'Please check if config is correct!'


def make_subs(config, sub_list, name=''):
    """
    Args:
        config: str
        sub_list: list((oldpattern, newpattern))
    """
    for (old, new) in sub_list:
        if not re.search(old, config):
            warnings.warn(f'Pattern "{old}" unmatched in config: {name}!')
            continue
        config = re.sub(old, new, config)
    return config

def gen_test_script(name, epoch='latest', gpu_id='0', suffix=''):
    print(f'test_kuro.sh --cfg {name} --suffix {suffix} --epoch {epoch} --gpu {gpu_id}')

for name, epoch in exps.items():
    config = open(f'configs/widerface/{name}.py').read()
    try:
        check_is_test(config)
    except Exception as e:
        print(name)
        raise e
#     print(name)
    new_config = make_subs(config, sub_list, name)
    # os.makedirs('configs/widerface/test_time', exist_ok=True)
    with open(f'configs/widerface/test_time/{name}{suffix}.py', 'w') as wf:
        wf.write(new_config)
    gen_test_script(name, epoch, suffix=suffix)

