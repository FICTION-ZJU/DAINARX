import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from CreatData import creat_data
from src.utils import *

from src.CurveSlice import Slice, slice_curve
from src.ChangePoints import FeatureExtractor, find_change_point
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state


def run(data_list, config):
    get_feature = FeatureExtractor(config['dim'], config['need_bias'], config['other_items'])
    slice_data = []
    for data in data_list:
        change_points, err_data = find_change_point(data, get_feature)
        print(change_points)
        slice_curve(slice_data, data, change_points, get_feature)
    Slice.fit_threshold(slice_data)
    # TODO: 修改聚类
    clustering(slice_data)
    adj = guard_learning(slice_data, config['kernel'])
    sys = build_system(slice_data, adj, get_feature, config['need_bias'], config['other_items'])
    return sys


def get_config(json_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    if json_path.isspace():
        config = {'dt': 0.01, 'total_time': 10, 'dim': 3,
                  'need_bias': False, 'other_items': '', 'kernel': 'linear'}
    else:
        with open(json_path) as f:
            json_file = json.load(f)
        config = json_file.get('config', {})
        config.setdefault('dt', 0.01)
        config.setdefault('total_time', 10)
        config.setdefault('dim', 3)
        config.setdefault('need_bias', False)
        config.setdefault('other_items', '')
        config.setdefault('kernel', 'linear')
        f.close()
    return config, get_hash_code(json_file, config)


def main(json_path: str, data_path='data', need_creat=None):
    config, hash_code = get_config(json_path)
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    if need_creat is None:
        need_creat = check_data_update(hash_code, data_path)
    if need_creat:
        print("Data being generated!")
        creat_data(json_path, data_path, config['dt'], config['total_time'])
        save_hash_code(hash_code, data_path)

    mode_list = []
    data = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    for root, dirs, files in os.walk(data_path):
        print("Loading data!")
        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['arr_0'], npz_file['arr_1']
            print("GT: ", get_ture_chp(mode_data_temp))
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)

    print("Be running!")
    sys = run(data[1:], config)

    print("Start simulation")
    fit_idx = 0
    data = data[fit_idx]
    mode_list = mode_list[fit_idx]

    init_state = get_init_state(data, mode_list, config['dim'])
    fit_data = [data[:, i] for i in range(config['dim'])]
    mode_data = []
    sys.reset(init_state)
    for i in range(data.shape[1] - config['dim']):
        state, mode = sys.next()
        fit_data.append(state)
        mode_data.append(mode)
    fit_data = np.array(fit_data)

    for var_idx in range(data.shape[0]):
        plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
        plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
        plt.show()


if __name__ == "__main__":
    main("./automata/test.json")
