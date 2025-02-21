import json
import os
import re
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from CreatData import creat_data
from src.DEConfig import FeatureExtractor
from src.utils import *

from src.CurveSlice import Slice, slice_curve
from src.ChangePoints import find_change_point
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state
from src.Evaluation import eva_trace, Evaluation


def run(data_list, config, evaluation: Evaluation):
    get_feature = FeatureExtractor(len(data_list[0]), dim=config['dim'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'])
    Slice.clear()
    slice_data = []
    chp_list = []
    for data in data_list:
        change_points = find_change_point(data, get_feature, w=config['window_size'])
        chp_list.append(change_points)
        print("ChP:\t", change_points)
        slice_curve(slice_data, data, change_points, get_feature)
    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")
    Slice.Method = config['clustering_method']
    Slice.fit_threshold(slice_data)
    clustering(slice_data)
    evaluation.recording_time("clustering")
    adj = guard_learning(slice_data, config['kernel'], config['class_weight'])
    evaluation.recording_time("guard_learning")
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    return sys


def get_config(json_path, evaluation: Evaluation):
    logging.basicConfig(level=logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    default_config = {'dt': 0.01, 'total_time': 10, 'dim': 3, 'window_size': 10, 'clustering_method': 'fit',
                      'minus': False, 'need_bias': True, 'other_items': '', 'kernel': 'linear',
                      'class_weight': 1.0}
    config = {}
    if json_path.isspace() or json_path == '':
        config = default_config
    else:
        with open(json_path) as f:
            json_file = json.load(f)
            evaluation.submit(gt_mode_num=len(json_file.get('automation', {'mode': []})['mode']))
            json_config = json_file.get('config', {})
            for (key, val) in default_config.items():
                if key in json_config.keys():
                    config[key] = json_config.pop(key)
                else:
                    config[key] = val
            if len(json_config) != 0:
                raise Exception('Invalid parameter: ' + str(json_config))
            f.close()
    return config, get_hash_code(json_file, config)


def main(json_path: str, data_path='data', need_creat=None, need_plot=True):
    evaluation = Evaluation(json_path)
    config, hash_code = get_config(json_path, evaluation)
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
    gt_list = []

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
            gt_list.append(get_ture_chp(mode_data_temp))
            print("GT:\t", get_ture_chp(mode_data_temp))
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)

    train_idx = 1
    print("Be running!")
    evaluation.submit(gt_chp=gt_list[train_idx:])
    evaluation.start()
    sys = run(data[train_idx:], config, evaluation)
    print("Start simulation")
    fit_idx = 0
    data = data[fit_idx]
    mode_list = mode_list[fit_idx]

    init_state = get_init_state(data, mode_list, config['dim'])
    fit_data = [data[:, i] for i in range(config['dim'])]
    mode_data = list(mode_list[:config['dim']])
    sys.reset(init_state)
    for i in range(data.shape[1] - config['dim']):
        state, mode = sys.next()
        fit_data.append(state)
        mode_data.append(mode)
    fit_data = np.array(fit_data)
    evaluation.submit(mode_num=len(sys.mode_list))
    print(f"mode number: {len(sys.mode_list)}")

    if need_plot:
        for var_idx in range(data.shape[0]):
            plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
            plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
            plt.show()
        plt.plot(np.arange(len(mode_list)), mode_list, color='c')
        plt.plot(np.arange(len(mode_data)), mode_data, color='r')
        plt.show()
    evaluation.submit(fit_mode=mode_list, fit_data=np.transpose(fit_data),
                      gt_mode=mode_data, gt_data=data, dt=config['dt'])
    return evaluation.calc()


if __name__ == "__main__":
    eval_log = main("./automata/FaMoS/simple_heating_system.json")
    print("Evaluation log:")
    for key_, val_ in eval_log.items():
        print(f"{key_}: {val_}")
