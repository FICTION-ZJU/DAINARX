import json
import logging
import re

from CreatData import creat_data
from src.DEConfig import FeatureExtractor
from src.utils import *

from src.CurveSlice import Slice, slice_curve
from src.ChangePoints import find_change_point
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state
from src.Evaluation import Evaluation
from src.HybridAutomata import HybridAutomata


import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def run(data_list, input_data, config, evaluation: Evaluation):
    input_data = np.array(input_data)
    get_feature = FeatureExtractor(len(data_list[0]), len(input_data[0]),
                                   dim=config['dim'], dt=config['dt'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'])
    Slice.clear()
    slice_data = []
    chp_list = []
    for data, input_val in zip(data_list, input_data):
        change_points = find_change_point(data, input_val, get_feature, w=config['window_size'])
        chp_list.append(change_points)
        print("ChP:\t", change_points)
        slice_curve(slice_data, data, input_val, change_points, get_feature)
    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")
    Slice.Method = config['clustering_method']
    Slice.fit_threshold(slice_data)
    clustering(slice_data, config['self_loop'])
    evaluation.recording_time("clustering")
    adj = guard_learning(slice_data, get_feature, config)
    evaluation.recording_time("guard_learning")
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    evaluation.submit(slice_data=slice_data)
    return sys, slice_data


def get_config(json_path, evaluation: Evaluation):
    logging.basicConfig(level=logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    default_config = {'dt': 0.01, 'total_time': 10, 'dim': 3, 'window_size': 10, 'clustering_method': 'fit',
                      'minus': False, 'need_bias': True, 'other_items': '', 'kernel': 'linear', 'svm_c': 1e6,
                      'class_weight': 1.0, 'need_reset': False, 'self_loop': False}
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
    HybridAutomata.LoopWarning = not config['self_loop']
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
    input_list = []
    gt_list = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    for root, dirs, files in os.walk(data_path):
        print("Loading data!")
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group())):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            gt_list.append(change_point_list)
            print("GT:\t", change_point_list.tolist())
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)
            input_list.append(npz_file['input'])

    plot_with_mode(data[0][0], mode_list[0])

    test_num = 6

    print("Be running!")
    evaluation.submit(gt_chp=gt_list[test_num:])
    evaluation.submit(train_mode_list=mode_list[test_num:])
    evaluation.start()
    sys, slice_data = run(data[test_num:], input_list[test_num:], config, evaluation)
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")
    all_fit_mode, all_gt_mode = get_mode_list(slice_data, mode_list[test_num:])
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_list[:test_num]

    init_state_test = get_init_state(data_test, mode_map, mode_list_test, config['dim'])
    fit_data_list, mode_data_list = [], []
    draw_index = 0  # 为None则全画
    for data, mode_list, input_list, init_state in zip(data_test, mode_list_test, input_list_test, init_state_test):
        fit_data = [data[:, i] for i in range(config['dim'])]
        mode_data = list(mode_list[:config['dim']])
        sys.reset(init_state, input_list[:, :config['dim']])
        for i in range(config['dim'], data.shape[1]):
            state, mode, switched = sys.next(input_list[:, i])
            fit_data.append(state)
            mode_data.append(mode_map_inv[mode])
        fit_data = np.array(fit_data)
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)
        if need_plot and (draw_index == 0 or draw_index is None):
            need_plot = not need_plot
            for var_idx in range(data.shape[0]):
                plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
                plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
                plt.show()
            plt.plot(np.arange(len(mode_list)), mode_list, color='c')
            plt.plot(np.arange(len(mode_data)), mode_data, color='r')
            plt.show()
        if draw_index is not None:
            draw_index -= 1
    evaluation.submit(fit_mode=mode_data_list, fit_data=np.array(fit_data_list),
                      gt_mode=mode_list_test, gt_data=data_test, dt=config['dt'])
    return evaluation.calc()


if __name__ == "__main__":
    eval_log = main("./automata/non_linear/duffing.json")
    print("Evaluation log:")
    for key_, val_ in eval_log.items():
        print(f"{key_}: {val_}")
