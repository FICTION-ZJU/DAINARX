import numpy as np

from src.CurveSlice import Slice
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import os


def get_ture_chp(data):
    last = None
    change_points = [0]
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx)
        idx += 1
        last = now
    change_points.append(len(data))
    return change_points


def dict_hash(obj):
    sorted_items = sorted(obj.items())
    hash_obj = hashlib.sha256()
    for item in sorted_items:
        hash_obj.update(str(item).encode())
    return hash_obj.hexdigest()


def get_hash_code(json_file, config):
    mark = json_file.copy()
    if mark.get('config') is not None:
        mark.pop('config')
    mark['dt'] = config['dt']
    mark['total_time'] = config['total_time']
    return dict_hash(mark)


def check_data_update(hash_code, data_path):
    hash_path = os.path.join(data_path, 'info.sha256')
    ori_hash_code = ''
    if os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            ori_hash_code = f.read()
            f.close()
    return ori_hash_code != hash_code


def save_hash_code(hash_code, data_path):
    hash_path = os.path.join(data_path, 'info.sha256')
    with open(hash_path, 'w') as f:
        f.write(hash_code)
        f.close()


def max_bipartite_matching(fit_mode, gt_mode):

    cnt = {}
    edges = []
    for mode_a, mode_b in zip(fit_mode, gt_mode):
        if mode_b is not None and mode_b >= 0:
            cnt[(mode_a, -mode_b)] = cnt.get((mode_a, mode_b), 0) + 1

    G = nx.Graph()
    l_nodes, r_nodes = [], []
    for (f, t), num in cnt.items():
        l_nodes.append(f)
        r_nodes.append(t)
        edges.append((f, t, {'weight': num}))
    G.add_nodes_from(list(set(l_nodes)), bipartite=0)
    G.add_nodes_from(list(set(r_nodes)), bipartite=1)
    G.add_edges_from(edges)
    matching = nx.max_weight_matching(G, weight='weight')
    res = {}
    res_inv = {}
    for (f, t) in matching:
        res[-f] = t
        res_inv[t] = -f
    return res, res_inv


def get_mode_list(slice_data: list[Slice], gt_mode_list):
    all_fit_mode = []
    for slice in slice_data:
        if slice.mode is None:
            all_fit_mode.append(np.full(slice.length, -1))
        else:
            all_fit_mode.append(np.full(slice.length, slice.mode))
    all_fit_mode = np.concatenate(all_fit_mode)
    all_gt_mode = np.concatenate(gt_mode_list)
    return all_fit_mode, all_gt_mode


def split_into_segments(mode):
    segments = []
    n = len(mode)
    if n == 0:
        return segments
    start = 0
    current_mode = mode[0]
    for i in range(1, n):
        if mode[i] != current_mode:
            segments.append((start, i, current_mode))
            start = i
            current_mode = mode[i]
    segments.append((start, n, current_mode))
    return segments


def plot_with_mode(data, mode, show=True):
    # 示例数据

    segments = split_into_segments(mode)

    # 创建颜色映射
    unique_modes = sorted(set(mode))
    colors = plt.cm.get_cmap('tab10', len(unique_modes))
    mode_to_color = {m: colors(idx) for idx, m in enumerate(unique_modes)}

    plt.figure(figsize=(10, 6))

    last_mode = None

    for start, end, m in segments:
        if last_mode is not None:
            plt.plot([start - 1, start], data[(start - 1):(start + 1)],
                     color=mode_to_color[last_mode], linewidth=2,
                     label=f'Mode {last_mode}')
        plt.plot(range(start, end), data[start:end],
                 color=mode_to_color[m], linewidth=2, label=f'Mode {m}')
        last_mode = m

    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Data Value', fontsize=12)
    plt.title('Data Segmented by Mode with Colored Lines', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.7)

    # 处理图例去重
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if show:
        plt.show()


if __name__ == '__main__':
    data = [5, 3, 8, 4, 7]
    mode = [0, 0, 1, 1, 0]
    plot_with_mode(data, mode)
