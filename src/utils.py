import numpy as np

from src.CurveSlice import Slice
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


if __name__ == '__main__':
    a = ['A1', 'A2', 'A3', 'A4']
    b = ['B1', 'B2', 'B3', 'B4']

    matching_result = max_bipartite_matching(a, b)
    print(matching_result)
