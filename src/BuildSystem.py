import copy

import numpy as np
from src.CurveSlice import Slice
from src.HybridAutomata import Node, HybridAutomata
from src.DE import DE


class ModelFun:
    def __init__(self, model):
        self.model = copy.copy(model)

    def __call__(self, *args):
        return self.model.predict([[*args]])[0] > 0.5


def build_system(data: list[Slice], res_adj: dict, get_feature, has_bias=False, other_items=""):
    data_of_mode = {}
    for cur in data:
        if data_of_mode.get(cur.mode) is None:
            data_of_mode[cur.mode] = [[] for _ in range(len(cur.data))]
        for i in range(len(cur.data)):
            data_of_mode[cur.mode][i].append(cur.data[i])
    mode_list = {}
    for (mode, cur_list) in data_of_mode.items():
        var_list = []
        de_list = []
        # TODO: 耦合的差分方程
        for cur in cur_list:
            var_list.append('x' + str(len(var_list)))
            de_list.append(DE(get_feature(cur), [], has_bias, other_items))
        mode_list[mode] = Node(var_list, de_list)
    adj = {}
    for (u, v), model in res_adj.items():
        if adj.get(u) is None:
            adj[u] = []
        adj[u].append((v, ModelFun(model)))
    return HybridAutomata(mode_list, adj)


def get_init_state(data, mode_list, bias):
    # TODO: match correct mode
    init_state = {'mode': mode_list[bias - 1]}
    for i in range(data.shape[0]):
        init_state['x' + str(i)] = data[i, (bias - 1)::-1]
    return init_state
