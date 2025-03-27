import re

import numpy as np
from src.DE import ODE
from src.ODE_System import ODESystem
import json
import matplotlib.pyplot as plt
from math import *


class Node:
    def __init__(self, var_list, ode_list):
        self.var_list = var_list.copy()
        self.ode_list = ode_list

    @classmethod
    def from_str(cls, var_list, eq_list):
        ode_list = []
        for var in var_list:
            fl = False
            for eq in eq_list:
                if var not in eq:
                    continue
                fl = True
                ode_list.append(ODE(eq, var_name=var, method='rk'))
                break
            if not fl:
                raise Exception("There is no matching equation for variable {}.".format(var))
        return cls(var_list, ode_list)

    def next(self, *args):
        res = []
        for ode in self.ode_list:
            res.append(ode.next(*args))
        return res

    def load(self, other_node):
        for i in range(len(self.var_list)):
            self.ode_list[i].load(other_node.ode_list[i])

    def reset(self, init_state):
        for i in range(len(self.var_list)):
            self.ode_list[i].clear(init_state.get(self.var_list[i]))


class HybridAutomata:
    LoopWarning = True

    def __init__(self, mode_list, adj, init_mode=None):
        if init_mode is None:
            self.mode_state = None
        else:
            self.mode_state = init_mode
        self.mode_list = mode_list
        self.adj = adj

    @classmethod
    def from_json(cls, info: dict):
        var_list = re.split(r"\s*,\s*", info['var'])
        input_expr = info.get('input')
        if input_expr is None:
            input_list = []
        else:
            input_list = re.split(r"\s*,\s*", info.get('input'))
        mode_list = {}

        adj = {}
        for mode in info['mode']:
            mode_id = mode['id']
            mode_list[mode_id] = ODESystem(mode['eq'], var_list, input_list)
            adj[mode_id] = []
        for edge in info['edge']:
            u_v = re.findall(r'\d+', edge['direction'])
            fun = eval('lambda ' + info['var'] + ':' + edge['condition'])
            reset_val = edge.get("reset", {})
            adj[int(u_v[0])].append((int(u_v[1]), fun, reset_val))
        return cls(mode_list, adj)

    def getInput(self):
        return self.mode_list[self.mode_state].getInput()

    def next(self, *args):
        res = list(self.mode_list[self.mode_state].next(*args))
        mode_state = self.mode_state
        vis = set()
        via_list = []
        is_cycle = False
        switched = False
        while True:
            fl = True
            for to, fun, reset_val in self.adj.get(self.mode_state, {}):
                if fun(*res):
                    # self.mode_list[to].load(self.mode_list[self.mode_state], reset_val)
                    self.mode_state = to
                    switched = True
                    if to in vis:
                        if HybridAutomata.LoopWarning:
                            print("warning: find loop!")
                        is_cycle = True
                    vis.add(to)
                    via_list.append((to, reset_val))
                    fl = False
                    break
            if fl or is_cycle:
                if len(via_list) != 0:
                    to, reset_val = via_list[0] if is_cycle else via_list[-1]
                    self.mode_list[to].load(self.mode_list[mode_state], reset_val)
                    self.mode_state = to
                break
        return res, mode_state, switched

    def reset(self, init_state, *args):
        self.mode_state = init_state.get('mode', self.mode_state)
        self.mode_list[self.mode_state].reset(init_state, *args)


if __name__ == "__main__":
    with open('../automata/test.json', 'r') as f:
        data = json.load(f)
        sys = HybridAutomata.from_json(data['automation'])
        res = []
        for i in range(1000):
            res.append(sys.next(0.01)[0])
        res = np.array(res)
        x2_data = res[:, 1]
        plt.plot(np.arange(0, len(x2_data)), x2_data)
        plt.show()
