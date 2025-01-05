import numpy as np
from collections import deque
from math import *
import matplotlib.pyplot as plt
import re


class ODESystem:
    @staticmethod
    def analyticalExpression(expr_list: str, var_list: list[str]):
        pattern = r'(' + '|'.join(map(re.escape, var_list)) + r')\[(\d+)\]'
        var_pattern = r'(\S+)\[(\d+)\]'

        def repl(match):
            string = match.group(1)
            number = match.group(2)
            idx = var_list.index(string)
            return f"x[{idx}][{number}]"

        def get_info(string):
            match = re.search(var_pattern, string)
            return var_list.index(match.group(1)), int(match.group(2))

        res = [(lambda x: 0) for _ in var_list]
        dim_list = [0 for _ in var_list]
        expr_list = expr_list.split(',')
        for expr in expr_list:
            var_expr, eq_expr = expr.split('=')
            eq_expr = re.sub(pattern, repl, eq_expr)
            idx, dim = get_info(var_expr)
            print(eq_expr)
            res[idx] = eval("lambda x: " + eq_expr)
            dim_list[idx] = dim
        return dim_list, res

    def __init__(self, expr_list, var_list, init_state=None, method='e'):
        if init_state is None:
            init_state = []
        dim_list, eq_list = ODESystem.analyticalExpression(expr_list, var_list)
        self.var_list = var_list
        self.dim_list = dim_list
        self.eq_list = eq_list
        self.var_num = len(eq_list)
        self.state = init_state.copy()
        self.init_state = init_state.copy()
        self.max_dim = max(self.dim_list)
        self.method = method
        self.fit_state_size()

    def fit_state_size(self):
        state = np.zeros((self.var_num, self.max_dim)).astype(np.float64)
        for idx_var in range(min(self.var_num, len(self.state))):
            for idx_dim in range(min(self.dim_list[idx_var], len(self.state[idx_var]))):
                state[idx_var][idx_dim] = self.state[idx_var][idx_dim]
        self.state = state

    def rk_fun(self, y):
        res = np.roll(y, -1, axis=1)
        for idx in range(self.var_num):
            res[idx][self.dim_list[idx] - 1] = self.eq_list[idx](y)
        return res

    def next(self, dT: float):
        if self.method == "rk":
            k1 = self.rk_fun(self.state)
            k2 = self.rk_fun(self.state + 0.5 * dT * k1)
            k3 = self.rk_fun(self.state + 0.5 * dT * k2)
            k4 = self.rk_fun(self.state + dT * k3)
            self.state = self.state + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            for idx_var in range(self.var_num):
                last_d = self.eq_list[idx_var](self.state)
                for i in range(self.dim_list[idx_var] - 1, -1, -1):
                    self.state[idx_var][i] = self.state[idx_var][i] + dT * last_d
                    last_d = self.state[idx_var][i]
        return self.state[:, 0]

    def reset(self, init_state):
        res = []
        for var in self.var_list:
            res.append(init_state.get(var, []))
        self.clear(res)

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state.copy()
        else:
            self.state = init_state.copy()
        self.fit_state_size()

    def load(self, other_sys):
        self.state = other_sys.state.copy()
        self.fit_state_size()


if __name__ == "__main__":
    eq = r"""x1[1] = x1[0] + (-271.6981 * x1[0] + -377.3585 * x2[0] + 377.3585 * 24) ,
                    x2[1] = x2[0] + (454.5455 * x1[0] + -45.4545 * x2[0] + 0 * 24)"""
    sys = ODESystem(eq, ['x1', 'x2'])
    res = []
    for i in range(100):
        res.append(sys.next(0.01))
    res = np.array(res)
    plt.plot(np.arange(res.shape[0]), res[:, 0])
    plt.show()
