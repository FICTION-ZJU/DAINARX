import time

import numpy as np
from collections import deque
from math import *
import matplotlib.pyplot as plt
import re


class ODE:
    @staticmethod
    def analyticalExpression(expr, val_name):
        general = re.sub(val_name, 'x', expr)
        prefix = re.sub("=(.)*", "", general)
        suffix = re.sub("(.)*=", "", general)
        dim = int(re.findall(r'\d+', prefix)[0])
        return dim, suffix

    def __init__(self, eq: str, init_state=None, var_name='x', method='e'):
        r"""
        :param eq:
            Expression of ordinary differential equation.
            The k derivative of x is expressed as x[k].
            The left side of the equal sign is the highest derivative of the equation,
            and the right side is the expression.
            Implicit functions are not supported.
        :param init_state:
            The initial state of the variable,
            init_state[k] represents the initial state of the k derivative of the variable.
            The default value is 0 for all dimensions.
            If the number of dimensions is less than +1, 0 is filled backward.
        :param var_name:
            The name of the variable in the expression, default is 'x'.
        """
        if init_state is None:
            init_state = []

        try:
            dim, expr = ODE.analyticalExpression(eq, var_name)
            self.dim = dim
            self.eq = eval("lambda x: " + expr)
            self.state = np.array(init_state).astype(np.float64)
            self.fit_state_size()
            # check eq
            self.eq(self.state)
        except Exception as ex:
            raise Exception("invalid ode \"{}\"".format(eq), ex)
        self.method = method
        self.init_state = self.state.copy()

    def fit_state_size(self):
        self.state = np.resize(self.state, (self.dim,)).astype(np.float64)

    def rk_fun(self, y):
        res = np.roll(y, -1)
        res[self.dim - 1] = self.eq(y)
        return res

    def next(self, dT: float):
        if self.method == "rk":
            k1 = self.rk_fun(self.state)
            k2 = self.rk_fun(self.state + 0.5 * dT * k1)
            k3 = self.rk_fun(self.state + 0.5 * dT * k2)
            k4 = self.rk_fun(self.state + dT * k3)
            self.state = self.state + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            last_d = self.eq(self.state)
            for i in range(self.dim - 1, -1, -1):
                self.state[i] = self.state[i] + dT * last_d
                last_d = self.state[i]
        return self.state[0]

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state.copy()
        else:
            self.state = init_state.copy()
        self.fit_state_size()

    def load(self, other_ode):
        self.state = other_ode.state.copy()
        self.fit_state_size()


class DE:
    @staticmethod
    def analyticalExpression(expr_list: str):
        if expr_list == '':
            return []
        expr_list = expr_list.split(',')
        fun_list = []
        for expr in expr_list:
            if expr.isspace():
                continue
            fun_list.append(eval("lambda x: " + expr))
        return fun_list

    def __init__(self, eq, init_state, has_bias=False, other_item: str = ''):
        # if type(init_state) == np.array:
        #     init_state = init_state.tolist()
        self.init_state = init_state
        self.state = deque(init_state)
        self.eq = eq.copy()
        self.fun_list = DE.analyticalExpression(other_item)
        self.dim = len(eq) - len(self.fun_list) - int(has_bias)
        self.has_bias = has_bias
        self.fit_state_size()

    def fit_state_size(self):
        while len(self.state) > self.dim:
            self.state.pop()
        while len(self.state) < self.dim:
            self.state.append(0)

    def next(self):
        val = 0
        for i in range(self.dim):
            val += self.state[i] * self.eq[i]
        self.state.appendleft(0)
        for fun_idx in range(len(self.fun_list)):
            k = self.eq[fun_idx + self.dim]
            val += self.fun_list[fun_idx](self.state) * k
            fun_idx += 1
        if self.has_bias:
            val += self.eq[-1]
        self.state[0] = val
        self.state.pop()
        return val

    def clear(self, init_state=None):
        if init_state is None:
            self.state = deque(self.init_state.copy())
        else:
            self.state = deque(init_state)
        self.fit_state_size()

    def load(self, other_ed):
        self.state = other_ed.state.copy()
        self.fit_state_size()


if __name__ == "__main__":
    ode = ODE("x1[1] = 2 * x1[0]", var_name='x1', method='e', init_state=[21])
    # ode = DE([1, 1], [1, 1])
    X = []
    Y = []
    start = time.time()
    for pos in range(500):
        X.append(pos)
        Y.append(ode.next(0.01))
    print(time.time() - start)
    print(Y[:10])
    plt.plot(X, Y)
    plt.show()
