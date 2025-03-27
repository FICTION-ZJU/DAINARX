import numpy as np
from collections import deque
from math import *
from src.DEConfig import FeatureExtractor
from src.Reset import ResetFun
import matplotlib.pyplot as plt
import re


class DESystem:
    def __init__(self, eq, init_state, input_data, config: FeatureExtractor, reset_fun: ResetFun = None):
        self.var_num = len(eq)
        self.config = config
        self.init_state = init_state.copy()
        self.state = init_state.copy()
        self.input_data = input_data.copy()
        self.eq = eq.copy()
        self.fit_state_size()
        self.var_list = [('x' + str(idx)) for idx in range(self.var_num)]
        self.reset_fun = reset_fun

    def fit_state_size(self):
        state = [deque(0 for _ in range(self.config.dim)) for _ in range(self.var_num)]
        input_data = [deque(0 for _ in range(self.config.dim)) for _ in range(self.config.input_num)]
        for i in range(min(len(self.state), self.var_num)):
            for j in range(min(len(self.state[i]), self.config.dim)):
                state[i][j] = self.state[i][j]
        for i in range(min(len(self.input_data), self.config.input_num)):
            for j in range(min(len(self.input_data[i]), self.config.dim)):
                input_data[i][j] = self.input_data[i][j]
        self.state = state
        self.input_data = input_data

    def next(self, input_val=None):
        if input_val is None:
            input_val = []
        res = []
        for i in range(self.config.input_num):
            self.input_data[i].appendleft(input_val[i])
        if self.reset_fun is None or not self.reset_fun.valid():
            for i in range(self.var_num):
                res.append(np.dot(self.config.get_items(self.state, self.input_data, i), self.eq[i]))
        else:
            res = self.reset_fun(self.state, self.input_data, self.var_num)
        for i in range(self.var_num):
            self.state[i].pop()
            self.state[i].appendleft(res[i])
        for i in range(self.config.input_num):
            self.input_data[i].pop()
        return res

    def reset(self, init_state, input_data, reset_fun=None):
        res = []
        for var in self.var_list:
            res.append(init_state.get(var, []))
        self.clear(res, input_data)
        if reset_fun is not None:
            reset_fun.clear()
        self.reset_fun = reset_fun

    def clear(self, init_state, input_data):
        self.state = init_state.copy()
        self.input_data = input_data.copy()
        self.fit_state_size()

    def load(self, other, reset_fun=None):
        self.state = other.state.copy()
        self.input_data = other.input_data.copy()
        self.fit_state_size()
        if reset_fun is not None:
            reset_fun.clear()
        self.reset_fun = reset_fun

