import re

import numpy as np
from src.DE import DE


class FeatureExtractor:
    @staticmethod
    def unfoldItem(expr_list, idx, var_num):
        res = []
        expr_list = expr_list.copy()
        for expr in expr_list:
            expr = re.sub(r'x\[', 'x[' + str(idx) + '][', expr)
            expr = re.sub(r'x(\d)', r'x[\1]', expr)
            if 'x_' not in expr:
                res.append(expr)
                continue
            for i in range(var_num):
                if i == idx:
                    continue
                res.append(re.sub(r'x_\[', 'x[' + str(i) + '][', expr))
        return res

    @staticmethod
    def unfoldDigit(expr_list, dim):
        res = []
        for expr in expr_list:
            for i in range(dim):
                res.append(re.sub(r'\[\?', r"[" + str(i), expr))
        return res

    @staticmethod
    def extractValidExpression(expr_list, idx):
        res = []
        for expr in expr_list:
            if expr.isspace() or expr == '':
                continue
            expr = re.sub(r'\[(\d+)', lambda x: '[' + str(int(x.group(1)) - 1), expr)
            if ':' not in expr:
                res.append(expr)
            else:
                filed, s = expr.split(':')
                if 'x' + str(idx) in filed:
                    res.append(s)
        return res

    @staticmethod
    def findMaxDim(expr):
        numbers = re.findall(r']\[(\d+)', expr)
        return max([int(num) for num in numbers])

    @staticmethod
    def analyticalExpression(expr_list: str, var_num, dim):
        res = [[] for _ in range(var_num)]
        res_dim = [[] for _ in range(var_num)]
        expr_list = expr_list.split(';')
        for idx in range(var_num):
            expr = FeatureExtractor.extractValidExpression(expr_list, idx)
            expr = FeatureExtractor.unfoldDigit(expr, dim)
            expr = FeatureExtractor.unfoldItem(expr, idx, var_num)
            for s in expr:
                res[idx].append(eval('lambda x: ' + s))
                res_dim[idx].append(FeatureExtractor.findMaxDim(s) + 1)
        return res, res_dim

    def __init__(self, var_num: int, dim: int, need_bias: bool = False, minus: bool = False, other_items: str = ''):
        self.var_num = var_num
        self.dim = dim
        self.minus = minus
        self.need_bias = need_bias
        self.fun_list, self.fun_dim = FeatureExtractor.analyticalExpression(other_items, var_num, dim)

    def get_items(self, data, idx, max_dim=None):
        res = []
        if max_dim is None:
            max_dim = self.dim
        max_dim = min(self.dim, max_dim)
        for i in range(len(data[idx]) - self.dim, len(data[idx]) - self.dim + max_dim):
            res.append(data[idx][i])
        for i in range(len(data[idx]) - self.dim + max_dim, len(data[idx])):
            res.append(0.)
        for (fun, dim) in zip(self.fun_list[idx], self.fun_dim[idx]):
            if dim > max_dim:
                res.append(0.)
            else:
                res.append(fun(data))
        if self.need_bias:
            res.append(1.)
        return res

    def work_minus(self, data, is_list: bool):
        res = []
        err = []
        max_dim = []
        for idx in range(self.var_num):
            now_dim = 1
            while True:
                a, b = [], []
                if is_list:
                    for block in data:
                        self.append_data_only(a, b, block, idx, now_dim)
                else:
                    self.append_data_only(a, b, data, idx, now_dim)
                x = np.linalg.lstsq(a, b, rcond=None)[0]
                a, b = np.array(a), np.array(b)
                now_err = max(np.abs((a @ x) - b))
                if now_dim == self.dim or now_err < 1e-8:
                    res.append(x)
                    max_dim.append(now_dim)
                    err.append(now_err)
                    break
                now_dim += 1
        return res, err, max_dim

    def work_normal(self, data, is_list: bool):
        res = []
        err = []
        var_num = len(data) if not is_list else len(data[0])
        matrix_list = [[] for _ in range(var_num)]
        b_list = [[] for _ in range(var_num)]
        if is_list:
            for block in data:
                self.append_data(matrix_list, b_list, block)
        else:
            self.append_data(matrix_list, b_list, data)
        for a, b in zip(matrix_list, b_list):
            x = np.linalg.lstsq(a, b, rcond=None)[0]
            res.append(x)
            err.append(max(np.abs((a @ x) - b)))
        return res, err, [self.dim for _ in range(self.var_num)]

    def __call__(self, data, is_list=False, need_err=False):
        if self.minus:
            return self.work_minus(data, is_list)
        else:
            return self.work_normal(data, is_list)

    def append_data(self, matrix_list, b_list, data: np.array, max_dim=None):
        data = np.array(data)
        for i in range(len(data[0]) - self.dim):
            if i == 0:
                this_line = data[:, (self.dim - 1)::-1]
            else:
                this_line = data[:, (self.dim + i - 1):(i - 1):-1]
            for idx in range(len(this_line)):
                matrix_list[idx].append(self.get_items(this_line, idx, max_dim))
                b_list[idx].append(data[idx][i + self.dim])

    def append_data_only(self, matrix_a, b, data: np.array, idx, max_dim=None):
        data = np.array(data)
        for i in range(len(data[0]) - self.dim):
            if i == 0:
                this_line = data[:, (self.dim - 1)::-1]
            else:
                this_line = data[:, (self.dim + i - 1):(i - 1):-1]
            matrix_a.append(self.get_items(this_line, idx, max_dim))
            b.append(data[idx][i + self.dim])
