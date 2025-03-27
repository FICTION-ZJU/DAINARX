import numpy as np

from src.DEConfig import FeatureExtractor
from src.CurveSlice import Slice


class ResetFun:
    def __init__(self, get_feature, fit_dim, eq_list):
        self.get_feature = get_feature
        self.fit_dim = fit_dim
        self.eq_list = eq_list
        self.cnt = 0

    @classmethod
    def from_slice(cls, get_feature: FeatureExtractor, f_list: list[Slice], t_list: list[Slice]):
        fit_dim = max(max(s.fit_dim for s in t_list))
        eq_list = []
        for i in range(fit_dim):
            this_data = []
            this_input = []
            for f_slice, t_slice in zip(f_list, t_list):
                this_data.append(np.concatenate((f_slice.data[:, -(get_feature.dim - i):],
                                                 t_slice.data[:, :(i + 1)]), axis=1))
                this_input.append(np.concatenate((f_slice.input_data[:, -(get_feature.dim - i):],
                                                 t_slice.input_data[:, :(i + 1)]), axis=1))
            eq_list.append(get_feature(this_data, this_input, is_list=True)[0])
        return cls(get_feature, fit_dim, eq_list)

    def __call__(self, state, input_data, var_num):
        res = []
        for i in range(var_num):
            res.append(np.dot(self.get_feature.get_items(state, input_data, i), self.eq_list[self.cnt][i]))
        self.cnt += 1
        return res

    def valid(self):
        return self.cnt < self.fit_dim

    def clear(self):
        self.cnt = 0

