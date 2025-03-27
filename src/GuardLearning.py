from sklearn.svm import SVC
import numpy as np

from src.CurveSlice import Slice
from src.Reset import ResetFun


def guard_learning(data: list[Slice], get_feature, config):
    positive_sample = {}
    negative_sample = {}
    slice_data = {}
    need_reset = config['need_reset']
    for i in range(len(data)):
        if not data[i].valid:
            continue
        mode = data[i].mode
        if negative_sample.get(mode) is None:
            negative_sample[mode] = []
        negative_sample[mode].append(np.transpose(data[i].data[:, :-1]))
        if i == 0 or not data[i - 1].valid or data[i].isFront:
            continue
        idx = (data[i - 1].mode, data[i].mode)
        if positive_sample.get(idx) is None:
            positive_sample[idx] = []
            slice_data[idx] = [[], []] if need_reset else None
        positive_sample[idx].append(data[i - 1].data[:, -1])
        if need_reset:
            slice_data[idx][0].append(data[i - 1])
            slice_data[idx][1].append(data[i])
    for (key, val) in negative_sample.items():
        negative_sample[key] = np.concatenate(val)

    if need_reset:
        for key, val in slice_data.items():
            slice_data[key] = ResetFun.from_slice(get_feature, val[0], val[1])

    adj = {}
    for (u, v), sample in positive_sample.items():
        svc = SVC(C=config['svm_c'], kernel=config['kernel'], class_weight={0: config['class_weight'], 1: 1})
        label = np.concatenate((np.zeros(negative_sample[u].shape[0]), np.ones(len(positive_sample[(u, v)]))))
        sample = np.concatenate((negative_sample[u], positive_sample[(u, v)]))
        svc.fit(sample, label)
        adj[(u, v)] = (svc, slice_data[(u, v)])
    return adj
