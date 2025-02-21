from sklearn.svm import SVC
import numpy as np

from src.CurveSlice import Slice


def guard_learning(data: list[Slice], kernel='linear', class_weight=1.0):
    positive_sample = {}
    negative_sample = {}
    for i in range(len(data)):
        mode = data[i].mode
        if negative_sample.get(mode) is None:
            negative_sample[mode] = []
        negative_sample[mode].append(np.transpose(data[i].data))
        if i == 0 or data[i - 1].mode is None or data[i].isFront:
            continue
        idx = (data[i - 1].mode, data[i].mode)
        if positive_sample.get(idx) is None:
            positive_sample[idx] = []
        positive_sample[idx].append(data[i].data[:, 0])
    for (key, val) in negative_sample.items():
        negative_sample[key] = np.concatenate(val)

    adj = {}
    for (u, v), sample in positive_sample.items():
        svc = SVC(C=1e6, kernel=kernel, class_weight={0: class_weight, 1: 1})
        label = np.concatenate((np.zeros(negative_sample[u].shape[0]), np.ones(len(positive_sample[(u, v)]))))
        sample = np.concatenate((negative_sample[u], positive_sample[(u, v)]))
        svc.fit(sample, label)
        adj[(u, v)] = svc
    return adj
