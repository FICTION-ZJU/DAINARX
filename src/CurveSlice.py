import numpy as np


class Slice:
    RelativeErrorThreshold = []
    AbsoluteErrorThreshold = []
    ToleranceRatio = 0.5

    @staticmethod
    def get_dis(v1, v2):
        dis = np.linalg.norm(v1 - v2, ord=1)
        d1 = np.linalg.norm(v1, ord=1)
        d2 = np.linalg.norm(v2, ord=1)
        d_min = min(d1, d2)
        relative_dis = dis / max(d_min, 1e-6)
        return relative_dis, dis

    @staticmethod
    def fit_threshold_one(feature1, feature2):
        assert len(feature1) == len(feature2)
        while len(Slice.RelativeErrorThreshold) < len(feature1):
            Slice.RelativeErrorThreshold.append(1e-1)
            Slice.AbsoluteErrorThreshold.append(1e-1)
        idx = 0
        for v1, v2 in zip(feature1, feature2):
            relative_dis, dis = Slice.get_dis(v1, v2)
            if relative_dis > 1e-4:
                Slice.RelativeErrorThreshold[idx] = \
                    min(Slice.RelativeErrorThreshold[idx], relative_dis * Slice.ToleranceRatio)
            if dis > 1e-4:
                Slice.AbsoluteErrorThreshold[idx] = \
                    min(Slice.AbsoluteErrorThreshold[idx], dis * Slice.ToleranceRatio)
            idx += 1
        return True

    @staticmethod
    def fit_threshold(data: list):
        for i in range(len(data)):
            if data[i].isFront:
                continue
            Slice.fit_threshold_one(data[i].feature, data[i - 1].feature)

    def __init__(self, data, get_feature, isFront):
        self.data = data
        self.get_feature = get_feature
        self.feature = np.array(get_feature(data)).astype(np.float64)
        self.mode = None
        self.isFront = isFront

    def setMode(self, mode):
        self.mode = mode

    def __and__(self, other):
        _, err = self.get_feature([self.data, other.data], is_list=True, need_err=True)
        return np.sum(err) < 0.001


def slice_curve(cut_data, data, change_points, get_feature):
    last = 0
    for point in change_points:
        if point == 0:
            continue
        cut_data.append(Slice(data[:, last:point], get_feature, last == 0))
        last = point
    return cut_data
