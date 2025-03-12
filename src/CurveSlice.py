import numpy as np


class Slice:
    RelativeErrorThreshold = []
    AbsoluteErrorThreshold = []
    ToleranceRatio = 0.1
    FitErrorThreshold = 1.
    Method = 'fit'

    @staticmethod
    def clear():
        Slice.RelativeErrorThreshold = []
        Slice.AbsoluteErrorThreshold = []
        Slice.ToleranceRatio = 0.1
        Slice.FitErrorThreshold = 1.
        Slice.Method = 'fit'

    @staticmethod
    def get_dis(v1, v2):
        dis = np.linalg.norm(v1 - v2, ord=1)
        d1 = np.linalg.norm(v1, ord=1)
        d2 = np.linalg.norm(v2, ord=1)
        d_min = min(d1, d2)
        relative_dis = dis / max(d_min, 1e-6)
        return relative_dis, dis

    @staticmethod
    def fit_threshold_one(get_feature, data1, data2):
        feature1 = data1.feature
        feature2 = data2.feature
        assert len(feature1) == len(feature2)
        _, err, fit_dim = get_feature([data1.data, data2.data],
                                      [data1.input_data, data2.input_data], is_list=True)
        if fit_dim <= max(data1.fit_dim, data2.fit_dim):
            Slice.FitErrorThreshold = min(Slice.FitErrorThreshold, max(err) * Slice.ToleranceRatio)
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
                    min(Slice.AbsoluteErrorThreshold[idx], max(dis * Slice.ToleranceRatio, 1e-6))
            idx += 1
        return True

    @staticmethod
    def fit_threshold(data: list):
        for i in range(len(data)):
            if data[i].isFront:
                continue
            Slice.fit_threshold_one(data[i].get_feature, data[i], data[i - 1])

    def __init__(self, data, input_data, get_feature, isFront):
        self.data = data
        self.input_data = input_data
        self.get_feature = get_feature
        self.feature, _, self.fit_dim = get_feature(data, input_data)
        self.mode = None
        self.isFront = isFront

    def setMode(self, mode):
        self.mode = mode

    def test_set(self, data, input_data, other_fit_dim):
        _, err, fit_dim = self.get_feature([self.data] + data, [self.input_data] + input_data, is_list=True)
        dim_condition = True
        for i in range(len(fit_dim)):
            dim_condition = dim_condition and fit_dim[i] <= max(self.fit_dim[i], other_fit_dim[i])
        return dim_condition and max(err) < Slice.FitErrorThreshold

    def __and__(self, other):
        if Slice.Method == 'dis':
            idx = 0
            for v1, v2 in zip(self.feature, other.feature):
                relative_dis, dis = Slice.get_dis(v1, v2)
                if relative_dis > Slice.RelativeErrorThreshold[idx] and \
                        dis > Slice.AbsoluteErrorThreshold[idx]:
                    return False
                idx += 1
            return True
        else:
            _, err, fit_dim = self.get_feature([self.data, other.data],
                                               [self.input_data, other.input_data], is_list=True)
            dim_condition = True
            for i in range(len(fit_dim)):
                dim_condition = dim_condition and fit_dim[i] <= max(self.fit_dim[i], other.fit_dim[i])
            return dim_condition and max(err) < Slice.FitErrorThreshold


def slice_curve(cut_data, data, input_data, change_points, get_feature):
    last = 0
    for point in change_points:
        if point == 0:
            continue
        cut_data.append(Slice(data[:, last:point], input_data[:, last:point], get_feature, last == 0))
        last = point
    return cut_data
