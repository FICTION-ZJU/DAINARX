import numpy as np
from src.DE import DE


def mergeChangePoints(data, th: float):
    data = np.unique(np.sort(data))
    res = []
    last = None
    for pos in data:
        if last is None or pos - last > th:
            res.append(pos)
        last = pos
    return res


def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=None):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param input_data: Input of system.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param merge_th: Change point merge threshold. The default value is w.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points = []
    if merge_th is None:
        merge_th = w

    last_chp = 0
    while last_chp < data.shape[1]:
        lp = get_feature.dim + 1
        rp = data.shape[1] - last_chp
        while lp < rp:
            mid = (lp + rp + 1) // 2
            # print(mid)
            feature, now_err, fit_dim = get_feature(data[:, last_chp:(last_chp + mid)],
                                                    input_data[:, last_chp:(last_chp + mid)])
            eps = get_feature.get_eps(data[:, last_chp:(last_chp + mid)])
            if max(now_err) > eps:
                rp = mid - 1
            else:
                lp = mid
        last_chp += lp
        change_points.append(last_chp)

    if w < 0:
        tail_len = 0
        pos = 0
        last = None
        while pos + w < data.shape[1]:
            feature, now_err, fit_dim = get_feature(data[:, pos:(pos + w)], input_data[:, pos:(pos + w)])
            if last is not None:
                if (max(now_err) > eps) and tail_len == 0:
                    change_points.append(pos + w - 1)
                    tail_len = w
                tail_len = max(tail_len - 1, 0)
            last = fit_dim
            pos += 1

    res = mergeChangePoints(change_points, get_feature.dim + 1)
    if res[-1] != data.shape[1]:
        res.append(data.shape[1])
    res.insert(0, 0)

    return res
