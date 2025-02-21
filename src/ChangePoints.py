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


def find_change_point(data: np.array, get_feature, w: int = 10, merge_th=None):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param merge_th: Change point merge threshold. The default value is w.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points = []
    error_datas = []
    tail_len = 0
    pos = 0
    last = None
    if merge_th is None:
        merge_th = w

    while pos + w < data.shape[1]:
        feature, now_err, fit_dim = get_feature(data[:, pos:(pos + w)])
        if last is not None:
            if (max(now_err) > 1e-8 or fit_dim != last) and tail_len == 0:
                change_points.append(pos + w - 2)
                tail_len = w
            tail_len = max(tail_len - 1, 0)
        last = fit_dim
        pos += 1

    res = mergeChangePoints(change_points, merge_th)
    res.append(data.shape[1])
    res.insert(0, 0)

    return res
