import numpy as np
import bisect
import time
from src.utils import get_ture_chp


class Evaluation:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.last_time = None
        self.time_list = []
        self.data = {}

    def start(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def stop(self, name):
        self.time_list.append((name, time.time() - self.start_time))

    def recording_time(self, name):
        now = time.time()
        self.time_list.append((name, now - self.last_time))
        print(f"recording {name} time: {now - self.last_time}")
        self.last_time = now

    def submit(self, **args):
        self.data.update(args)

    def calc(self):
        res = {"name": self.name}
        fit_mode = get_ture_chp(self.data['fit_mode'])
        fit_data = self.data['fit_data']
        gt_mode = get_ture_chp(self.data['gt_mode'])
        gt_data = self.data['gt_data']
        dt = self.data['dt']
        diff = np.abs(fit_data - gt_data)

        for var_idx in range(diff.shape[0]):
            diff[var_idx] /= np.max(np.abs(gt_data[var_idx]))

        res["tc"] = max(max_min_abs_diff(fit_mode, gt_mode), max_min_abs_diff(gt_mode, fit_mode)) * dt

        train_tc = 0.0
        for chp, gt in zip(self.data["chp"], self.data["gt_chp"]):
            train_tc = max(train_tc, max_min_abs_diff(chp, gt), max_min_abs_diff(gt, chp)) * dt
        res["clustering_error"] = abs(self.data['gt_mode_num'] - self.data['mode_num'])
        res["train_tc"] = train_tc
        res["max_diff"] = np.max(diff)
        res["mean_diff"] = np.mean(diff)
        res["time"] = self.time_list.copy()

        return res


def max_min_abs_diff(a, b):
    sorted_b = sorted(b)
    max_diff = 0

    # 处理a中的每个元素，找到在sorted_b中的最小差
    for x in a:
        pos = bisect.bisect_left(sorted_b, x)
        if pos == 0:
            diff = abs(sorted_b[0] - x)
        elif pos == len(sorted_b):
            diff = abs(sorted_b[-1] - x)
        else:
            left = sorted_b[pos - 1]
            right = sorted_b[pos]
            diff = min(abs(x - left), abs(x - right))
        max_diff = max(max_diff, diff)

    return max_diff


def eva_trace(mode, trace, gt_mode, gt_trace, Ts):
    tc = max(max_min_abs_diff(mode, gt_mode), max_min_abs_diff(gt_mode, mode))
    mean_diff = np.mean(np.abs(trace - gt_trace))
    return tc * Ts, mean_diff



