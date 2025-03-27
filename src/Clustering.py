from src.CurveSlice import Slice


def clustering(data: list[Slice], self_loop=False):
    tot_mode = 1
    last_mode = None
    mode_dict = {}
    delay_list = []
    for i in range(len(data)):
        data[i].idx = i
        if not data[i].valid:
            continue
        if data[i].isFront:
            last_mode = None

        if Slice.Method == 'dis':
            for j in range(i):
                if (data[j].mode != last_mode) and (data[i] & data[j]):
                    data[i].setMode(data[j].mode)
                if data[i].mode is not None:
                    break
        else:
            fit_cnt = 0
            for idx, val in mode_dict.items():
                if idx != last_mode and data[i].test_set(val):
                    data[i].mode = idx
                    fit_cnt += 1
                elif len(val) == 2:
                    if data[i].test_set([val[0]]) or data[i].test_set([val[1]]):
                        delay_list.append(val[-1])
                        mode_dict[idx].pop()
                        fit_cnt = 2
            if fit_cnt == 1:
                if len(mode_dict[data[i].mode]) < 3:
                    mode_dict[data[i].mode].append(data[i])
            elif fit_cnt > 1:
                delay_list.append(data[i])
                data[i].mode = -1

        if data[i].mode is None:
            data[i].mode = tot_mode
            mode_dict[tot_mode] = [data[i]]
            tot_mode += 1
        if not self_loop:
            last_mode = data[i].mode
    if Slice.Method != 'fit':
        return
    for s in delay_list:
        for idx, val in mode_dict.items():
            if not s.isFront and data[s.idx - 1].mode == idx:
                continue
            if s.test_set(val):
                s.mode = idx
                if len(val) < 3:
                    mode_dict[idx].append(s)
                break
        if s.mode is None or s.mode == -1:
            s.mode = tot_mode
            mode_dict[tot_mode] = [s]
            tot_mode += 1
