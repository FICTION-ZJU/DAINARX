from src.CurveSlice import Slice


def clustering(data: list[Slice]):
    tot_mode = 1
    last_mode = None
    for i in range(len(data)):
        if data[i].isFront:
            last_mode = None
        for j in range(i):
            if (data[j].mode != last_mode) and (data[i] & data[j]):
                data[i].setMode(data[j].mode)
            if data[i].mode is not None:
                break
        if data[i].mode is None:
            data[i].mode = tot_mode
            tot_mode += 1
        last_mode = data[i].mode
