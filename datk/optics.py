import math
import numpy as np


def smart_arccos(wl, value, threshold = 0.01):
    if len(wl) != len(value):
        print('[ERROR] smart_arccos: len(wl) != len(value)')
        return None
    peaks = []
    ret = []
    arccos = np.arccos(value)
    for i in range(len(wl)):
        if i == 0 or i == len(wl) - 1:
            continue
        front = (arccos[i] - arccos[i - 1]) / (wl[i] - wl[i - 1])
        back = (arccos[i + 1] - arccos[i]) / (wl[i + 1] - wl[i])
        if front * back < 0 and (arccos[i] < threshold or arccos[i] > math.pi - threshold):
            peaks.append(i)
    peaks.append(len(wl))
    for k in range(2):
        tmp = []
        for i, idx in enumerate(peaks):
            if i == 0:
                start_idx = 0
            else:
                start_idx = peaks[i - 1]
                
            for j in range(start_idx, idx):
                if i % 2 == k:
                    tmp.append(arccos[j])
                else:
                    tmp.append(-arccos[j])
        ret.append(np.array(tmp))
    return ret


def cauchy(wl, n, k):
    return np.array([complex(n[0] + n[1] / w ** 2 + n[2] / w ** 4, k[0] + k[1] / w ** 2 + k[2] / w ** 4) for w in wl])


def jones_rotate(degree):
    radian = degree * math.pi / 180.0
    ret = np.eye(2, dtype=complex)
    ret[0, 0] = np.cos(radian)
    ret[0, 1] = np.sin(radian)
    ret[1, 0] = np.sin(radian) * -1.0
    ret[1, 1] = np.cos(radian)
    return ret
