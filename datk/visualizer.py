import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from scipy.interpolate import griddata
import datk
import math


def fill_wafer(coords, values, n=0, radius=150):
    c, v = coords, values
    for n_tube in range(3, 3 + n):
        c, v = preinterpolate(c, v, n_tube, radius, mode=0)
        c, v = preinterpolate(c, v, n_tube, radius, mode=1)
    return c, v


def preinterpolate(coords, values, n_tubes, radius=150, mode=0):
    sorted_data = sorted([(coords[i], values[i], datk.dist((0.0, 0.0), coords[i])) for i in range(len(coords))], key=lambda x: x[2])
    ds = [i for i in range(0, radius + n_tubes, int(radius / n_tubes))]
    n_blocks = [1] + [2 ** (i + 2) for i in range(n_tubes - 1)]

    ret_coords = []
    ret_values = []

    last_idx = -1
    for t in range(n_tubes):
        min_d, max_d = ds[t], ds[t + 1]
        center_d = (min_d + max_d) / 2
        theta_offset = 360 / n_blocks[t]
        for b in range(n_blocks[t]):
            if mode == 0:
                min_theta, max_theta = (b - 1 / 2) * theta_offset, (b + 1 / 2) * theta_offset
            elif mode == 1:
                min_theta, max_theta = b * theta_offset, (b + 1) * theta_offset
            center_theta = min_theta + theta_offset / 2
            if b == 0:
                if mode == 0:
                    center_theta = 0.0
                elif mode == 1:
                    center_theta = min_theta + theta_offset / 2
            if t == 0:
                center_d = 0.0
            center_x, center_y = center_d * math.cos(math.radians(center_theta)), center_d * math.sin(math.radians(center_theta))
            tmp_idx = last_idx
            total = 0
            cnt = 0
            for i in range(last_idx + 1, len(sorted_data)):
                include = False
                if min_d <= sorted_data[i][2] < max_d:
                    theta = math.degrees(math.atan2(sorted_data[i][0][1], sorted_data[i][0][0]))
                    if theta < 0:
                        theta = 360 + theta

                    if mode == 0:
                        if b == 0:
                            if min_theta + 360 <= theta < 360 or 0 <= theta < max_theta:
                                include = True
                    elif mode == 1:
                        if b == 0:
                            if min_theta <= theta < max_theta:
                                include = True
                    if b != 0:
                        if min_theta <= theta < max_theta:
                            include = True
                    if include:
                        tmp_idx = i
                        if not np.isnan(sorted_data[i][1]):
                            total += sorted_data[i][1]
                            cnt += 1
                elif max_d <= sorted_data[i][2]:
                    break
            if cnt != 0:
                ret_coords.append((center_x, center_y))
                ret_values.append(total / cnt)
        last_idx = tmp_idx
    return np.concatenate([np.array(coords), np.array(ret_coords)]), np.concatenate([np.array(values), np.array(ret_values)])


def draw_pdf(subplot, data, bins=10, ddof=0, color=None, annotate=False, fontsize=10):
    y, x, _ = subplot.hist(data, bins, density=True, color=color)
    mu, sigma = np.mean(data), np.std(data, ddof=ddof)
    subplot.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='r')
    if annotate:
        for i in range(len(y)):
            subplot.text(x=x[i], y=y[i],
                     s='{:.2f}%'.format(y[i] * 100),
                     fontsize=fontsize,
                     color='black')


def draw_labeled_value(subplot, labels, values, query):
    value = datk.get_labeled_value(labels, values, query)
    subplot.plot(query, value, '-o')

    
def draw_labeled_values(subplot, labels, values, query):
    labels = list(labels)
    values = list(values)
    idx = datk.get_labeled_index(labels[0], query)
    if idx is None:
        idx = []
    elif type(idx) is not list:
        idx = [idx]
    for i in idx:
        subplot.plot([i for i in range(1, len(labels) + 1)], [values[v][i] for v in range(len(values))], 'o', label=labels[0][i])


def draw_coords(subplot, coords, labels, annotate=True):
    coords = np.array(coords)
    labels = np.array(labels)
    for i, label in enumerate(set(labels)):
        indices = np.where(labels == label)[0]
        group = coords[indices, :]
        subplot.scatter(group[:, 0], group[:, 1], label=label, s=50, alpha=0.7)
        if annotate:
            subplot.annotate(label, xy=(group[0][0], group[0][1]), xytext=(-5, 5), textcoords='offset points')


def fill_edge(coords, values, edge):
    ret_coords = []
    ret_values = []
    for x in edge[0]:
        for y in [edge[1][0], edge[1][-1]]:
            i, _ = datk.get_nearest_neighbor((x, y), coords)
            ret_coords.append((x, y))
            ret_values.append(values[i])
    for x in [edge[0][0], edge[0][-1]]:
        for y in edge[1]:
            i, _ = datk.get_nearest_neighbor((x, y), coords)
            ret_coords.append((x, y))
            ret_values.append(values[i])
    return np.concatenate([np.array(coords), np.array(ret_coords)]), np.concatenate([np.array(values), np.array(ret_values)])


def draw_wafer(subplot, coords, values, labels=None, annotate=True, fontsize=10, cmap='jet_r', clim=None, size=(300, 300), interpolate=0):
    if labels is None:
        labels = values

    rx, ry = size[0] / 2, size[1] / 2
    edge = ([i for i in range(-int(rx), int(rx) + 1, int(size[0] / 6))], [i for i in range(-int(ry), int(ry) + 1, int(size[1] / 6))])
    coords, values = fill_wafer(coords, values, interpolate, int(min(rx, ry)))
    coords, values = fill_edge(coords, values, edge)

    grid_x, grid_y = np.mgrid[-int(rx): int(rx) + 1: 1, -int(ry): int(ry) + 1: 1]
    img = griddata(coords, values, (grid_x, grid_y), method='cubic')
    ell = Ellipse((0, 0), size[0], size[1], fill=False)

    subplot.add_patch(ell)
    plot = subplot.imshow(img.T, cmap=cmap, interpolation='none', extent=(-int(rx), int(rx) + 1, -int(ry), int(ry) +  1), origin='lower', clip_path=ell, clip_on=True)

    if annotate:
        for i in range(len(labels)):
            subplot.scatter(coords[i][0], coords[i][1], c='black', s=3)
            if type(labels[i]) is float:
                v = str(round(labels[i], 2))
            else:
                v = str(labels[i])
            subplot.annotate(v, xy=coords[i], xytext=(0, 3), textcoords='offset points', ha='center', fontsize=fontsize)
    if clim is not None:
        plot.set_clim(clim)
    return plot


def legend(subplot, labels, colors):
    lines = []
    for color in colors:
        lines.append(Line2D([0], [0], linewidth=7.0, linestyle='-', color=color))
    subplot.legend(lines, labels, loc='best', borderpad=0.7)


def draw_nsigma(subplot, x, mu, sigma, n=3, color='r', alpha=0.1, linestyle='solid', label=''):
    mu = np.array(mu)
    sigma = np.array(sigma)

    subplot.plot(x, mu, color, linestyle=linestyle, label=label)
    for i in range(1, n + 1):
        subplot.fill_between(x, mu - sigma * i, mu + sigma * i, color=color, alpha=alpha)


def macchiato(img, value_points, decay=1.0):
    tmp = []
    for coord in value_points:
        val = value_points[coord]
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = (int(coord[0]) + i, int(coord[1]) + j)
            if (0 <= x < img.shape[0]) and (0 <= y < img.shape[1]) and np.isnan(img[x, y]):
                img[x, y] = val * decay
                tmp.append(((x, y), val * decay))

    for coord, val in tmp:
        value_points[coord] = val


def proliferate(img, value_points, window_size=5):
    if window_size % 2 is 0:
        window_size += 1
    window_size = int(window_size)
    offset = int(window_size / 2)
    step = offset + 1
    iter_points = np.array([[i, j] for i in range(0, img.shape[0], step) for j in range(0, img.shape[1], step)])

    for x, y in iter_points:
        if np.isnan(img[x, y]):
            val = np.nanmean(img[x - offset: x + step + 1, y - offset: y + step + 1])

            if not np.isnan(val):
                img[x, y] = val
                value_points[x, y] = val


def deprecated_preinterpolate(img, value_points, window_size=None):
    min_dist = datk.dist(img.shape, (0, 0))
    max_dist = 0.0
    for coord1 in value_points:
        for coord2 in value_points:
            d = datk.dist(coord1, coord2)
            if 2.0 < d < min_dist:
                min_dist = d
            if d > max_dist:
                max_dist = d

    if window_size is None:
        upper = (min_dist + max_dist) / 2
    else:
        upper = window_size

    l = sorted(datk.get_primes(int(upper)), reverse=True)
    for i in l:
        if i > 2:
            proliferate(img, value_points, window_size=i)


def extract_coords(value_points, normalization=False, img_shape=None):
    if normalization:
        if img_shape is not None:
            return np.array([[float(coord[0]) / img_shape[0], float(coord[1]) / img_shape[1]] for coord in value_points])
    return np.array([[coord[0], coord[1]] for coord in value_points])


def extract_values(value_points):
    return np.array([value_points[coord] for coord in value_points])


class DataPainter():
    def __init__(self, fmt=None, ax=None):
        self.pressed = False
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            self.xdata = []
            self.ydata = []
            self.line = self.ax.plot(self.xdata, self.ydata)[0]
        else:
            self.ax = ax
            if len(self.ax.get_lines()) > 0:
                self.line = self.ax.get_lines()[0]
                data = self.line.get_data()
                self.xdata, self.ydata = list(data[0]), list(data[1])
        self.fignum = self.ax.figure.number
                
    def show(self):
        if plt.fignum_exists(self.fignum) is False:
            fig = plt.figure()
            fig.suptitle('DataPainter')
            self.ax = fig.add_subplot(111)
            self.line = self.ax.plot(self.xdata, self.ydata)[0]
            self.fignum = self.ax.figure.number
        self.connect()
        plt.show(block=False)
    
    def connect(self):
        self.cid_press = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        return self

    def clear(self):
        self.xdata = []
        self.ydata = []
        self.line.set_data(self.xdata, self.ydata)
        self.ax.figure.canvas.draw()

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)

    def close(self):
        self.disconnect()
        plt.close(self.ax.figure)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.pressed = True
    
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if self.pressed:
            self.xdata.append(event.xdata)
            self.ydata.append(event.ydata)
            self.line.set_data(self.xdata, self.ydata)
            self.ax.figure.canvas.draw()
    
    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.pressed = False


class MiniWafer(object):

    '''
    shot_coords\n
    key: int\n
    value: (x, y)\n
    \n
    wafer_info\n
    center_shot = 1\n
    size = (rx, ry) ... how many shots in radius?\n
    name = str\n
    \n
    chip_info
    size = (w, h)\n
    num = (y, x) ... in one shot\n
    shot_pos = (y, x) ... in one shot\n
    '''

    def __init__(self, shot_coords, wafer_info, chip_info, padding=1, values={}):
        self.shot_coords = shot_coords
        self.wafer_info = wafer_info
        self.chip_info = chip_info
        self.shot_info = {}
        self.padding = padding
        self.w = 0
        self.h = 0
        self.img_w = 0
        self.img_h = 0
        self.pos = {}
        self.img = {}
        self.values = {}
        self.init_frame(shot_coords)
        self.add_values(values)
        
    def init_frame(self, shot_coords):
        self.shot_coords = shot_coords
        (sx, sy), (ex, ey) = datk.get_rect(self.shot_coords.values())
        self.shot_info['num'] = (sy - ey + 1, ex - sx + 1)
        self.w = int(self.shot_info['num'][1] * self.chip_info['size'][0] * self.chip_info['num'][1])
        self.h = int(self.shot_info['num'][0] * self.chip_info['size'][1] * self.chip_info['num'][0])
        self.shot_info['size'] = (int(self.w / self.shot_info['num'][1]), int(self.h / self.shot_info['num'][0]))
        self.img_w = self.w + 2 * self.padding * self.shot_info['size'][0]
        self.img_h = self.h + 2 * self.padding * self.shot_info['size'][1]
        self.clear()
        shots = self.shot_coords.keys()
        self.pos = {}
        for shot in shots:
            x, y = self.shot_coords[shot]
            dx, dy = (x - sx, sy - y)
            self.pos[shot] = ((dy + self.padding) * self.shot_info['size'][1], (dx + self.padding) * self.shot_info['size'][0])

    def clear(self):
        self.img = np.full((self.img_h, self.img_w), np.nan)
        self.values = {}

    def add_value(self, shot, value):
        start_pos = self.pos[shot]
        r = start_pos[0] + self.chip_info['shot_pos'][0] * self.chip_info['size'][1]  # y axis
        c = start_pos[1] + self.chip_info['shot_pos'][1] * self.chip_info['size'][0]  # x axis

        for r_i in range(r, r + self.chip_info['size'][1]):
            for c_i in range(c, c + self.chip_info['size'][0]):
                self.values[r_i, c_i] = value
                self.img[r_i, c_i] = value

    def add_values(self, values):
        for shot in values:
            self.add_value(shot, values[shot])

    def draw(self, subplot, clim=(0.0, 1.0)):
        ell = Ellipse((self.pos[self.wafer_info['center_shot']][1], self.img_h - self.pos[self.wafer_info['center_shot']][0] - self.shot_info['size'][1]),
                      2 * self.wafer_info['size'][0] * self.chip_info['num'][1] * self.chip_info['size'][0],
                      2 * self.wafer_info['size'][1] * self.chip_info['num'][0] * self.chip_info['size'][1], fill=False)

        for shot in self.pos:
            subplot.add_patch(Rectangle((self.pos[shot][1], self.img_h - self.pos[shot][0]), self.shot_info['size'][0], -self.shot_info['size'][1], fill=False))

        subplot.add_patch(ell)
        plot = subplot.imshow(self.img, cmap='jet_r', interpolation='none', extent=(0, self.img_w, 0, self.img_h), origin='upper', clip_path=ell, clip_on=True)
        plot.set_clim(clim[0], clim[1])
        subplot.set_xticks([])
        subplot.set_yticks([])
        return plot

    def interpolate(self, window_size=None):
        macchiato(self.img, self.values)
        deprecated_preinterpolate(self.img, self.values, window_size=window_size)
        points = extract_coords(self.values, normalization=True, img_shape=self.img.shape)
        values = extract_values(self.values)
        grid_x, grid_y = np.mgrid[0: 1: self.img_w * 1j, 0: 1: self.img_h * 1j]
        self.img = griddata(points, values, (grid_x, grid_y), method='cubic')

    @staticmethod
    def example():
        shot_coords = {}
        for shot in range(1, 6):
            shot_coords[shot] = (0, shot - 1)
        for shot in range(6, 14):
            shot_coords[shot] = (1, 9 - shot)
        for shot in range(14, 22):
            shot_coords[shot] = (2, shot - 18)
        for shot in range(22, 28):
            shot_coords[shot] = (3, 24 - shot)
        for shot in range(28, 30):
            shot_coords[shot] = (4, shot - 29)
        for shot in range(30, 35):
            shot_coords[shot] = (0, 29 - shot)
        for shot in range(35, 45):
            shot_coords[shot] = (-1, shot - 40)
        for shot in range(45, 53):
            shot_coords[shot] = (-2, 48 - shot)
        for shot in range(53, 61):
            shot_coords[shot] = (-3, shot - 57)
        for shot in range(61, 67):
            shot_coords[shot] = (-4, 63 - shot)
        for shot in range(67, 69):
            shot_coords[shot] = (-5, shot - 68)

        wafer_info = {}
        wafer_info['center_shot'] = 1
        wafer_info['size'] = (5.5, 5.5)
        wafer_info['name'] = 'Sample Wafer'

        chip_info = {}
        chip_info['size'] = (2, 1)
        chip_info['num'] = (6, 3)
        chip_info['shot_pos'] = (3, 1)

        values = {}
        for shot in shot_coords:
            values[shot] = float(np.random.rand(1))

        return MiniWafer(shot_coords, wafer_info, chip_info, values=values)
