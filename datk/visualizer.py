import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from scipy.interpolate import griddata
import datk


def draw_coords(subplot, data, labels, annotate=True):
    data = np.array(data)
    labels = np.array(labels)
    for i, label in enumerate(set(labels)):
        indices = np.where(labels == label)[0]
        group = data[indices, :]
        subplot.scatter(group[:, 0], group[:, 1], label=label, s=50, alpha=0.7)
        if annotate:
            subplot.annotate(label, xy=(group[0][0], group[0][1]), xytext=(-5, 5), textcoords='offset points')


def legend(subplot, labels, colors):
    lines = []
    for color in colors:
        lines.append(Line2D([0], [0], linewidth=7.0, linestyle='-', color=color))
    subplot.legend(lines, labels, loc='best', borderpad=0.7)


def draw_nsigma(subplot, x, mu, sigma, n=3, color='r', alpha=0.1, label=''):
    mu = np.array(mu)
    sigma = np.array(sigma)

    subplot.plot(x, mu, color, label=label)
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


def preinterpolate(img, value_points, window_size=None):
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


def extract_coords(value_points, normalization=False, img_shape=(0, 0)):
    if normalization:
        if img_shape is not (0, 0):
            return np.array([[float(coord[0]) / img_shape[0], float(coord[1]) / img_shape[1]] for coord in value_points])
    return np.array([[coord[0], coord[1]] for coord in value_points])


def extract_values(value_points):
    return np.array([value_points[coord] for coord in value_points])


class Wafer(object):

    '''
    wafer_info
    center_shot = 1
    size = (rx, ry) ... how many shots in radius?
    name = str

    chip_info
    size = (w, h)
    num = (y, x) ... in one shot
    shot_pos = (y, x) ... in one shot
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
        r = start_pos[0] + self.chip_info['shot_pos'][0] * self.chip_info['size'][1] # y axis
        c = start_pos[1] + self.chip_info['shot_pos'][1] * self.chip_info['size'][0] # x axis

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
        preinterpolate(self.img, self.values, window_size=window_size)
        points = extract_coords(self.values, normalization=True, img_shape=self.img.shape)
        values = extract_values(self.values)
        grid_x, grid_y = np.mgrid[0: 1: self.img_w * 1j, 0: 1: self.img_h * 1j]
        self.img = griddata(points, values, (grid_x, grid_y), method='cubic')


def get_sample_wafer():
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

    return Wafer(shot_coords, wafer_info, chip_info, values=values)
