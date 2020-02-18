import numpy as np
import pandas as pd
from scipy import interpolate


def join(df1, df2):
    ret = pd.concat([df1, df2], axis=1, join='inner')
    return ret.loc[:, ~ret.columns.duplicated()]


def get_column_indices(df):    
    ret = {}
    if df.index.name is None:
        ret['index'] = 0
    else:
        ret[df.index.name] = 0
    for i, c in enumerate(df.columns):
        ret[c] = i + 1
    return ret


def interpolate1d(x, y, sample):
    if len(x) != len(y):
        print('[ERROR] interpolate1d: len(x) != len(y)')
        return None
    tck = interpolate.splrep(x, y)
    ret = []
    for s in sample:
        ret.append(interpolate.splev(s, tck))
    return ret


def convert_df_to_dict(df, keep_columns=None):
    if keep_columns is None:
        keep_columns = df.columns
    ret = {}
    col_idx = get_column_indices(df)
    for row in df.itertuples():
        val = []
        for col in keep_columns:
            val.append(row[col_idx[col]])
        ret[row[0]] = val
    return ret


def match(df1, df2, left, right, match_map):
    left_col_idx = get_column_indices(df1)
    right_col_idx = get_column_indices(df2)
    ret_cols = {}
    for col in left_col_idx:
        if col in right_col_idx:
            col = col + '_1'
        ret_cols[col] = []
    for col in right_col_idx:
        if col in left_col_idx:
            col = col + '_2'
        ret_cols[col] = []
    for left_row in df1.itertuples():
        if left_row[left_col_idx[left]] not in match_map:
            continue
        for right_row in df2.itertuples():
            if right_row[right_col_idx[right]] == match_map[left_row[left_col_idx[left]]]:
                for left_col in left_col_idx:
                    val = left_row[left_col_idx[left_col]]
                    if left_col not in ret_cols:
                        left_col = left_col + '_1'
                    ret_cols[left_col].append(val)
                for right_col in right_col_idx:
                    val = right_row[right_col_idx[right_col]]
                    if right_col not in ret_cols:
                        right_col = right_col + '_2'
                    ret_cols[right_col].append(val)

    return pd.DataFrame(ret_cols)


def remove_nan(*args):
    n = -1
    keep_idx = None
    for arg in args:
        
        if type(arg) is not list and type(arg) is not np.ndarray:
            print('[ERROR] remove_nan: type of the argument should be list or np.ndarray')
            return None
        if n == -1:
            n = len(arg)
            keep_idx = [~np.isnan(arg[i]) for i in range(n)]
        else:
            if n != len(arg):
                print('[ERROR] remove_nan: the lengths of arguments should match')
                return None
            else:
                for i in range(n):
                    keep_idx[i] = keep_idx[i] and ~np.isnan(arg[i])
    if len(args) == 1:
        return [args[0][i] for i in range(len(args[0])) if keep_idx[i]]
    else:
        return tuple([[arg[i] for i in range(len(arg)) if keep_idx[i]] for arg in args])


def corrcoef(v1, v2):
    return np.corrcoef(remove_nan(v1, v2))[0, 1]


def rsq(v1, v2):
    return corrcoef(v1, v2) ** 2


def mse(v1, v2):
    a1, a2 = np.array(v1), np.array(v2)
    return np.nanmean((a1 - a2) ** 2)


def rmse(v1, v2):
    return np.sqrt(mse(v1, v2))


def filter_labeled_values(labels, values, query, boundary, include=True):
    ret = []
    for i, value in enumerate(get_labeled_values(labels, values, query)):
        valid_label = get_labeled_value(labels[i], labels[i], query)
        if type(valid_label) is not list:
            valid_label = [valid_label]
        for label in valid_label:
            if label is None:
                print('[ERROR] filter_labeled_values: no such label')
                return None
        valid_boundary_idx = get_labeled_index(query, valid_label)
        
        if type(boundary) is not list:
            boundary = [boundary]

        keep = True
        for j, idx in enumerate(valid_boundary_idx):
            lo = boundary[idx][0]
            hi = boundary[idx][1]
            
            cond = (lo <= value[j] < hi)
            if not include:
                cond = not cond
            if not cond:
                keep = False
                break
        
        if keep:
            ret.append(True)
        else:
            ret.append(False)
    return ret


def get_labeled_index(labels, query):
    if type(query) is list:
        ret = []
        for q in query:
            idx = get_labeled_index(labels, q)
            if idx is not None:
                ret.append(idx)
        return ret
    else:
        if query in labels:
            return labels.index(query)
        else:
            return None


def get_labeled_value(labels, values, query):
    if type(query) is list:
        ret = []
        for q in query:
            val = get_labeled_value(labels, values, q)
            if val is not None:
                ret.append(val)
        return ret
    else:
        if query in labels:
            return values[labels.index(query)]
        else:
            return None


def get_labeled_values(labels, values, query):
    ret = []
    labels = list(labels)
    values = list(values)
    for i in range(len(labels)):
        val = get_labeled_value(labels[i], values[i], query)
        if val is None:
            val = []
        elif type(val) is not list:
            val = [val]
        ret.append(val)
    return ret


def grand_moments(groups, ddof=0):
    # http://www.burtonsys.com/climate/composite_standard_deviations.html
    N = 0
    gs = 0
    ess = 0
    tgss = 0
    for ((m, v), n) in groups:
        N += n
        gs += m * n
        ess += v * (n - ddof)
    gm = gs / N
    for ((m, v), n) in groups:
        tgss += (m - gm) ** 2 * n
    gv = (ess + tgss) / (N - ddof)
    return ((gm, gv), N)


def concat(dfs):
    if type(dfs) is not list:
        if type(dfs) is pd.core.frame.DataFrame:
            return dfs
        else:
            print('[ERROR] concat: type of the argument should be list or pandas.core.frame.DataFrame')
            return None
    ret = dfs[0]
    for i in range(1, len(dfs)):
        ret = pd.concat([ret, dfs[i]], ignore_index=False, sort=False)
    ret = ret.reset_index()
    if 'index' in ret.columns:
        ret = ret.drop('index', axis=1)
    return ret


def op_decoder(df, query):
    op = ''
    for t in ['<=', '>=', '=', '<', '>']:
        if t in query:
            op = t
            break
    if op == '':
        return [False]
    c, v = query.split(op)
    c = c.strip()
    v = v.strip()
    
    dtype = type(df[c].iloc[0])
    if op == '=':
        return df[c] == dtype(v)
    elif op == '<':
        return df[c] < dtype(v)
    elif op == '>':
        return df[c] > dtype(v)
    elif op == '<=':
        return df[c] <= dtype(v)
    elif op == '>=':
        return df[c] >= dtype(v)
    else:
        return [False]

    
def search(df, query, query_type='and'):
    ret = df.loc[[False], :]
    if len(df) <= 0:
        pass
    elif type(query) is list:
        if query_type == 'and':
            cond = [True for _ in range(len(df))]
        elif query_type == 'or':
            cond = [False for _ in range(len(df))]
        else:
            print('[ERROR] search: query_type should be "and" or "or"')
            return ret
        for q in query:
            if query_type == 'and':
                cond = op_decoder(df, q) & cond
            elif query_type == 'or':
                cond = op_decoder(df, q) | cond
        ret = df.loc[cond, :]
    elif type(query) is str:
        ret = df.loc[op_decoder(df, query), :]
    else:
        print('[ERROR] search: type of the argument should be str or list')
    return ret


def get_primes(n):
    primes = []
    if n < 2:
        return primes
    for i in range(2, n):
        is_prime = True
        for j in primes:
            if i % j == 0:
                is_prime = False
                break
            elif j > i ** 0.5:
                break
        if is_prime:
            primes.append(i)
    return primes


def get_rect(coords):
    min_x, max_y, max_x, min_y = (float('inf'), float('-inf'), float('-inf'), float('inf'))
    for x, y in coords:
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    return (min_x, max_y), (max_x, min_y)


def dist(p, q):
    if (type(p) is not tuple and type(p) is not list and type(p) is not np.ndarray) or (type(q) is not tuple and type(q) is not list and type(q) is not np.ndarray):
        return np.abs(p - q)
    if len(p) is not len(q):
        print('[ERROR] dist: two vectors need to be in the same dimensions')
        return None
    sum = 0
    for i in range(len(p)):
        sum += (p[i] - q[i]) ** 2
    return np.sqrt(sum)


def get_nearest_neighbor(x, vectors):
    min_dist = float('inf')
    min_idx = -1
    min_vector = None
    for i, v in enumerate(vectors):
        d = dist(x, v)
        if d is None:
            return None
        if d < min_dist:
            min_dist = d
            min_idx = i
            min_vector = v
    return min_idx, min_vector


def get_labeled_coords_converter(source, target):
    ret = {}
    for s in source:
        min_dist = float('inf')
        min_label = -1
        for t in target:
            d = dist(source[s], target[t])
            if d < min_dist:
                min_dist = d
                min_label = t
        ret[s] = min_label
    return ret


class BaseDict(object):
    def __init__(self, dict={}):
        self.data = dict
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return self.__str__()
    
    def __iter__(self):
        return self.data.__iter__()
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def dict(self):
        return self.data
    
    
class ChainingDict(BaseDict):
    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
    
    def __str__(self):
        return 'ChainingDict(' + str(self.data) + ')'


class ArrayDict(BaseDict):
    def keys(self):
        return np.array(list(self.data.keys()))
    
    def values(self):
        return np.array(list(self.data.values()))
    
    def __str__(self):
        return 'ArrayDict(' + str(self.data) + ')'
