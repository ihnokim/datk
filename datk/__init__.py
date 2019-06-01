import numpy as np
import pandas as pd


def join(df1, df2):
    ret = pd.concat([df1, df2], axis=1, join='inner')
    return ret.loc[:, ~ret.columns.duplicated()]


def concat(dfs):
    if type(dfs) is not list:
        if type(dfs) is pd.core.frame.DataFrame:
            return dfs
        else:
            print('[ERROR] concat: type of the argument should be a list or pandas.core.frame.DataFrame')
            return None
    ret = dfs[0]
    for i in range(1, len(dfs)):
        ret = pd.concat([ret, dfs[i]], ignore_index=False, sort=False)
    ret = ret.reset_index()
    if 'index' in ret.columns:
        ret = ret.drop('index', axis=1)
    return ret


def search(df, query, by='value'):
    ret = df.loc[[False], :]
    if len(df) <= 0:
        pass

    elif by == 'index':
        if type(query) is not list:
            query = [query]
        ret = df.loc[query, :]

    elif type(query) is list:
        if by == 'value':
            cond = [False for _ in range(len(df))]
            for q in query:
                # indices += search(df, q, by, dtype)
                c, v = q.split('=')
                c = c.strip()
                v = v.strip()
                dtype = type(df[c].iloc[0])
                cond = (df[c] == dtype(v)) | cond
            ret = df.loc[cond, :]

    elif type(query) is str:
        if by == 'value':
            c, v = query.split('=')
            c = c.strip()
            v = v.strip()
            dtype = type(df[c].iloc[0])
            ret = df.loc[df[c] == dtype(v), :]

    else:
        pass

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
    if (type(p) is not tuple and type(p) is not list) or (type(q) is not tuple and type(q) is not list):
        return np.abs(p - q)
    if len(p) is not len(q):
        print('[ERROR] dist: two vectors need to be in the same dimensions')
        return 0
    sum = 0
    for i in range(len(p)):
        sum += (p[i] - q[i]) ** 2
    return np.sqrt(sum)


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
