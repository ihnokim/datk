import numpy as np
import pandas as pd


def join(df1, df2):
    ret = pd.concat([df1, df2], axis = 1, join = 'inner')
    return ret.loc[:, ~ret.columns.duplicated()]


def concat(dfs):
    if type(dfs) is not list:
        if type(dfs) is pd.core.frame.DataFrame:
            return dfs
        else:
            print('error: !!')
            return None
    ret = dfs[0]
    for i in range(1, len(dfs)):
        ret = pd.concat([ret, dfs[i]], ignore_index=False, sort=False)
    ret = ret.reset_index()
    if 'index' in ret.columns:
        ret = ret.drop('index', axis = 1)
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
