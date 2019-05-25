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
    ret = pd.concat([ret, dfs[i]], ignore_index=False)
  ret = ret.reset_index()
  if 'index' in ret.columns:
    ret = ret.drop('index', axis = 1)
  return ret


def search(df, query, by='col=val', dtype=str):
  indices = []
  if type(query) is list:
    for q in query:
      indices += search(df, q, by, dtype)
  elif type(query) is str:
    if by == 'col=val':
      c, v = query.split('=')
      a = np.array(df[c] == dtype(v))
      indices = list(np.where(a == True)[0])
    # elif by == ?
  return indices