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


def search(df, query, by='value'):
  ret = df.loc[[False], :]
  if len(df) <= 0:
    pass

  elif type(query) is list:
    cond = [False for _ in range(len(df))]
    for q in query:
      # indices += search(df, q, by, dtype)
      if by == 'value':
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
    # elif by == ?
  elif type(query) is int:
    pass
  else:
    pass

  return ret