import abc
import numpy as np


class Interface:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self, config):
        pass
    
    @abc.abstractmethod
    def test(self):
        pass
    
    @abc.abstractmethod
    def select(self):
        pass
    
    @abc.abstractmethod
    def insert(self):
        pass
    
    @abc.abstractmethod
    def remove(self):
        pass
    
    @staticmethod
    @abc.abstractmethod
    def connect(config):
        pass
    
    @abc.abstractmethod
    def disconnect(self):
        pass


def isnan(value):
    ret = False
    try:
        ret = np.isnan(value)
    except Exception:
        pass
    finally:
        return ret
