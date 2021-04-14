import abc


class Interface:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self, config):
        pass
