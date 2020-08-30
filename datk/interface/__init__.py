import abc


class Interface:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def test(self):
        pass
    
    @abc.abstractmethod
    def select(self):
        pass
    
    @abc.abstractmethod
    def insert(self):
        pass
    
    @staticmethod
    @abc.abstractmethod
    def connect(config):
        pass
    
    @abc.abstractmethod
    def disconnect(self):
        pass
