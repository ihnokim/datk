# from enum import Enum, auto
from enum import Enum
from queue import Queue
import dill, bson
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from datk.interface.mongodb import MongoDB
import inspect
from collections.abc import Iterable
from copy import copy


class Status(Enum):
    NEW = 1 # auto()
    READY = 2
    WAITING = 3
    RUNNING = 4
    TERMINATED = 5


class Delivery:
    def __init__(self, name=None, item=None, sender=None):
        self.name = name
        self.item = item
        self.sender = sender

    def cancel(self):
        self.item = None
        self.sender = None
    
    def __lshift__(self, sender):
        if isinstance(sender, str):
            self.sender = sender
            self.item = None
        elif isinstance(sender, Task):
            self.sender = sender.id
            self.item = None
        else:
            raise TypeError(type(sender))

    def send(self, item):
        self.item = item

    def __str__(self):
        return '%s(name: %s, sender: %s)' % (self.__class__.__name__, self.name, self.sender)

    def __repr__(self):
        return self.__str__()


class Scheduler:
    def __init__(self, config=None):
        if config is not None:
            try:
                self.config = config
                self.mc = MongoDB(config)
                self.jobs_collection = self.config['collection'] + '_jobs'
                self.scheduler_collection = self.config['collection'] + '_scheduler'
                jobstores = {
                    'default': MongoDBJobStore(database=self.config['db'],
                                               collection=self.jobs_collection,
                                               client=self.mc.client)
                }
                self.scheduler = BackgroundScheduler(jobstores=jobstores)
            except Exception as e:
                print(e)
                self.scheduler = BackgroundScheduler()
        else:
            self.scheduler = BackgroundScheduler()
        self.scheduler.start()
    
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        self.scheduler.shutdown()
        self.mc.close()
        
    def add_job(self, func, id, trigger, **kwargs):
        self.mc.client[self.config['db']][self.scheduler_collection].insert_one({'_id': id, 'trigger': trigger})
        return self.scheduler.add_job(func=func, id=id, trigger=trigger, **kwargs)
    
    def remove_job(self, id, **kwargs):
        self.mc.client[self.config['db']][self.scheduler_collection].remove({'_id': id})
        return self.scheduler.remove_job(job_id=id, **kwargs)


class DeliveryService(Iterable):
    def __init__(self, deliveries=None):
        if deliveries is None:
            self.deliveries = dict()
        elif isinstance(deliveries, dict):
            self.deliveries = deliveries
        else:
            raise TypeError(type(deliveries))
    
    def __iter__(self):
        return iter(self.deliveries.values())
    
    def __setitem__(self, key, value):
        self.deliveries[key] = value
    
    def __getitem__(self, key):
        return self.deliveries[key]
    
    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, list(self.deliveries.values()))
    
    def __repr__(self):
        return self.__str__()
    
    def keys(self):
        return self.deliveries.keys()
    
    def values(self):
        return self.deliveries.values()
    
    def items(self):
        return self.deliveries.items()
    
    def cancel_deliveries(self, sender=None):
        for delivery in self:
            if sender is None or delivery.sender == sender:
                delivery.cancel()
    
    def send_deliveries(self, sender=None, item=None):
        for delivery in self:
            if sender is None or delivery.sender == sender:
                delivery.send(item)
    
    def return_deliveries(self, sender=None):
        self.send_deliveries(sender=sender, item=None)
    
    def inquire(self, name):
        return self.__getitem__(name)
    
    def request(self, name, sender=None):
        self.__setitem__(name, Delivery(name=name, sender=sender))
    
    def get_senders(self):
        return {k: v.sender for k, v in self.items() if v.sender is not None}


class Task:
    def __init__(self, action, id=None):
        if id is None:
            self.id = action.__name__
        else:
            self.id = id
        
        self.status = None
        
        self.action = None
        self.source = None
        self.description = None
        
        self.parents = None
        self.children = None
        self.delivery_service = None
        
        self.set_action(action)
        self.init()
    
    def set_action(self, action):
        self.action = action
        self.source = inspect.getsource(self.action)
        self.description = self.action.__doc__.strip() if self.action.__doc__ is not None else str()
    
    def init(self):
        self.parents = dict()
        self.children = dict()
        self.delivery_service = DeliveryService()
        for arg in inspect.getfullargspec(self.action).args:
            self.delivery_service.request(arg)
        self.update_status(Status.NEW)
    
    def get_ready(self):
        self.update_status(Status.READY)
        self.delivery_service.return_deliveries()
    
    @staticmethod
    def return_delivery(sender, delivery):
        if isinstance(delivery, Iterable):
            for d in delivery:
                if d.sender == sender:
                    d.send(None)
        elif isinstance(delivery, Delivery):
            if delivery.sender == sender:
                delivery.send(None)
        else:
            raise TypeError(type(delivery))

    def receive(self, arg):
        return self.delivery_service.inquire(arg)

    def __rshift__(self, other):
        if isinstance(other, self.__class__):
            self.children[other.id] = other
            other.parents[self.id] = self

            other.delivery_service.return_deliveries(sender=self.id)
            self.get_ready()
        elif isinstance(other, Iterable):
            for t in other:
                self.__rshift__(t)
        else:
            raise TypeError(type(other))

    def __str__(self):
        return '%s(id: %s, action: %s, args: %s, receive: %s, status: %s)' %\
               (self.__class__.__name__, self.id, self.action.__name__,
                inspect.getfullargspec(self.action).args,
                self.delivery_service.get_senders(),
                self.status)
    
    def __repr__(self):
        return self.__str__()

    def update_status(self, status):
        self.status = status
    
    def __call__(self, *args, **kwargs):
        self.update_status(Status.RUNNING)

        for delivery in self.delivery_service:
            if delivery.sender is not None and delivery.name not in kwargs:
                kwargs[delivery.name] = delivery.item

        ret = self.action(*args, **kwargs)

        for c in self.children.values():
            c.delivery_service.send_deliveries(sender=self.id, item=ret)
        
        self.update_status(Status.TERMINATED)
        return ret
    
    def clone(self):
        tmp = copy(self)
        tmp.init()
        return tmp
    
    def save(self, filename):
        dill.dump(self, open(filename, 'wb'))
        
    @staticmethod
    def load(filename):
        return dill.load(open(filename, 'rb'))
    
    def to_binary(self):
        return bson.Binary(dill.dumps(self))
    
    @staticmethod
    def from_binary(b):
        return dill.loads(b)
    
    def to_db(self, config):
        tmp = self.clone()
        mc = MongoDB(config)
        document = dict()
        document['_id'] = tmp.id
        document['binary'] = tmp.to_binary()
        document['summary'] = tmp.__str__()
        document['source'] = tmp.source
        document['description'] = tmp.description
        mc.client[config['db']][config['collection']].insert_one(document)
        mc.close()
    
    @staticmethod
    def from_db(id, config):
        mc = MongoDB(config)
        ret = mc.client[config['db']][config['collection']].find_one({'_id': id})
        if ret is None:
            return ret
        else:
            return Task.from_binary(ret['binary'])
    
    
class Workflow:
    def __init__(self, id, task, subscribe=None):
        self.id = id
        self.root = None
        self.roots = None
        self.output = None
        self.set_root(task)
        if isinstance(subscribe, Task):
            self.subscribe = subscribe.id
        elif isinstance(subscribe, str):
            self.subscribe = subscribe
        else:
            self.subscribe = None
        self.tasks = dict()
        self.init()
    
    def init(self):
        self.output = None
        self.roots = self.get_roots(init=True)
    
    def set_root(self, task):
        if not isinstance(task, Task):
            raise TypeError(type(task))
        self.root = task
    
    def __str__(self):
        ret = 'Workflow(id: %s, root: %s, subscribe: %s)\n' % (self.id, self.root.id, self.subscribe)
        stack = list()
        hierarchy = 0
        first_token = True
        for root in self.roots:
            stack.append((root, hierarchy))
            while len(stack):
                t, h = stack.pop()
                if not first_token:
                    ret += '\n'
                else:
                    first_token = False
                ret += '|%s %s' % ('-' * h, t)
                for c_id, c in t.children.items():
                    stack.append((c, h+1))
        return ret
    
    def __repr__(self):
        return self.__str__()

    def get_leaves(self):
        leaves = set()
        q = Queue()
        q.put(self.root)
        while not q.empty():
            t = q.get()
            for c_id, c in t.children.items():
                q.put(c)
            if len(t.children) == 0:
                leaves.add(t)
        return leaves
    
    def get_roots(self, init=False):
        roots = set()
        q = Queue()
        for leave in self.get_leaves():
            q.put(leave)
        while not q.empty():
            t = q.get()
            if init:
                t.get_ready()
                self.tasks[t.id] = t
            for p in t.parents.values():
                q.put(p)
            if len(t.parents) == 0:
                roots.add(t)
        return roots
    
    def recursive_run(self, task, args):
        for p_id, p in task.parents.items():
            if task.status != Status.TERMINATED:
                self.recursive_run(p, args)
        if task.status == Status.READY:
            tmp = task(**args[task.id])
            if task.id == self.subscribe:
                self.output = tmp
            
    def __call__(self, args):
        for leave in self.get_leaves():
            self.recursive_run(leave, args)
        return self.output

    def to_file(self, filename):
        self.init()
        dill.dump(self, open(filename, 'wb'))
    
    @staticmethod
    def from_file(filename):
        return dill.load(open(filename, 'rb'))
    
    def to_binary(self):
        self.init()
        return bson.Binary(dill.dumps(self))
    
    @staticmethod
    def from_binary(b):
        return dill.loads(b)

    def to_db(self, config):
        self.init()
        mc = MongoDB(config)
        tmp = dict()
        tmp['_id'] = self.id
        tmp['binary'] = self.to_binary()
        tmp['structure'] = self.__str__()
        mc.client[config['db']][config['collection']].insert_one(tmp)
        mc.close()
    
    @staticmethod
    def from_db(id, config):
        mc = MongoDB(config)
        ret = mc.client[config['db']][config['collection']].find_one({'_id': id})
        if ret is None:
            return ret
        else:
            return Workflow.from_binary(ret['binary'])
