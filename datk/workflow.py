# from enum import Enum, auto
from enum import Enum
from queue import Queue
import dill, bson
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from interface.mongodb import MongoDB
import time
import inspect


class Status(Enum):
    NEW = 1 # auto()
    READY = 2
    WAITING = 3
    RUNNING = 4
    TERMINATED = 5


class Scheduler:
    def __init__(self, config=None):
        if config is not None:
            self.mc = MongoDB(config)
            jobstores = {
                'default': MongoDBJobStore(database=config['db'], collection=config['collection'], client=self.mc.client)
            }
            self.scheduler = BackgroundScheduler(jobstores=jobstores)
        else:
            self.scheduler = BackgroundScheduler()
        
        self.scheduler.start()
        self.add_job = self.scheduler.add_job
        self.remove_job = self.scheduler.remove_job
    
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        self.scheduler.shutdown()
        self.mc.close()


class Task:
    def __init__(self, task_id, action=None, args=None):
        self.task_id = task_id
        self.parents = dict()
        self.children = dict()
        self.inheritance = dict()
    
        self.action = action
        self.args = args
        self.wait_for = dict() # parent task_id -> arg
        self.status = Status.NEW
    
    def get_ready(self):
        self.status = Status.READY
        for p_id in self.inheritance:
            self.inheritance[p_id] = None
        for p_id, arg in self.wait_for.items():
            self.args[arg] = None
        
    def __rshift__(self, other):
        if isinstance(other, self.__class__):
            self.children[other.task_id] = other
            other.inheritance[self.task_id] = None
            other.parents[self.task_id] = self
            self.get_ready()
            return None
        if isinstance(other, list):
            for t in other:
                self.__rshift__(t)
            return None
        raise TypeError(type(other))
        return None
    
    def __str__(self):
        return '%s(ID: %s, ACTION: %s, ARGS: %s, STATUS: %s)' % (self.__class__.__name__, self.task_id, self.action.__name__, inspect.getargspec(self.action), self.status)
    
    def __repr__(self):
        return self.__str__()
    
    def run(self, **kwargs):
        if len(kwargs) == 0:
            args = self.args
        else:
            args = kwargs
        self.status = Status.RUNNING
        ret = None
        if self.action is not None:
            for t_id, arg in self.wait_for.items():
                args[arg] = self.inheritance[t_id]
            ret = self.action(**args)
        for c_id, c in self.children.items():
            c.inheritance[self.task_id] = ret
        self.status = Status.TERMINATED
        return ret
    
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
    
    
class Workflow:
    def __init__(self, workflow_id, task=None, subscribe=None):
        self.workflow_id = workflow_id
        self.set_root(task)
        self.subscribe = subscribe
        self.tasks = dict()
        self.init()
    
    def init(self):
        self.output = None
        self.args = dict()
        self.roots = self.get_roots(init=True)
    
    def set_root(self, task):
        if not isinstance(task, Task):
            raise TypeError(type(task))
            return None
        self.root = task
    
    def __str__(self):
        ret = 'Workflow(ID: %s, root: %s, subscribe: %s)\n' % (self.workflow_id, self.root.task_id, self.subscribe)
        stack = list()
        hierarchy = 0
        first_token = True
        for root in self.roots:
            stack.append((root, hierarchy))
            while(len(stack)):
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
        leaves = self.get_leaves()
        q = Queue()
        for leave in leaves:
            q.put(leave)
        while not q.empty():
            t = q.get()
            if init:
                if t.task_id not in self.args:
                    self.args[t.task_id] = t.args
                t.get_ready()
                self.tasks[t.task_id] = t
            for p_id, p in t.parents.items():
                q.put(p)
            if len(t.parents) == 0:
                roots.add(t)
        return roots
    
    def recursive_run(self, task):
        for p_id, p in task.parents.items():
            if task.status != Status.TERMINATED:
                self.recursive_run(p)
        if task.status == Status.READY:
            if task.task_id in self.args:
                tmp = task.run(**self.args[task.task_id])
            else:
                tmp = task.run()
            if task.task_id == self.subscribe:
                self.output = tmp
            
    def run(self):
        leaves = self.get_leaves()
        for leave in leaves:
            self.recursive_run(leave)
        return self.output

    def save(self, filename):
        self.init()
        dill.dump(self, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        return dill.load(open(filename, 'rb'))
    
    def to_binary(self):
        self.init()
        return bson.Binary(dill.dumps(self))
    
    @staticmethod
    def from_binary(b):
        return dill.loads(b)

    def to_db(self, config, update=False):
        self.init()
        mc = MongoDB(config)
        tmp = dict()
        tmp['workflow_id'] = self.workflow_id
        tmp['binary'] = self.to_binary()
        tmp['structure'] = self.__str__()
        tmp['description'] = dict()
        for t_id, t in self.tasks.items():
            if t.action.__doc__ is not None:
                tmp['description'][t_id] = t.action.__doc__.strip()
            else:
                tmp['description'][t_id] = ''
        mc.insert_one(collection=config['collection'], data=tmp, keys=['workflow_id'], overwrite=update)
    
    @sataticmethod
    def from_db(workflow_id, config):
        mc = MongoDB(config)
        ret = mc.db[config['collection']].find_one({'workflow_id': workflow_id})
        if ret is None:
            return ret
        else:
            return Workflow.from_binary(ret['binary'])
