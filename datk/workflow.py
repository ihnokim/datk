# from enum import Enum, auto
from enum import Enum
from queue import Queue
import dill


class Status(Enum):
    NEW = 1 # auto()
    READY = 2
    WAITING = 3
    RUNNING = 4
    TERMINATED = 5


class Scheduler:
    def __init__(self):
        self.tasks = dict()
        self.workflows = dict()


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
        
    def __rshift__(self, other):
        if isinstance(other, self.__class__):
            self.children[other.task_id] = other
            other.inheritance[self.task_id] = None
            other.parents[self.task_id] = self
            self.status = other.status = Status.READY
            return None
        if isinstance(other, list):
            for t in other:
                self.__rshift__(t)
            return None
        raise TypeError(type(other))
        return None
    
    def __str__(self):
        return '%s(ID: %s, ACTION: %s, STATUS: %s)' % (self.__class__.__name__, self.task_id, self.action.__name__, self.status)
    
    def __repr__(self):
        return self.__str__()
    
    def run(self):
        self.status = Status.RUNNING
        ret = None
        if self.action is not None:
            for t_id, arg in self.wait_for.items():
                self.args[arg] = self.inheritance[t_id]
            ret = self.action(**self.args)
        for c_id, c in self.children.items():
            c.inheritance[self.task_id] = ret
        self.status = Status.TERMINATED
        return ret
    
    def save(self, filename):
        dill.dump(self, open(filename, 'wb'))
        
    @staticmethod
    def load(filename):
        return dill.load(open(filename, 'rb'))
    
    
class Workflow:
    def __init__(self, workflow_id, task=None, subscribe=None):
        self.workflow_id = workflow_id
        self.set_root(task)
        self.subscribe = subscribe
        self.output = None
    
    def set_root(self, task):
        if not isinstance(task, Task):
            raise TypeError(type(task))
            return None
        self.root = task
    
    def __str__(self):
        ret = ''
        stack = list()
        hierarchy = 0
        roots = self.get_roots()
        first_token = True
        for root in roots:
            stack.append((root, hierarchy))
            while(len(stack)):
                t, h = stack.pop()
                if not first_token:
                    ret += '\n'
                else:
                    first_token = False
                ret += '%s %s' % ('-' * h, t)
                for c_id, c in t.children.items():
                    stack.append((c, h+1))
        return ret
    
    def __repr__(self):
        return self.__str__()

    def get_leaves(self):
        leaves = set()
        q = Queue()
        q.put(self.root)
        while(not q.empty()):
            t = q.get()
            for c_id, c in t.children.items():
                q.put(c)
            if len(t.children) == 0:
                leaves.add(t)
        return leaves
    
    def get_roots(self):
        roots = set()
        leaves = self.get_leaves()
        q = Queue()
        for leave in leaves:
            q.put(leave)
        while(not q.empty()):
            t = q.get()
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
            tmp = task.run()
            if task.task_id == self.subscribe:
                self.output = tmp
            
    def run(self):
        leaves = self.get_leaves()
        for leave in leaves:
            self.recursive_run(leave)
        return self.output

    def save(self, filename):
        dill.dump(self, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        return dill.load(open(filename, 'rb'))
