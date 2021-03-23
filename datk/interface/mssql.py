import pymssql
from datk.interface import Interface


class MSSQL(Interface):
    def __init__(self, config):
        self.config = config
    
    def query(self, q):
        ret = list()
        try:
            conn = pymssql.connect(host=self.config['ip'], port=int(self.config['port']),
                                   user=self.config['id'], password=self.config['pw'],
                                   database=self.config['db'],
                                   charset=self.config['charset'],
                                   autocommit=False, as_dict=True)
            
            cursor = conn.cursor()
            if type(q) == str:
                cursor.execute(q)
                ret = cursor.fetchall()
            elif type(q) == list:
                for qn in q:
                    cursor.execute(qn)
                    ret += cursor.fetchall()
            
            cursor.close()
            conn.commit()
            conn.close()
            conn = None
        except Exception as e:
            print(e)
            conn.rollback()
        finally:
            if conn is not None:
                conn.close()
        return ret
    
    def insert(self, table, update=True, **kwargs):
        q = self.make_insert_query(table=table, update=update, **kwargs)
        self.query(q)
    
    def make_insert_query(self, table, update=True, **kwargs):
        columns = kwargs.keys()
        values = [str(kwargs[c]) for c in columns]
        if update:
            q = "replace"
        else:
            q = "insert"
        q += " into %s.%s (%s) " % (self.config['db'], table, ', '.join(columns))
        q += "values ('%s')" % "', '".join(values)
        return q
    
    def insert_many(self, table, rows, update=True):
        qs = list()
        for row in rows:
            qs.append(self.make_insert_query(table=table, update=update, **row))
        self.query(qs)
    
    def select(self, table, **kwargs):
        q = "select * from %s.%s " % (self.config['db'], table)
        if len(kwargs) > 0:
            q += " where "
            q += self.query_encoder(**kwargs)
        return self.query(q)
    
    def exec(self, procedure, **kwargs):
        q = self.make_exec_query(procedure, **kwargs)
        return self.query(q)
    
    def exec_many(self, procedure, params=[]):
        qs = list()
        for param in params:
            qs.append(self.make_exec_query(procedure, **param))
        return self.query(qs)
    
    def make_exec_query(self, procedure, **kwargs):
        q = 'exec %s' % procedure
        tokens = list()
        for k, v in kwargs.items():
            tokens.append("@%s = '%s'" % (k, v))
        if len(tokens) > 0:
            q += ' ' + ', '.join(tokens)
        return q
    
    def remove(self, table, **kwargs):
        q = "delete from %s.%s" % (self.config['db'], table)
        if len(kwargs) > 0:
            q += " where "
            q += self.query_encoder(**kwargs)
        return self.query(q)
    
    def close(self):
        pass
    
    @staticmethod
    def query_encoder(custom_op={}, **kwargs):
        tokens = list()
        for k, v in kwargs.items():
            if type(v) is list:
                tokens.append(k + " in ('{}')".format("', '".join([str(x) for x in v])))
            else:
                tokens.append(k + " = '{}'".format(v))
        
        for k, v in custom_op.items():
            for op, val in v.items():
                tokens.append("%s %s '%s'" % (k, op, val))
        return ' and '.join(tokens)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
