import pymysql
from datk.interface import Interface
from sshtunnel import SSHTunnelForwarder
import numpy as np


class MySQL(Interface):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    def query(self, q):
        ssh = None
        conn = None
        ret = None
        if self.config['sshtunnel'] == 'true':

            ssh = SSHTunnelForwarder((self.config['ssh_ip'], self.config['ssh_port']),
                                     ssh_password=self.config['ssh_pw'],
                                     ssh_username=self.config['ssh_id'],
                                     remote_bind_address=(self.config['ip'], int(self.config['port'])))
            ssh.start()
            host = '127.0.0.1'
            port = ssh.local_bind_port
        else:
            host = self.config['ip']
            port = int(self.config['port'])

        conn = pymysql.connect(host=host, port=port,
                               user=self.config['id'], passwd=self.config['pw'],
                               db=self.config['db'],
                               charset=self.config['charset'],
                               cursorclass=pymysql.cursors.DictCursor)
        try:
            cursor = conn.cursor()
            if type(q) == str:
                cursor.execute(q)
            elif type(q) == list:
                for qn in q:
                    cursor.execute(qn)
            ret = cursor.fetchall()
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
            if ssh is not None:
                ssh.stop()
        return ret
    
    def insert(self, table, update=True, **kwargs):
        q = self.make_insert_query(table=table, update=update, **kwargs)
        self.query(q)
    
    def make_insert_query(self, table, update=True, **kwargs):
        columns = kwargs.keys()
        tmp = [kwargs[c] for c in columns]
        values = list()
        for v in tmp:
            if type(v) == str:
                values.append("'" + v + "'")
            elif isnan(v):
                values.append('null')
            else:
                values.append(str(v))
        if update:
            q = "replace"
        else:
            q = "insert"
        q += " into %s.%s (%s) " % (self.config['db'], table, ', '.join(columns))
        q += "values (%s)" % ", ".join(values)
        return q
    
    def insert_many(self, table, rows, update=True):
        qs = list()
        for row in rows:
            qs.append(self.make_insert_query(table=table, update=update, **row))
        self.query(qs)
    
    def select(self, table, **kwargs):
        q = "select * from %s.%s" % (self.config['db'], table)
        if len(kwargs) > 0:
            q += " where "
            q += self.query_encoder(**kwargs)
        return self.query(q)
    
    def remove(self, table, **kwargs):
        q = "delete from %s.%s" % (self.config['db'], table)
        if len(kwargs) > 0:
            q += " where "
            q += self.query_encoder(**kwargs)
        return self.query(q)
    
    def close(self):
        pass
    
    @staticmethod
    def query_encoder(custom_op=None, **kwargs):
        tokens = list()
        if custom_op is None:
            custom_op = dict()
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


def isnan(value):
    ret = False
    try:
        ret = np.isnan(value)
    except Exception:
        pass
    finally:
        return ret