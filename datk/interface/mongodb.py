from pymongo import MongoClient
from datk.interface.Interface import Interface


class MongoDB(Interface):
    def __init__(self, config):
        self.config = config
        self.ssh = None
        if self.config['sshtunnel'] == 'true':
            from sshtunnel import SSHTunnelForwarder
            self.ssh = SSHTunnelForwarder(self.config['ssh_ip'], ssh_password=self.config['ssh_pw'], ssh_username=self.config['ssh_id'], remote_bind_address=(self.config['ip'], int(self.config['port'])))
            self.ssh.start()
            self.client = MongoClient(host='127.0.0.1', port=self.ssh.local_bind_port, username=self.config['id'], password=self.config['pw'], authSource=self.config['auth'])
        else:
            self.client = MongoClient(host=self.config['ip'], port=int(self.config['port']), username=self.config['id'], password=self.config['pw'], authSource=self.config['auth'])
        self.db = self.client[self.config['db']]
        self.closed = False
        
    def insert_one(self, collection, data, keys, overwrite=True):
        if overwrite:
            ks = dict()
            ds = dict()
            for k in data:
                if k != '_id':
                    if k in keys:
                        ks[k] = data[k]
                    else:
                        ds[k] = data[k]
            self.db[collection].update(ks, {'$set': ds}, upsert=True, multi=False)
        else:
            self.db[collection].insert_one(data)

    def select(self, collection, query):
        return list(self.db[collection].find(query))
    
    def close(self):
        if not self.closed:
            self.client.close()
            if self.config['sshtunnel'] == 'true':
                if self.ssh is not None:
                    self.ssh.stop()
            self.closed = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
