from pymongo import MongoClient
from datk.interface import Interface
from sshtunnel import SSHTunnelForwarder


class MongoDB(Interface):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.ssh = None
        if self.config['sshtunnel'] == 'true':

            self.ssh = SSHTunnelForwarder(self.config['ssh_ip'],
                                          ssh_password=self.config['ssh_pw'],
                                          ssh_username=self.config['ssh_id'],
                                          remote_bind_address=(self.config['ip'], int(self.config['port'])))
            self.ssh.start()
            host = '127.0.0.1'
            port = self.ssh.local_bind_port
        else:
            host = self.config['ip']
            port = int(self.config['port'])

        self.client = MongoClient(host=host,
                                  port=port,
                                  username=self.config['id'],
                                  password=self.config['pw'],
                                  authSource=self.config['auth'])
        self.closed = False
    
    def close(self):
        if not self.closed:
            self.client.close()
            if self.config['sshtunnel'] == 'true':
                if self.ssh is not None:
                    self.ssh.stop()
            self.closed = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
