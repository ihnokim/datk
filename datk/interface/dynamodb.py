import boto3
from botocore.config import Config
import pandas as pd
from datk.interface import Interface


class DynamoDB(Interface):
    def __init__(self, config):
        self.config = config
        self.client = None
        try:
            self.client = self.connect(config)
        except Exception as e:
            print(e)
            self.client = None
    
    @staticmethod
    def connect(config):
        return boto3.client(service_name=config['service_name'],
                             region_name=config['region_name'],
                             aws_access_key_id=config['aws_access_key_id'],
                             aws_secret_access_key=config['aws_secret_access_key'],
                             endpoint_url=config['endpoint_url'],
                             config=Config(retries=dict(max_attempts=3)))
    
    def list_tables(self, name):
        ret = list()
        if self.client is None:
            return ret
        return self.client.list_tables(ExclusiveStartTableName=name)['TableNames']
    
    def select(self, table, query, columns=None):
        ret_items = list()
        params = {'TableName': table,
                  'KeyConditionExpression': query}
        if columns is not None:
            params['ProjectionExpression'] = ','.join(columns)
        response = self.client.query(**params)
        items = [{y: self.array_parser(x[y]['S']) for y in x} for x in response.get('Items') if 'dummy' not in x]
        
        ret_items += items
        df = pd.DataFrame(ret_items)
        if columns is not None:
            return df[columns]
        else:
            return df
    
    def describe_table(self, table):
        return self.client.describe_table(TableName=table)['Table']
    
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
    
    @staticmethod
    def array_parser(s, delim='?', dtype=str):
        tmp = s.split(delim)
        ret = [dtype(i) for i in tmp]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
