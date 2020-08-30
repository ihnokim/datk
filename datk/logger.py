import json
import logging
from logging.config import dictConfig


def getLogger(config):
    with open(config, 'r') as f:
        config = json.load(f)
        dictConfig(config)
    return logging.getLogger()
