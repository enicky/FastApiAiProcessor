import logging, coloredlogs
from os import listdir, getcwd
import os
from os.path import isfile, join

logger = logging.getLogger(__name__)
coloredlogs.install("INFO", logger = logger)



class Models():
    def __init__(self) -> None:
        pass
    
    def list_models(self):
        MODELS_PATH = os.getenv('AI_MODELS_PATH')
        path = 'models/' if MODELS_PATH is None else MODELS_PATH
        logger.debug(f'Using the following PATH : {path}')
        onlyfiles = [f for f in listdir(path=path) if isfile(join(path, f))]
        onlyfiles = list(filter(lambda k: '.pt' in k, onlyfiles))
        return onlyfiles
