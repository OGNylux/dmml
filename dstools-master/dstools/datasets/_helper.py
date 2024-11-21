import os

def _load_file(file):
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, 'data')
    
    return os.path.join(data_path, file)
