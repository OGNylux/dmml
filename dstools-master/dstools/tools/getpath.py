import os
import inspect

def getpath(*args):
    current_path = os.getcwd()
    extend_path = []
    
    for p in args:
        if not isinstance(p, (str, list, tuple, set)):
            raise Exception(f'getpath - argument {p}: Please provide a string or a list/tuple/set of strings!')
        else:
            extend_path.extend([p] if isinstance(p, str) else [str(elem) for elem in p])
            
    return os.path.join(current_path, *extend_path)
