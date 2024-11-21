import numpy as np
import pandas as pd
from ._helper import _load_file

class LifeexpecSeries(pd.Series):
    @property
    def _constructor(self):
        return LifeexpecSeries
    
    @property
    def _constructor_expanddim(self):
        return LifeexpecDataFrame

class LifeexpecDataFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return LifeexpecDataFrame
    
    @property
    def _constructor_sliced(self):
        return LifeexpecSeries

def lifeexpec():
    data_file = _load_file('lifeexpec.csv')
    
    df = LifeexpecDataFrame(pd.read_csv(data_file))
    df.attrs['cleaned'] = False
    
    return df