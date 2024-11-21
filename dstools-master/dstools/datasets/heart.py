import numpy as np
import pandas as pd
from ._helper import _load_file

class HeartSeries(pd.Series):
    @property
    def _constructor(self):
        return HeartSeries
    
    @property
    def _constructor_expanddim(self):
        return HeartDataFrame

class HeartDataFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return HeartDataFrame
    
    @property
    def _constructor_sliced(self):
        return HeartSeries

def heart():
    data_file = _load_file('heart.csv')
    
    df = HeartDataFrame(pd.read_csv(data_file))
    df.attrs['cleaned'] = False
    
    return df