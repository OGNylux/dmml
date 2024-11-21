import numpy as np
import pandas as pd
from ._helper import _load_file

class ShoppingSeries(pd.Series):
    @property
    def _constructor(self):
        return ShoppingSeries
    
    @property
    def _constructor_expanddim(self):
        return ShoppingDataFrame
    

class ShoppingDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return ShoppingDataFrame
    
    @property
    def _constructor_sliced(self):
        return ShoppingSeries
    
    def clean(self, unit = 'imperial'):
        return self
    
    def to_numeric(self):
        # clean df if not cleaned
        if not self.attrs['cleaned']:
            df = self.clean()
        else:
            df = self
        
        cat = df.select_dtypes(['category']).columns
        cat_codes = [x + '_cat' for x in cat]
        
        df[cat_codes] = df[cat].apply(lambda x: x.cat.codes)
        
        df = df.select_dtypes(include = ['number'])
        
        return df

    def for_regression(self):
        # clean df if not cleaned
        if not self.attrs['cleaned']:
            df = self.clean()
        else:
            df = self

        df.drop("Gender", axis=1, inplace=True)

        return df


def shopping():
    """
    Shopping frequency
    
    Description
    -----------
    The Shopping data frame has 40 rows and 4 columns.

    This data frame contains the following columns:
    - Shopping: Number of shopping activities per month
    - Age: Age in years
    - Gender: Gender (1 = male, 0 = female)
    - Income: Monthly income in [1000 EUR]
    """

    data_file = _load_file('shopping.csv')
    
    df = ShoppingDataFrame(pd.read_csv(data_file, delimiter = ';'))
    df.attrs['cleaned'] = False
    
    return df