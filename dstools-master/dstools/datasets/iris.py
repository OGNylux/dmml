import numpy as np
import pandas as pd
from ._helper import _load_file

class IrisSeries(pd.Series):
    @property
    def _constructor(self):
        return IrisSeries
    
    @property
    def _constructor_expanddim(self):
        return IrisDataFrame
    

class IrisDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return IrisDataFrame
    
    @property
    def _constructor_sliced(self):
        return IrisSeries
    
    def clean(self, unit = 'imperial'):
        df = self
        
        if not df.attrs['cleaned']:
            # column types
            df['Class'] = df['Class'].astype('category')
            
            # set clean state
            df.attrs['cleaned'] = True
        
        return df
    
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

    def as_ndarray(self):
        """Returns X and y as Numpy-Arrays"""
        X = np.array(self.iloc[:, 0:4])
        y = np.array(self.iloc[:,4])
        return X, y

    def for_logreg(self, species):
        """Returns X and y as Numpy-Arrays in numeric format with only two classes
        Parameter
        ----------
        species: which species to exclude
        0...setosa
        1...virginica
        2...versicolor"""
        df = self
        # Exclude selected species
        if species not in df['Class'].values:
            print(species, " is not a valid species name.")
            return 1
        else:
            df = df[df['Class'] != species]

        X, y = df.as_ndarray()
        X = df.iloc[:, 0:4]
        
        return X, y

def iris():
    data_file = _load_file('iris.csv')
    
    df = IrisDataFrame(pd.read_csv(data_file))
    df.attrs['cleaned'] = False
    
    return df
