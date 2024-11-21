import numpy as np
import pandas as pd
from ._helper import _load_file

class FishSeries(pd.Series):
    @property
    def _constructor(self):
        return FishSeries
    
    @property
    def _constructor_expanddim(self):
        return FishDataFrame
    

class FishDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return FishDataFrame
    
    @property
    def _constructor_sliced(self):
        return FishSeries
    
    def clean(self, unit = 'imperial'):
        df = self
        
        if not df.attrs['cleaned']:
            # column types
            df['Art'] = df['Art'].astype('category')
            
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

    def for_logreg(self):
        df = self
        y_cat = df["Art"]
        df = df.to_numeric()
        X = df.iloc[:,1:]
        y = df["Art_cat"]

        return X, y, y_cat

    def for_kmeans(self, scaled=True):
        """Returns datasets for kMeans Clustering
            X_kmeans: Features for kMeans Clustering
            X_desc: Features to describe the result

        Parameters:
        scaled: True: Applies z-Transformation to X_kmeans
        """
        df = self
    
        X_kmeans = df.iloc[:,1:]
        X_desc = df.iloc[:,0:1]

        if scaled == True:
            X_kmeans =  (X_kmeans - X_kmeans.mean(axis=0)) / (X_kmeans.std(axis=0))
        return X_kmeans, X_desc

def fish():
    data_file = _load_file('fish.csv')
    
    df = FishDataFrame(pd.read_csv(data_file, delimiter = ','))
    df.attrs['cleaned'] = False
    
    return df
