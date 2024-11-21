import numpy as np
import pandas as pd
from ._helper import _load_file

class BodyfatSeries(pd.Series):
    @property
    def _constructor(self):
        return BodyfatSeries
    
    @property
    def _constructor_expanddim(self):
        return BodyfatDataFrame
    

class BodyfatDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return BodyfatDataFrame
    
    @property
    def _constructor_sliced(self):
        return BodyfatSeries
    
    def convert(self, unit = 'imperial'):
        weights = ['Weight']
        lengths = ['Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
        
        df = self
        
        # convert Weight and Height to specified unit
        if unit in ('imperial', 'metric') and df.attrs['unit'] != unit:
            df.attrs['unit'] = unit
            
            if unit == 'metric':
                df[weights] = df[weights].applymap(lambda x: round(x * 0.45359237, 2))
                df[lengths] = df[lengths].applymap(lambda x: round(x / 0.39370, 2))
            else:
                df[weights] = df[weights].applymap(lambda x: round(x / 0.45359237, 2))
                df[lengths] = df[lengths].applymap(lambda x: round(x * 0.39370, 2))
        
        return df
    
    def clean(self):
        df = self
        
        if not df.attrs['cleaned']:
            # drop Density
            df = df.drop(['Density'], axis = 1)
            
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
    
    def for_regression(self):
        # clean df if not cleaned
        if not self.attrs['cleaned']:
            df = self.clean()
        else:
            df = self
        
        X = df.loc[:,df.columns != 'Percent']
        y = df['Percent']

        return X, y


def bodyfat():
    data_file = _load_file('bodyfat.csv')
    to_convert = ['Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
    
    df = BodyfatDataFrame(pd.read_csv(data_file))
    df[['Weight', 'Height']] = df[['Weight', 'Height']].applymap(lambda x: round(x, 2))
    df[to_convert] = df[to_convert].applymap(lambda x: round(x * 0.39370, 2))
    
    df.attrs['cleaned'] = False
    df.attrs['unit'] = 'imperial'
    
    return df
