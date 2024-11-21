import numpy as np
import pandas as pd
from ._helper import _load_file

class TitanicSeries(pd.Series):
    @property
    def _constructor(self):
        return TitanicSeries
    
    @property
    def _constructor_expanddim(self):
        return TitanicDataFrame
    

class TitanicDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return TitanicDataFrame
    
    @property
    def _constructor_sliced(self):
        return TitanicSeries
    
    def clean(self, unit = 'imperial'):
        df = self
        
        if not df.attrs['cleaned']:
            # NA: unknown
            for col in ['Cabin', 'Embarked']:
                df[col] = np.where(df[col].isna(), 'Unknown', df[col])
            
            # NA: mean
            df['Age'] = np.where(df['Age'].isna(), df['Age'].mean(), df['Age'])
            
            # column types
            df['Survived'] = df['Survived'].astype('category')
            df['Sex'] = df['Sex'].astype('category')
            df['Embarked'] = df['Embarked'].astype('category')
            
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

    def get_tree_data(self): 
        df = self
        df.to_numeric()

        X = df.drop(['PassengerId','Survived','Sex','Embarked','Name','SibSp','Ticket','Parch', 'Cabin', 'Survived_cat'], axis=1)
        y = df['Survived']

        return X, y

def titanic():
    data_file = _load_file('titanic.csv')
    
    df = TitanicDataFrame(pd.read_csv(data_file, delimiter = ';'))
    df.attrs['cleaned'] = False
    
    return df
