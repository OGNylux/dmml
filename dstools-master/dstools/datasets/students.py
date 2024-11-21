import numpy as np
import pandas as pd
from ._helper import _load_file

class StudentsSeries(pd.Series):
    @property
    def _constructor(self):
        return StudentsSeries
    
    @property
    def _constructor_expanddim(self):
        return StudentsDataFrame
    

class StudentsDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return StudentsDataFrame
    
    @property
    def _constructor_sliced(self):
        return StudentsSeries
    
    def clean(self, unit = 'imperial'):
        df = self
        
        if not df.attrs['cleaned']:
            # NA: unknown
            for col in ['Gender', 'Eyes', 'Hair', 'Smoker', 'EduMother', 'EduFather', 'SmokeMother', 'SmokeFather', 'ZodiacSign']:
                df[col] = np.where(df[col].isna(), 'Unknown', df[col].str.title())
            
            # NA: mean by gender
            for col in ['Weight', 'Size', 'Shoesize']:
                df[col] = df[col].fillna(df.groupby('Gender')[col].transform('mean'))
            
            # NA: median
            for col in ['Mathgrade', 'Germangrade', 'Englishgrade']:
                df[col] = np.where(df[col].isna(), df[col].median(), df[col])
            
            # NA: mean
            df['SizeMother'] = np.where(df['SizeMother'].isna(), df['SizeMother'].mean(), df['SizeMother'])
            df['SizeFather'] = np.where(df['SizeFather'].isna(), df['SizeFather'].mean(), df['SizeFather'])
            
            # column types
            df['Gender'] = df['Gender'].astype('category')
            df['Eyes'] = df['Eyes'].astype('category')
            df['Hair'] = df['Hair'].astype('category')
            df['Smoker'] = df['Smoker'].astype('category')
            df['EduMother'] = df['EduMother'].astype('category')
            df['SmokeMother'] = df['SmokeMother'].astype('category')
            df['EduFather'] = df['EduFather'].astype('category')
            df['SmokeFather'] = df['SmokeFather'].astype('category')
            df['ZodiacSign'] = df['ZodiacSign'].astype('category')
            
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


def students():
    data_file = _load_file('students.csv')
    
    df = StudentsDataFrame(pd.read_csv(data_file, delimiter = ';'))
    df.attrs['cleaned'] = False
    
    return df
