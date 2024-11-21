import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from ._helper import _load_file

def get_numeric(val):
        # Replace Euro sign. If no suffix (M,K), this is the result
        res = val.replace('€', '')
        if 'M' in val:
            res = float(res.replace('M', ''))*1000000
        elif 'K' in val:
            res = float(res.replace('K', ''))*1000
        return float(res)

class FifaSeries(pd.Series):
    @property
    def _constructor(self):
        return FifaSeries
    
    @property
    def _constructor_expanddim(self):
        return FifaDataFrame
    

class FifaDataFrame(pd.DataFrame):    
    @property
    def _constructor(self):
        return FifaDataFrame
    
    @property
    def _constructor_sliced(self):
        return FifaSeries
    
    def convert(self, unit = 'imperial'):
        weights = ['Weight']
        lengths = ['Height']
        
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

    def get_position_info(self):
        """Returns a description of players positions
        """
        data = {'Column':['GK',
                        'LWB','LCB','LB','CB','RB','RCB','RWB',
                        'LDM','CDM','RDM','LM','LCM','CM','RCM','RM','LAM','CAM','RAM',
                        'LF','CF','RF','LW','RW','RS','ST','LS'],
                'Name': ['Goalkeeper',
                        'Left Wing Back','Left Center Back','Left Back','Center Back','Right Back','Right Center Back','Right Wing Back',
                        'Left Defensive Midfielder','Center Defensive Midfielder','Right Defensive Midfielder',
                        'Left Midfielder','Left Center Midfielder','Center Midfielder','Right Center Midfielder','Right Midfielder',
                        'Left Attacking Midfielder','Center Attacking Midfielder','Right Attacking Midfielder',
                        'Left Forward','Center Forward','Right Forward','Left Wing','Right Wing',
                        'Right Striker','Striker','Left Striker'],
                'Position': ['Torwart',
                             'linker Außenverteidiger','linker Innenverteidiger','linker Verteidiger','Innenverteidiger','rechter Verteidiger','rechter Innenverteidiger','rechter Außenverteidiger',
                             'linkes, defensives Mittelfeld','zentrales, defensives Mittelfeld','rechtes, defensives Mittelfeld',
                             'linkes Mittelfeld','halblinkes Mittelfeld','zentrales Mittelfeld','halbrechtes Mittelfeld','rechtes Mittelfeld',
                             'linkes, offensives Mittelfeld','zentrales, offensives Mittelfeld','rechtes, offensives Mittelfeld',
                             'Linksaußen','Mittelstürmer','Rechtsaußen','linker Flügel','rechter Flügel',
                             'rechter Stürmer','Stürmer','linker Stürmer'],
                'Area': ['Goal',
                        'Defense','Defense','Defense','Defense','Defense','Defense','Defense',
                        'Midfield','Midfield','Midfield','Midfield','Midfield','Midfield','Midfield','Midfield','Midfield','Midfield','Midfield',
                        'Attack','Attack','Attack','Attack','Attack','Attack','Attack','Attack']}
        
        return DataFrame(data)

    def get_column_description(self, colname):
        """Returns information about the columns in the clean dataset"""
        if colname not in self.columns:
            print("Column ", colname, " does not exist.")
            return
        return colname

    def convert_position_data(self):
        df = self

        positions = df.get_position_info()
        pos_columns = positions["Column"]

        for pos in pos_columns:
            if pos != 'GK':
                df[pos] = df[pos].fillna('0+0')
                df[pos] = df[pos].str.split("+", expand=True)[0]
                df[pos] = df[pos].astype(int)

        return df

    def clean(self):
        df = self
        
        if not df.attrs['cleaned']:
            # drop columns 'Row', 'ID', 'Photo', 'Flag', 'Club Logo', 'Real Face'
            df = df.drop(['Row', 'ID', 'Photo', 'Flag', 'Club Logo', 'Real Face'], axis = 1)
            
            # Fill NaN in column "Release Clause" with "€-1"
            df["Release Clause"] = df["Release Clause"].fillna("€-1")

            # Convert columns 'Value', 'Wage' and 'Release Clause' to numeric
            df["Value"] = df['Value'].apply(lambda x: get_numeric(x))
            df["Wage"] = df["Wage"].apply(lambda x: get_numeric(x))
            df["Release Clause"] = df["Release Clause"].apply(lambda x: get_numeric(x))

            # Convert positional columns (LS, ST, RS,...) to numeric
            df = df.convert_position_data()

            # Drop rows with missing values for "Preferred Foot"
            df.dropna(subset = ['Preferred Foot'], inplace=True)

            # Drop columns "Joined" and "Loaned From" (very unformatted values)
            df.drop(["Joined","Loaned From"], axis=1, inplace=True)

            # Todo: Add conversion of height and weight to convert-method
            df['Height'] = round(df['Height'].apply(lambda x: float(x.split("'")[0])*30.48+float(x.split("'")[1])*2.54),0)
            df['Weight'] = round(df['Weight'].apply(lambda x: round(float(x.split('lbs')[0])*0.45359237, 2)),1)

            # Convert to datetime
            df['Contract Valid Until'] = df['Contract Valid Until'].apply(lambda x: pd.to_datetime(x))
            

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


def fifa():
    data_file = _load_file('fifa.csv')
    #to_convert = ['Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
    
    df = FifaDataFrame(pd.read_csv(data_file))
    #df[['Weight', 'Height']] = df[['Weight', 'Height']].applymap(lambda x: round(x, 2))
    #df[to_convert] = df[to_convert].applymap(lambda x: round(x * 0.39370, 2))
    
    df.attrs['cleaned'] = False
    df.attrs['unit'] = 'imperial'
    
    return df
