import numpy as np
import pandas as pd

def vif(df, threshold = 5, exclude = [], drop = False):
    """Returns the Variance Inflation Factor for every feature in descending order.
    
    Parameters:
    -----------
    df: Pandas DataFrame
    threshold: default is 5
    """
    df_vif = pd.DataFrame()

    df_vif['columns'] = df.columns
    df_vif['vif'] = np.diag(np.linalg.inv(df.corr()))
    
    df_vif = df_vif[df_vif['vif'] >= threshold].sort_values(by = ['vif'], ascending = False)
    
    if exclude:
        if type(exclude) in [list, tuple, set]:
            cols = set(df_vif['columns'].to_list()) - set(exclude)
        elif isinstance(exclude, str):
            cols = set(df_vif['columns'].to_list())
            
            if exclude in cols:
                cols.remove(exclude)
            else:
                raise KeyError(f'\'{exclude}\' is not a column in DataFrame!')
    else:
        cols = set(df_vif['columns'].to_list())
    
    if drop:
        df_vif = df.drop(cols, axis = 1)
    else:
        df_vif = df_vif[df_vif['columns'].isin(cols)]
    
    return df_vif
