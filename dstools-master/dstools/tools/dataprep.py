"""
The :mod:`dstools.tools._dataprep` module includes classes and
functions to inspect and prepare a dataset.
"""

# Author: Alexander Adrowitzer <alexander.adrowitzer@fhstp.ac.at>

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def count_outliers(col):
    Q1 = np.percentile(col, 25, method = 'midpoint')
    Q3 = np.percentile(col, 75, method = 'midpoint')
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    out_abs = sum((col < lower_limit) | (col > upper_limit))
    out_rel = round(100*out_abs/len(col),2)
    return out_abs, out_rel

def get_outlier_lists(df):
    res_abs = []
    res_rel = []
    for col in df.columns:        
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):            
            out_abs, out_rel = count_outliers(df[col])
            res_abs.append(out_abs)
            res_rel.append(out_rel)
        else:
            res_abs.append(0)
            res_rel.append(0)
    return res_abs, res_rel

def quality(df):
    """Quality inspection of a DataFrame

        Returns
        -------
        result : DataFrame
            Returns:
                number of rows and columns (formatted)
                datatype of columns
                number of missing values per column (absolute and relative)
                number of unique values per column
                number of outliers per column (0 for non-numeric columns)
        """   
    outlier_abs, outlier_rel = get_outlier_lists(df)
    result = pd.DataFrame(df.dtypes,columns=['type'])
    result['unique'] = pd.DataFrame(df.nunique())
    result['missing_abs'] = pd.DataFrame(df.isna().sum())
    result['missing_rel'] = pd.DataFrame(round(100*df.isna().sum()/len(df),2))
    result['outliers_abs'] = outlier_abs
    result['outliers_rel'] = outlier_rel
    numrows = df.shape[0]
    print("Dataframe has " + f'{numrows:,}' + " rows and " + str(df.shape[1]) + " columns.\n")
    print(str(result[result['missing_abs'] != 0].shape[0]) + " column(s) with missing values.\n")
    print(str(result[result['outliers_abs'] != 0].shape[0]) + " column(s) with outliers.\n")
    
    return result

#def remove_outliers(df):

#    return result

def class_means(X, y):
    """Method to compute the class means. The different classes are encoded in y."""
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means

def standardized_split(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Applies sklearn ``StandardScaler`` to output of 
    sklearn ``train_test_split``.
    """
    sc = StandardScaler()

    xtr, xte, ytr, yte = train_test_split(X, 
        y, 
        test_size=test_size, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=shuffle, 
        stratify=stratify)
    xtr, xte, ytr, yte = train_test_split(X,y)
    xtrain = sc.fit_transform(xtr)
    xtest = sc.transform(xte)
    return xtrain, xtest, ytr, yte

def ztransform(X):
    """Computes z-transformation by formula and not with sklearn ``StandardScaler``."""
    return (X - np.mean(X))/np.std(X)