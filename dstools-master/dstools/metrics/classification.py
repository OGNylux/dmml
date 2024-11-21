"""
The :mod:`dstools.metrics._classification` module includes classes and
functions to calculate and display classification metrics.

The code uses parts from sklearn library.
"""

# Author: Alexander Adrowitzer <alexander.adrowitzer@fhstp.ac.at>

import itertools
import numpy as np
import pandas as pd

def create_target_names(conf_mat):
    labels = []
    for i in range(1,conf_mat.shape[0]+1):
        labels.append("Class " + str(i))
    return labels

def to_predictions(ser, thresh):
    """Turns the probability values in a Series/column to probabilities based on a threshold.
    The resulting series can be used as y_pred in the confusion matrix

    Parameters:
    -----------
    ser: A Pandas Series with probability values
    thresh: Threshold to decide when predictions belong to class 0 or 1 
    """

    result = ser.apply(lambda x: np.where(x < thresh, 0, 1))

    return result
    
def confusion_matrix(y_pred, y_true, sums=False):
    """Displays the confusion matrix in BDS standard format.
    
    Parameters:
    -----------
    y_true: Actual classes (e.g. from testset) 
    y_pred: Predicted classes
    sums: Show row and column sums
    """
    cfm = pd.crosstab(y_pred, y_true, rownames=['Predicted'], colnames=['Actual'])
    cfm_vals = cfm.copy()
    if sums:
        cfm['Sums'] = np.sum(cfm_vals, axis=1)
        cfm.loc['Sums'] = np.append(np.sum(cfm_vals, axis=0).values,np.sum(np.sum(cfm_vals)))
        return cfm
    else:
        return cfm 

def accuracy(conf_mat):
    """Computes the accuracy of a NxN confusion matrix"""

    TP = np.sum(np.diag(conf_mat))
    N = np.sum(np.sum(conf_mat))
    return np.sum(TP/N)

def precision(conf_mat):
    return np.diag(conf_mat)/np.sum(conf_mat,axis=0)

def recall(conf_mat):
    return np.diag(conf_mat)/np.sum(conf_mat,axis=1)

def f1score(conf_mat):
    pre = precision(conf_mat)
    rec = recall(conf_mat)
    return 2 * (pre*rec)/(pre + rec) 

def precision_recall_f1(conf_mat):
    return precision(conf_mat), recall(conf_mat), f1score(conf_mat)

def macro_averages(conf_mat):
    mac_pre = np.sum(precision(conf_mat))/len(precision(conf_mat))
    mac_rec = np.sum(recall(conf_mat))/len(recall(conf_mat))
    mac_f1 = np.sum(f1score(conf_mat))/len(f1score(conf_mat))

    return mac_pre, mac_rec, mac_f1

def weighted_precision_recall_f1(conf_mat):
    support = np.sum(conf_mat, axis=1)
    weighted_pre = np.sum(support * precision(conf_mat))/np.sum(support)
    weighted_rec = np.sum(support * recall(conf_mat))/np.sum(support)
    weighted_f1 = np.sum(support * f1score(conf_mat))/np.sum(support)

    return weighted_pre, weighted_rec, weighted_f1

def chance_agree(conf_mat):
   prob_cols = np.sum(conf_mat, axis=0)/np.sum(np.sum(conf_mat))
   prob_rows = np.sum(conf_mat, axis=1)/np.sum(np.sum(conf_mat))
   return sum(prob_cols*prob_rows) 

def cohen_kappa(conf_mat):
    agree = accuracy(conf_mat)
    return (agree - chance_agree(conf_mat))/(1-chance_agree(conf_mat))

def classificationreport(conf_mat, labels=None, dec=3):
    """Returns an advanced classification report for a confusion matrix

    Parameters:
    conf_mat:   A nxn 2-dimensional confusion matrix 
                with predictions in rows and actual values in columns
    dec:        Number of decimals to display result
    """
    colnames = conf_mat.columns.values
    if colnames[-1] == 'Sums':
        conf_mat = conf_mat.drop('Sums', axis=0)
        conf_mat = conf_mat.drop('Sums', axis=1)
        
    if conf_mat.ndim != 2:
        raise ValueError(
                "Confusion Matrix must be 2 dimensional."
                )
    if conf_mat.shape[0] != conf_mat.shape[1]:
        raise ValueError(
            "Confusion Matrix must have same number of rows and columns."
            )

    headers = ["Precision","Recall", "F1", "Support"]
    if labels==None:
        target_names = create_target_names(conf_mat)
    else:
        target_names = labels
    pre, rec, f1 = precision_recall_f1(conf_mat)
    support = np.sum(conf_mat, axis=1)
    rows = zip(target_names, pre, rec, f1, support)

    mac_pre, mac_rec, mac_f1 = macro_averages(conf_mat)
    w_pre, w_rec, w_f1 = weighted_precision_recall_f1(conf_mat)
    N = np.sum(conf_mat)
    
    acc = accuracy(conf_mat)
    kappa = cohen_kappa(conf_mat)
    
    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), dec)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=dec)
    report += "\n"

    report += row_fmt.format("macro avg",mac_pre, mac_rec, mac_f1,np.sum(N),width=width, digits=dec)
    report += row_fmt.format("weighted avg",w_pre, w_rec, w_f1,np.sum(N),width=width, digits=dec)
    report += "\n"

    row_fmt_overall = "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
    report += row_fmt_overall.format("Accuracy", "","",acc,np.sum(N), width=width, digits=dec)
    report += row_fmt_overall.format("Cohen Kappa", "","", kappa, np.sum(N),width=width, digits=dec)
    return report