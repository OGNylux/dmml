"""
metrics
==================================
Several metrics for Data Science classification and
regression problems.
"""
from .classification import confusion_matrix
from .classification import to_predictions
from .classification import classificationreport
from .regression import adj_r2

__all__ = ['confusion_matrix', 'to_predictions', 'classificationreport','adj_r2']