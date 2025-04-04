import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold




class FilterMethods:
    def __init__(self, dataframe):
        self.df = dataframe 

    def feature_pairwise_corr(self, threshold):
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop
    
    def variance_threshold(self, threshold):
        return [column for column in self.df.columns if self.df[column].var() < threshold]
    
    