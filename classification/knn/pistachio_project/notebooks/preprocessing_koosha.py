from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




class PreprocessingKoosha:
    def __init__(self, train_data, test_data, validation_data):
        self.train = train_data
        self.test = test_data
        self.valid = validation_data

    def feature_pairwise_corr(self, threshold):
        corr_matrix = self.train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print("Features to drop due to high correlation:", to_drop)
        return self.train.drop(columns=to_drop), self.test.drop(columns=to_drop), self.valid.drop(columns=to_drop)
    
    def variance_threshold(self, threshold):
        selector = VarianceThreshold(threshold)
        selector.fit(self.train)
        features_kept = self.train.columns[selector.get_support(indices=True)]
        features_deleted = self.train.columns[~selector.get_support()]
        print("Features kept:", features_kept)
        print("Features deleted:", features_deleted)   
        return selector.fit_transform(self.train)

    

    def standardize(self):
        scaler = StandardScaler()
        self.train = scaler.fit_transform(self.train)
        self.test = scaler.transform(self.test)
        self.valid = scaler.transform(self.valid)
        return self.train, self.test, self.valid
    


    