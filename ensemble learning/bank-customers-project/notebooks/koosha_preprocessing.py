import pandas as pd
import numpy as np


class PreprocessingKoosha:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def fill_with_proba(self, target_feature):
        feature_probabilities = self.train_data[target_feature].value_counts(normalize=True)
        self.train_data[target_feature] = self.train_data[target_feature].apply(lambda x: np.random.choice(feature_probabilities.index, p=feature_probabilities.values) if pd.isnull(x) else x)
        self.test_data[target_feature] = self.test_data[target_feature].apply(lambda x: np.random.choice(feature_probabilities.index, p=feature_probabilities.values) if pd.isnull(x) else x)

    def getter_df(self):
        return self.train_data, self.test_data
    
    def frequency_encoding(self, target_feature):
        feature_freq = self.train_data[target_feature].value_counts() / len(self.train_data)
        feature_freq_dict = feature_freq.to_dict()  # Convert to dictionary to avoid mapping issues
        self.train_data[target_feature] = self.train_data[target_feature].map(feature_freq_dict)
        self.test_data[target_feature] = self.test_data[target_feature].map(feature_freq_dict).fillna(0)

    def feature_pairwise_corr(self, threshold):
        corr_matrix = self.train_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Fixed here
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        for i in to_drop:
            self.train_data.drop(columns=i, axis=1)
            self.test_data.drop(columns=i, axis=1)