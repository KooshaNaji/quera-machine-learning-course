import numpy as np

class SearchNullValues:
    def __init__(self, dataframe):
        self.df = dataframe
        self.have_null = None
    
    def col_have_na_to_arr(self):
        columns_have_null = self.df.isnull().any()
        self.have_null = columns_have_null[columns_have_null].index.tolist()
        print(self.have_null)
        return self.have_null




class FillNullsWithTargetColumn():
    def __init__(self, dataframe, target_column):
        self.df = dataframe
        self.means = None
        self.target_column = target_column

    def calculate_means_by_target_column(self):
        # Calculate means for each numeric column grouped by 'target colomn'
        self.means = self.df.groupby(self.target_column).mean(numeric_only=True)

    def fill_nulls_by_columns(self, column: str):
        if self.means is None:
            self.calculate_means_by_target_column()

        # Fill NaN values using precomputed means
        self.df[column] = self.df.groupby(self.target_column)[column].transform(
            lambda x: x.fillna(self.means.loc[x.name, column])
        )

    def get_means_df(self):
        self.calculate_means_by_target_column(self)
        return self.means







class StandardizeForEachFeature:
    def __init__(self, train_data, test_data, validation_data, scaler_obj):
        self.train = train_data
        self.test = test_data
        self.valid = validation_data
        self.scaler = scaler_obj

    def standardize_train(self, train, scaler):
        return scaler.fit_transform(train)
    
    def standardize_test(self, test, scaler):
        return scaler.transform(test)
    
    def standardize_valid(self, valid, scaler):
        return scaler.transform(valid)
    
    def standardize_all(self):
        self.train = self.standardize_train(self.train, self.scaler)
        self.test = self.standardize_test(self.test, self.scaler)
        self.valid = self.standardize_valid(self.valid, self.scaler)
        return self.train, self.test, self.valid
    

class FillNullsWithMedians:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.medians = self.df.median()

    def fill_null_median(self):
        self.df.fillna(self.medians, inplace=True)

    def get_filled_data(self):
        return self.df
    

class FS_PairwiseCorr:
    def __init__(self, dataframe, threshold:float):
        self.df = dataframe
        self.threshold = threshold
        self.selected = None
    
    def select_features(self):
        
        # computing the absolute values of correlation
        corr_matrix = self.df.corr().abs()

        # keeping only upper part of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # finding the highly correlated features
        THRESHOLD = 0.90
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > THRESHOLD)]
        print('Feature(s) to drop:', ', '.join(to_drop))

        # dropping the highly correlated features
        self.selected = self.df.drop(to_drop, axis=1)

        return self.selected

