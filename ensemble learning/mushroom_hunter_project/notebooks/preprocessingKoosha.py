class PreprocessingKoosha:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
    
    def frequency_encoding(self, target_feature):
        workclass_freq = self.train_data[target_feature].value_counts() / len(self.train_data)
        self.train_data[target_feature] = self.train_data[target_feature].map(workclass_freq)
        self.valid_data[target_feature] = self.valid_data[target_feature].map(workclass_freq).fillna(0)
        self.test_data[target_feature] = self.test_data[target_feature].map(workclass_freq).fillna(0)




    def get_data(self):
        return self.train_data, self.valid_data, self.test_data