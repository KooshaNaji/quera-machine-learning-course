import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class PreprocessingKoosha:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
    
    def frequency_encoding(self, target_feature):
        workclass_freq = self.train_data[target_feature].value_counts() / len(self.train_data)
        self.train_data[target_feature] = self.train_data[target_feature].map(workclass_freq)
        self.valid_data[target_feature] = self.valid_data[target_feature].map(workclass_freq)
        self.test_data[target_feature] = self.test_data[target_feature].map(workclass_freq)

        return self.train_data, self.valid_data, self.test_data
        


    def show(self, columns):
        num_features = len(columns) 
        fig, axes = plt.subplots(num_features, 2, figsize=(15, 40))  

        for i, col in enumerate(columns):
            
            
            sns.histplot(self.train_data[col], bins=30, kde=True, ax=axes[i, 0], color="blue", alpha=0.6)
            sns.histplot(self.test_data[col], bins=30, kde=True, ax=axes[i, 1], color="red", alpha=0.6)

            axes[i, 0].set_title(f"Train - {col}", fontsize=12)
            axes[i, 1].set_title(f"Test - {col}", fontsize=12)

            axes[i, 0].set_xlabel(col)
            axes[i, 1].set_xlabel(col)

        plt.tight_layout()  
        plt.show()


    def standardize(self):
        scaler = StandardScaler()
        self.train_data = scaler.fit_transform(self.train_data)
        self.valid_data = scaler.transform(self.valid_data)
        self.test_data = scaler.fit_transform(self.test_data)

        return self.train_data, self.valid_data, self.test_data
