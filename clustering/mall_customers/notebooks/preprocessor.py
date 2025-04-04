
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        
    def del_id(self):
        self.df = self.df.drop(columns='CustomerID', axis=1)

    def gender_str_to_num(self):
        self.df['Gender'].replace(['Male', 'Female'], [-1, 1], inplace=True)

    def handle_missing_values (self) :
        self.df.fillna(0, inplace=True)

    def scale_value(self):
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.df)
      
    def transform (self) : 
        self.del_id()
        self.gender_str_to_num()
        self.handle_missing_values()
        self.scale_value()
        return self.df
