class PreprocessingKoosha:
    def __init__(self, data_frame):
        self.df = data_frame

    def fill_with_mode(self, target_column):
        self.df[target_column].fillna(self.df[target_column].mode()[0], inplace=True)

    def get_data(self):
        return self.df